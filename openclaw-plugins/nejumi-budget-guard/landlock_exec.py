#!/usr/bin/env python3
"""Run a command with a task-scoped Landlock filesystem policy."""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path


LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
PR_SET_NO_NEW_PRIVS = 38

LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12

READ_ACCESS = (
    LANDLOCK_ACCESS_FS_EXECUTE
    | LANDLOCK_ACCESS_FS_READ_FILE
    | LANDLOCK_ACCESS_FS_READ_DIR
)
WRITE_ACCESS = (
    LANDLOCK_ACCESS_FS_WRITE_FILE
    | LANDLOCK_ACCESS_FS_REMOVE_DIR
    | LANDLOCK_ACCESS_FS_REMOVE_FILE
    | LANDLOCK_ACCESS_FS_MAKE_CHAR
    | LANDLOCK_ACCESS_FS_MAKE_DIR
    | LANDLOCK_ACCESS_FS_MAKE_REG
    | LANDLOCK_ACCESS_FS_MAKE_SOCK
    | LANDLOCK_ACCESS_FS_MAKE_FIFO
    | LANDLOCK_ACCESS_FS_MAKE_BLOCK
    | LANDLOCK_ACCESS_FS_MAKE_SYM
)
HANDLED_ACCESS = READ_ACCESS | WRITE_ACCESS

# x86_64 syscall numbers. The benchmark image is currently x86_64; fail closed
# on any other architecture rather than silently running without isolation.
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int),
        ("reserved", ctypes.c_uint),
    ]


def _syscall(libc: ctypes.CDLL, number: int, *args: object) -> int:
    result = int(libc.syscall(number, *args))
    if result < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return result


def _existing_roots(values: list[str]) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for raw in values:
        path = Path(raw).resolve(strict=False)
        if not path.exists():
            continue
        text = str(path)
        if text not in seen:
            roots.append(path)
            seen.add(text)
    return roots


def _add_path_rule(
    libc: ctypes.CDLL,
    ruleset_fd: int,
    path: Path,
    access: int,
) -> None:
    path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        attr = PathBeneathAttr(allowed_access=access, parent_fd=path_fd)
        _syscall(
            libc,
            SYS_LANDLOCK_ADD_RULE,
            ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(attr),
            0,
        )
    finally:
        os.close(path_fd)


def apply_landlock(read_only: list[Path], read_write: list[Path]) -> None:
    if os.uname().machine != "x86_64":
        raise RuntimeError(
            f"unsupported architecture for Landlock launcher: {os.uname().machine}"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    abi = _syscall(
        libc,
        SYS_LANDLOCK_CREATE_RULESET,
        0,
        0,
        LANDLOCK_CREATE_RULESET_VERSION,
    )
    if abi < 1:
        raise RuntimeError(f"Landlock ABI 1 or newer is required, found {abi}")

    ruleset_attr = RulesetAttr(handled_access_fs=HANDLED_ACCESS)
    ruleset_fd = _syscall(
        libc,
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        0,
    )
    try:
        for path in read_only:
            _add_path_rule(libc, ruleset_fd, path, READ_ACCESS)
        for path in read_write:
            _add_path_rule(libc, ruleset_fd, path, READ_ACCESS | WRITE_ACCESS)

        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        _syscall(libc, SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0)
    finally:
        os.close(ruleset_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--tmp", required=True)
    parser.add_argument("--home", required=True)
    parser.add_argument("--read-only", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve(strict=True)
    private_tmp = Path(args.tmp).resolve(strict=True)
    private_home = Path(args.home).resolve(strict=True)

    system_roots = _existing_roots(
        [
            "/bin",
            "/sbin",
            "/usr",
            "/lib",
            "/lib64",
            "/etc",
            "/dev",
            "/sys",
            *args.read_only,
        ]
    )
    apply_landlock(
        read_only=system_roots,
        read_write=[workspace, private_tmp, private_home],
    )
    os.chdir(workspace)
    os.execvpe(args.command[0], args.command, os.environ)
    return 127


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"NEJUMI_WORKSPACE_GUARD_LAUNCH_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(126)

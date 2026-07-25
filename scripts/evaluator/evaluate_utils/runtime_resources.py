from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FileDescriptorSnapshot:
    open_count: Optional[int]
    soft_limit: Optional[int]
    hard_limit: Optional[int]

    def format(self) -> str:
        open_text = "unknown" if self.open_count is None else str(self.open_count)
        soft_text = "unknown" if self.soft_limit is None else str(self.soft_limit)
        hard_text = "unknown" if self.hard_limit is None else str(self.hard_limit)
        return f"open={open_text}, soft_limit={soft_text}, hard_limit={hard_text}"


def _open_file_descriptor_count() -> Optional[int]:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return None


def file_descriptor_snapshot() -> FileDescriptorSnapshot:
    try:
        import resource

        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ImportError, OSError, ValueError):
        soft_limit = None
        hard_limit = None
    return FileDescriptorSnapshot(
        open_count=_open_file_descriptor_count(),
        soft_limit=soft_limit,
        hard_limit=hard_limit,
    )


def ensure_file_descriptor_capacity(
    minimum_soft_limit: int = 8192,
) -> FileDescriptorSnapshot:
    """Raise a low process FD soft limit without exceeding the host hard limit."""
    target = max(0, int(minimum_soft_limit))
    try:
        import resource

        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        if target > soft_limit:
            new_soft_limit = min(target, hard_limit)
            if new_soft_limit > soft_limit:
                resource.setrlimit(
                    resource.RLIMIT_NOFILE,
                    (new_soft_limit, hard_limit),
                )
    except (ImportError, OSError, ValueError) as exc:
        print(
            "Warning: unable to raise the process file-descriptor limit: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
    return file_descriptor_snapshot()


def log_file_descriptor_snapshot(label: str) -> FileDescriptorSnapshot:
    snapshot = file_descriptor_snapshot()
    print(f"Process file descriptors [{label}]: {snapshot.format()}", flush=True)
    return snapshot

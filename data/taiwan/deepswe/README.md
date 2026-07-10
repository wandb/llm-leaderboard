# DeepSWE Taiwan subset manifest

This directory contains fixed DeepSWE task-name subsets for the Taiwan leaderboard. It does not copy task source files; Pier reads the canonical `external/deep-swe/tasks` tree at evaluation time.

Additional `essential_*` subsets are selected from the public DeepSWE v1.1
rollout matrix to approximate full 113-task rank signal at lower cost. See
`docs/deepswe_essential_subset_selection.md` for the selection method and
validation metrics.

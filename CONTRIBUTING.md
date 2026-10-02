# Contributing

Create a fresh environment and install `python -m pip install -e ".[test]"`.
Run `python -m pytest -q` before submitting changes. Keep fixes small and explain
the behavior being changed and how it was verified.

Changes to losses, step adaptation, candidate selection, seed handling, stopping
rules, or iteration counts change the method. Do not silently change defaults
or reuse historical scores for a modified implementation. Add a regression test
and report both attack success and full computational cost.

Include model preprocessing, raw perturbation constraints, dataset split, seeds,
library versions and a minimal reproducer in bug reports. For incorrect success
flags, include a small synthetic classifier if possible. Do not attach private
datasets or large checkpoints to a source-code change.

Keep generated data, checkpoints, local environments and large results out of
Git. Update dependency notices if introducing third-party code. The GitHub
workflow belongs at the root of the repository made from this FARO directory.

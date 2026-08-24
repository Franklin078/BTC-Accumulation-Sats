# Environments

## Main (daily work)
Python 3.12+ with `requirements.txt`. Used by the analysis notebooks locally and the Colab runtime.

## Frozen tournament check (venv-py310)
Python 3.10.11 with `tournament_2025/requirements_tournament_py310.txt` (the 2025 tournament's pinned list, unmodified), plus `jupyter` and `nbconvert` for execution, registered as the Jupyter kernel `ss-py310`.

One Windows-specific note: the pinned list does not pin `torch` (it is only a transitive dependency of `pgmpy`). Recent torch builds fail to initialise on Windows when imported after scikit-learn 1.4.2, which is the import order the tournament Prelude uses, so this environment pins `torch==2.4.1`, under which the full Prelude import order works. The tournament's own evaluation ran on Linux, where the conflict does not occur; no pinned package version is altered.

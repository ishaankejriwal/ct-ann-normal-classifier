# Entrypoint for ANN vs NORMAL model training.

from __future__ import annotations

import warnings

# Import the package-level training runner.
from ann_normal_training.training import run_cross_validation

warnings.filterwarnings("ignore")


def main() -> None:
    # Execute the full cross-validation training workflow.

    # Delegate orchestration to the training package.
    run_cross_validation()


if __name__ == "__main__":
    main()

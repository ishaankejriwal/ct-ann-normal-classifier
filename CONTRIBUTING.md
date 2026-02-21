# Contributing Guide

## Workflow

- Create a feature branch from `main`.
- Keep each pull request focused on one change set.
- Write clear commit messages in imperative style.
- Link relevant issues in pull request descriptions.

## Development Setup

1. Create and activate a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Verify imports and syntax before opening a pull request.

```powershell
pip install -r requirements.txt
python -m py_compile "train.py" "ann_normal_training/__init__.py" "ann_normal_training/config.py" "ann_normal_training/logging_utils.py" "ann_normal_training/model.py" "ann_normal_training/dataset.py" "ann_normal_training/evaluation.py" "ann_normal_training/training.py"
```

## Code Standards

- Use clear names and small, testable functions.
- Keep comments concise and single-line where possible.
- Preserve existing data formats and output paths.
- Avoid unrelated refactors in the same pull request.

## Validation Checklist

Before submitting:

- Ensure code compiles without errors.
- Confirm the entrypoint runs: `python train.py`.
- Confirm no large artifacts are staged (`*.pt`, `*.pth`, logs, generated videos).

## Pull Request Checklist

- [ ] Scope is limited and clearly described.
- [ ] New/changed behavior is explained.
- [ ] Local validation steps and results are included.
- [ ] No secrets or local paths are committed.

## Data and Model Artifacts

- Do not commit datasets or generated frame directories.
- Do not commit model checkpoints unless explicitly requested.
- Share large artifacts via release assets or external storage links.

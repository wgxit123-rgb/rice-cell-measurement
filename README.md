# Source code for conditional cell-area error and instance recovery

This source snapshot accompanies *Conditional Cell-Area Error and Instance Recovery Can Diverge: An Empirical Evaluation*. It contains the rice representation model and development measurement code, the StomataSeg adaptation and evaluation implementation, an independent saved-scalar checker, and selected synthetic tests. The manifest lists the SHA-256 of each source file.

The source-only archive excludes images, annotations, model weights, predictions, experiment results, manuscript files, and historical run metadata. The adaptation and scoring entry points require data and version-specific input bindings. This archive allows code inspection and synthetic checks. Full model replay requires the original authorized inputs and environment.

## Package and test entry points

Use Python 3.10 or later. The project declares dependencies in `pyproject.toml` and `requirements.txt`.

```sh
python -m pip install -e .
python -m pip install pytest
python -m pytest tests/test_instance_metrics.py tests/test_measurement_fidelity.py engineering/stomataseg_v1/tests/test_measurement.py
python research/stomataseg_execution_20261026/scalar_checker/checker.py --help
```

The StomataSeg engine is under `engineering/stomataseg_v1`. The JSON schema there documents configuration fields. The separate fixed-study scoring entry point is under `research/stomataseg_execution_20261026/analysis_code`. The recorded protocol applies Hungarian matching before filtering matches at inclusive IoU thresholds. Its positive-reference endpoints leave unpaired predictions outside the reference-area residual.

## Dependencies and external source

The software calls NumPy, SciPy, Pillow, PyTorch, torchvision, scikit-image, OpenCV and PyYAML. Each dependency is installed separately under its own terms. Original StomataSeg images and COCO annotations are available from the third-party [StomataSeg v1 record](https://doi.org/10.5281/zenodo.18216859).

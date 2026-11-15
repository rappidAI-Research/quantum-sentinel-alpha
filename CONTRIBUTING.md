# Contributing

Quantum Sentinel-Alpha is research software in development. Start with the
[project status](docs/STATUS.md), [foundation decision](docs/FOUNDATION.md) and
[data-governance rules](docs/DATA_GOVERNANCE.md).

For an offline development environment, use Python 3.11 or newer:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m sentinel.foundation
python -m pytest -q
```

Model, quantization and training extras are optional; the default validation
does not require GPU time or model downloads. Keep changes focused and include
the affected checks and evidence in the pull request. Do not commit upstream
source excerpts, credentials, model weights or generated run artifacts.

Approval templates and synthetic evaluation fixtures are intentional project
assets. A template, detected license or passing test never authorizes training.
Dataset approvals and paid workloads require separate maintainer decisions.

The repository has no declared project-source license. This contribution guide
does not grant reuse rights or change upstream licensing terms.

Report vulnerabilities through [SECURITY.md](SECURITY.md).

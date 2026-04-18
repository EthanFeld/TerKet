# Contributing

## Development Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements.txt
pytest
```

## Repo Conventions

- Put package code under `src/terket/`
- Keep benchmark entrypoints under `benchmarks/`
- Write generated benchmark CSVs to `results/`
- Add or update tests with behavior changes
- Preserve optional-dependency behavior for native, quimb, and cotengra paths

## Benchmarks

Use unified entrypoint from repo root:

```powershell
python benchmarks/run_benchmarks.py head-to-head --suite smoke --repeats 1
python benchmarks/run_benchmarks.py structured-showcase --suite smoke
```

## Pull Requests

- Keep changes scoped
- Include tests or explain why tests do not apply
- Update README or benchmark docs when user-facing workflow changes
- Do not commit generated `results/` artifacts

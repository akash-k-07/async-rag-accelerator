# Contributing to Spec-RAG

Thank you for your interest in contributing to Spec-RAG.

## How to contribute

1. Fork the repository.
2. Create a feature branch:
   ```bash
git checkout -b feature/my-change
```
3. Make your changes and run tests.
4. Submit a pull request.

## Development workflow

- Use `python -m venv venv` to create a virtual environment.
- Activate the environment:
  - macOS/Linux: `source venv/bin/activate`
  - Windows: `venv\Scripts\activate`
- Install dependencies with:
  ```bash
pip install -r requirements.txt
```

## Recommended checks

- Run Python syntax checks:
  ```bash
./venv/bin/python3 -m py_compile spec_rag/nodes/handlers.py spec_rag/core/cache.py run_benchmark.py
```
- Run benchmark smoke tests after installing dependencies.

## Report issues

If you find a bug or want to request a feature, please open an issue on GitHub.

## Code style

- Keep changes small and specific.
- Use `snake_case` for functions and variables.
- Document new behavior in `README.md` when it affects usage.

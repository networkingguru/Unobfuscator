# Contributing to Unobfuscator

Thank you for your interest in contributing. The project is currently in private
development and not yet accepting external pull requests, but this document
describes conventions for when it opens up.

---

## Filing Issues

- Search existing issues before opening a new one.
- For bugs, include: Python version, OS, the relevant log lines from
  `python unobfuscator.py log`, and steps to reproduce.
- For feature requests, describe the use case — what are you trying to do
  and why doesn't the current behaviour work for you?

---

## Pull Requests

1. Fork the repo and create a feature branch from `main`.
2. Keep changes focused — one logical change per PR.
3. Add or update tests in `tests/` for any changed behaviour.
4. Run the test suite before opening a PR:
   ```bash
   python -m pytest tests/
   ```
5. Commit messages should follow the format already used in this repo:
   `type: short description` (e.g. `fix: ...`, `feat: ...`, `docs: ...`).

---

## Understanding the Codebase

- [PIPELINE.md](PIPELINE.md) — Plain-English description of the six processing
  phases. Start here before touching `stages/`.
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — User-facing documentation.
- `core/db.py` — All database schema and query helpers. Add new queries here,
  not inline in stage files.
- `stages/` — One file per pipeline stage. Each stage only calls helpers from
  `core/`; no inline SQL.

---

## Running Tests

```bash
python -m pytest tests/          # full suite
python -m pytest tests/ -x -v   # stop on first failure, verbose
```

The test suite uses an in-memory SQLite database — no external services required.

# Contributing to PrismLang

Thank you for your interest in contributing. PrismLang is developed by
Insight IT Solutions LLC and welcomes community contributions.

## Development Setup

```bash
git clone https://github.com/insightits/prismlang
cd prismlang
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v --cov=prismlang --cov-report=term-missing
```

## Running Linter

```bash
ruff check prismlang/
ruff format prismlang/
```

## Running Type Checker

```bash
mypy prismlang/
```

## Pull Request Guidelines

1. **One concern per PR** — bug fix, feature, or docs. Not all three.
2. **Tests required** — new code needs tests. Coverage must not decrease.
3. **Docstrings required** — all public functions, classes, and methods.
4. **No breaking changes in patch releases** — deprecate first, remove later.
5. **Benchmark before/after for performance PRs** — run `python -m benchmarks.run_all`.

## Taxonomy Contributions

Domain taxonomy packs (healthcare, finance, legal, etc.) are especially welcome.
See `demo/taxonomy_config.py` for the format. Open a PR in `prismlang/taxonomies/`.

## Reporting Issues

Use [GitHub Issues](https://github.com/insightits/prismlang/issues).
For security vulnerabilities, email prismrag@insightits.com directly — do not open a public issue.

## License

By contributing, you agree that your contributions will be licensed under Apache 2.0.

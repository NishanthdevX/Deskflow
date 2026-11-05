# PyPI Publishing & Release Guide

Deskflow supports PyPI distribution through **Trusted Publishing** (OpenID Connect / OIDC), enabling direct distribution so that users can run `uvx deskflow` anywhere without prior manual installation.

---

## 1. Safety Gate: `PUBLISH_TO_PYPI`

Publishing to PyPI is protected by an explicit repository variable check:
- The workflow `.github/workflows/pypi.yml` only runs if `vars.PUBLISH_TO_PYPI == 'true'`.
- By default, this variable is not set or set to `false`, preventing unintended uploads.
- No PyPI API tokens or passwords are stored in repository secrets.

---

## 2. Setting Up Trusted Publishing

To configure PyPI Trusted Publishing for Deskflow:
1. Log in to [pypi.org](https://pypi.org/) and navigate to your account settings or project management.
2. Under **Publishing**, add a new **GitHub Publisher**:
   - Owner: `NishanthdevX`
   - Repository name: `Deskflow`
   - Workflow name: `pypi.yml`
   - Environment name: `pypi`
3. In GitHub repository settings (`Settings` -> `Secrets and variables` -> `Actions` -> `Variables`):
   - Add repository variable: `PUBLISH_TO_PYPI` = `true`.

---

## 3. Release Procedure

1. Verify local test suite, linting, and type checking:
   ```bash
   uv run pytest
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src
   ```
2. Tag the release commit following Semantic Versioning (`vMAJOR.MINOR.PATCH`):
   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
   ```
3. Create a GitHub Release referencing the tag.
4. When `PUBLISH_TO_PYPI` is set to `"true"`, GitHub Actions builds the wheel and sdist, verifies hashes, and uploads the distribution directly to PyPI using short-lived OIDC tokens.


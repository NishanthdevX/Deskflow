# Security policy

## Data handling

Deskflow operates entirely locally. It does not make any network connections at runtime.
All telemetry, model files, and prediction logs are stored under the user's own home
directory in platform-standard locations (via `platformdirs`). No data is transmitted to
any external service.

## Privilege model

Deskflow runs as the current user without elevation. A small number of metrics require
administrator or root access on specific platforms; those are opt-in, clearly documented,
and never triggered automatically.

## Reporting a vulnerability

If you find a security vulnerability, please do not open a public GitHub issue.

Report it by emailing the maintainer at the address listed in `pyproject.toml`, or by
opening a private GitHub Security Advisory on this repository.

Include:
- A description of the vulnerability
- Steps to reproduce it
- The version of deskflow affected
- Any suggested mitigations

You will receive an acknowledgement within 48 hours and a substantive response within 7 days.


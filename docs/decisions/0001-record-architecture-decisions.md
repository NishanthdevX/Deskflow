# ADR 0001: Record architecture decisions

Date: 2026-11-01
Status: Accepted

## Context

Deskflow is a college ML project that will evolve through nine phases. Design choices
accumulate and their rationale becomes unclear over time. A lightweight record helps
contributors understand why constraints exist before proposing changes.

## Decision

We adopt the Architecture Decision Record (ADR) format for any decision that:

- Adds or removes a dependency
- Restricts or expands an architectural boundary (e.g. which modules may import which)
- Chooses one technology over another with trade-offs
- Documents a deliberate limitation

ADRs are stored in `docs/decisions/` as `NNNN-short-title.md`. The format is:

- **Date** (YYYY-MM-DD)
- **Status** (Proposed / Accepted / Superseded by NNNN)
- **Context** -- the situation and constraints
- **Decision** -- what was decided
- **Consequences** -- trade-offs and follow-on implications

## Consequences

- Every non-trivial architectural choice has a written record.
- Adding a new runtime dependency requires an ADR justifying why the existing set
  cannot meet the need (per AGENTS.md section 3).
- The numbering is monotonically increasing; superseded ADRs are kept for history.


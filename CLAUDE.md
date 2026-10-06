# CLAUDE.md

Two linked pure-Python projects sharing an error-state EKF + quaternion core: **Project A** — 15-state ES-EKF on real PX4 drone logs; **Project B** — 3U CubeSat attitude determination and control (ADCS) simulation.

## Start here (every new chat)

1. Read `docs/IMPLEMENTATION_PLAN.md` — the **status board** at the top says which group is next and which main-session model it wants; each group has a task table with the subagent and model for every task.
2. Read `PROGRESS.md` — what each finished group delivered, measured numbers, open decisions.
3. Read `CONVENTIONS.md` — fixed conventions (scalar-first Hamilton quaternions, body-to-reference; right/local attitude error; SI units; frames). Never change it silently.

## Working rules

- Work **one group at a time**. At the end of a group: full `pytest` green, update `PROGRESS.md` and the plan's status board, make one git commit, then **stop and wait** for the user's next prompt. Never start the next group unprompted.
- Delegate per the group's task table to the subagents in `.claude/agents/`: `py-implementer`, `test-author`, `data-wrangler`, `figures-runner`, `doc-writer` (Sonnet) and `math-reviewer` (Opus, read-only, only at ★ checkpoints). Run independent tasks in parallel.
- Spec files in `docs/specs/` are the contract between implementer and test-author agents; write the spec before delegating a module.
- It is **attitude** determination and control — never "altitude" except for genuine height.
- Every reported number must come from a conservation law, an independent library oracle, the PX4 log's EKF2, or a cited source; assumptions go in `assumptions.md`.

## Commands (Windows)

```bash
.venv/Scripts/python -m pytest -q          # all tests (~130 s)
.venv/Scripts/python -m pip install -e .    # after adding packages/modules
```

Python 3.14 venv at `.venv/` (fallback `py -3.12` if a wheel is missing).

## Layout

`common/` (quaternion, frames, ekf — shared), `sensor_fusion/` (Project A), `adcs/` (Project B); tests in `<package>/tests/`; figures in `<package>/report/figures/`.

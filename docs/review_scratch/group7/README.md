# Group 7 ★ review scratch scripts (reference only)

Written by the math-reviewer during the Group 7 ★ MEKF review and copied here so the numbers quoted in `assumptions.md` B-20 ("Scope and optimism") and `PROGRESS.md` (Group 7 findings) can be traced. **Not reviewed, not tested, not part of the package.** Rewrite them as a proper report script (e.g. `adcs/report/estimation_robustness.py`) before using their numbers in the design document.

- `nees_mc.py` — builds `truth_pointing5.npz` (5-orbit nadir-pointing truth + reference vectors, written next to the script) and runs the 40-seed NEES ensemble (eclipse-exit NEES 2.80–3.19; 47 % of seeds reach a 100-sample moving-mean NEES ≥ 10).
- `opt.py` — unmodelled 300 nT magnetometer bias (with and without σ_m = 300 nT in the filter) and the crude albedo case (needs `truth_pointing5.npz`).
- `fov.py` — hard ±60° CSS field of view (needs `truth_pointing5.npz`).
- `white.py` — whiteness of the magnetometer innovations (lag-1 autocorrelation −0.001); run from this directory.

They contain an absolute `sys.path` entry for the repo on the original machine; run with `.venv/Scripts/python docs/review_scratch/group7/<script>.py` after `nees_mc.py`.

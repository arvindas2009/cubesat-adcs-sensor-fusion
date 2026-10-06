"""Diagnose the worst Monte Carlo runs of the design campaign (Group 8 failure-mode analysis).

Re-runs the ``n`` design-campaign runs with the largest steady-state nadir error
(``monte_carlo_design.json``) with the same inputs (``adcs.monte_carlo.draw_run``,
same seeds, so the runs are reproduced exactly) and records where and why the
maximum occurs: time, sunlit/eclipse, time since the last eclipse entry or exit,
knowledge error and control error (estimate vs target) at the maximum, the
filter's 3-sigma at that time, and the fraction of steady-state samples above
4 deg. Writes ``adcs/report/mc_worst_runs.json`` and
``adcs/report/figures/mc_worst_run.png`` (the worst run).

Run: ``python -m adcs.report.mc_worst_runs [--n 5] [--workers 5]``.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPORT = Path(__file__).resolve().parent
FIG = REPORT / "figures"


def _diagnose(index: int) -> dict:
    from adcs import monte_carlo as mc
    from adcs import simulate as S
    from adcs.config import DEFAULT
    from adcs.orbit import orbital_period

    P = float(orbital_period(DEFAULT.orbit.a))
    spec = mc.draw_run(index)
    sensors, mekf_cfg = mc.CAMPAIGNS["design"]
    inertia = np.diag(np.diag(DEFAULT.spacecraft.inertia) * np.asarray(spec.inertia_scale))
    res = S.run_closed_loop(list(spec.y0), 6.0 * P, seed=list(spec.sim_seed), sensors=sensors,
                            mekf_cfg=mekf_cfg, inertia_true=inertia)
    sm = S.summarize_closed_loop(res)
    sw = res.meta["switch_time"]
    win = (res.mode == S.MODE_POINTING) & (res.t >= sw + P)
    idx = np.nonzero(win)[0]
    k = int(idx[np.argmax(res.nadir_err[win])])
    lit = res.lit
    j = k
    while j > 0 and lit[j] == lit[k]:
        j -= 1
    e = np.rad2deg(res.nadir_err[win])
    ke = np.rad2deg(np.linalg.norm(res.know_err, axis=1))
    out = {
        "index": index,
        "rate_degps": float(np.rad2deg(spec.rate)),
        "inertia_scale": list(spec.inertia_scale),
        "nadir_max_deg": sm["nadir_max_deg"],
        "nadir_rms_deg": sm["nadir_rms_deg"],
        "t_max_s": float(res.t[k]),
        "t_max_orbits": float(res.t[k] / P),
        "sunlit_at_max": bool(lit[k]),
        "time_since_last_transition_s": float(res.t[k] - res.t[j]),
        "transition": "sunlit start (eclipse exit)" if lit[k] else "eclipse entry",
        "know_err_at_max_deg": float(ke[k]),
        "know_err_vec_at_max_deg": np.rad2deg(res.know_err[k]).tolist(),
        "ctrl_err_at_max_deg": float(np.rad2deg(res.est_nadir_err[k])),
        "att_3sigma_at_max_deg": float(3.0 * np.rad2deg(res.att_sigma[k])),
        "frac_steady_above_4deg": float(np.mean(e > 4.0)),
        "frac_steady_above_5deg": float(np.mean(e > 5.0)),
        "frac_steady_know_outside_3sigma": float(np.mean(ke[win] > 3.0 * np.rad2deg(res.att_sigma[win]))),
        "nees_mean": sm["nees_mean"],
        "mag_bias_true_nT": sm["mag_bias_true_nT"],
        "mag_bias_err_end_nT": sm["mag_bias_err_end_nT"],
    }
    trace = {"t": res.t[::10], "nadir": np.rad2deg(res.nadir_err[::10]), "know": ke[::10],
             "sig3": 3.0 * np.rad2deg(res.att_sigma[::10]), "lit": lit[::10], "switch": sw, "P": P}
    return {"summary": out, "trace": trace}


def _plot(trace: dict, summ: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = trace["P"]
    t = trace["t"] / P
    fig, ax = plt.subplots(figsize=(10, 4.5))
    lit = trace["lit"]
    edges = np.flatnonzero(np.diff(np.r_[0, (~lit).astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color="0.88", lw=0)
    ax.plot(t, trace["nadir"], color="tab:red", lw=0.8, label="true nadir pointing error")
    ax.plot(t, trace["know"], color="tab:blue", lw=0.8, label="knowledge error |dtheta|")
    ax.plot(t, trace["sig3"], color="tab:blue", lw=0.8, ls="--", label="filter 3 sigma (sqrt tr P_att)")
    ax.axhline(5.0, color="k", ls=":", lw=1, label="REQ-PNT 5 deg")
    ax.axvline((trace["switch"] + P) / P, color="0.4", ls="-.", lw=0.8, label="steady-state window start")
    ax.set_xlim(trace["switch"] / P, t[-1])
    ax.set_ylim(0, 6.5)
    ax.set_xlabel("time [orbits]")
    ax.set_ylabel("angle [deg]")
    ax.set_title(f"Worst design-campaign run {summ['index']} (grey = eclipse): max {summ['nadir_max_deg']:.2f} deg "
                 f"at {summ['t_max_orbits']:.2f} orbits")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args(argv)
    data = json.loads((REPORT / "monte_carlo_design.json").read_text())
    runs = sorted((r for r in data["runs"] if r.get("nadir_max_deg") is not None),
                  key=lambda r: r["nadir_max_deg"], reverse=True)[: a.n]
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(_diagnose, [r["index"] for r in runs]))
    for r, res in zip(runs, results):
        if abs(res["summary"]["nadir_max_deg"] - r["nadir_max_deg"]) > 1e-9:
            print(f"WARNING: run {r['index']} not reproduced ({res['summary']['nadir_max_deg']} vs {r['nadir_max_deg']})")
    summaries = [res["summary"] for res in results]
    out = {
        "source": "monte_carlo_design.json",
        "n": len(summaries),
        "runs": summaries,
        "n_max_in_eclipse": int(sum(not s["sunlit_at_max"] for s in summaries)),
    }
    (REPORT / "mc_worst_runs.json").write_text(json.dumps(out, indent=1))
    FIG.mkdir(exist_ok=True)
    _plot(results[0]["trace"], summaries[0], FIG / "mc_worst_run.png")
    for s in summaries:
        print(f"run {s['index']}: max {s['nadir_max_deg']:.2f} deg at {s['t_max_orbits']:.2f} orbits, "
              f"{'sunlit' if s['sunlit_at_max'] else 'eclipse'} ({s['time_since_last_transition_s']:.0f} s after "
              f"{s['transition']}), knowledge {s['know_err_at_max_deg']:.2f} deg vs 3 sigma "
              f"{s['att_3sigma_at_max_deg']:.2f}, control {s['ctrl_err_at_max_deg']:.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

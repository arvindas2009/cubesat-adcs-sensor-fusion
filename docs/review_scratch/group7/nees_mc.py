import os, sys, time
os.environ["OPENBLAS_NUM_THREADS"]="1"
sys.path.insert(0, r"D:/Projects/Coding/ubeSat ADCS + Sensor Fusion")
import numpy as np
from multiprocessing import Pool
OUT = os.path.join(os.path.dirname(__file__), "truth_pointing5.npz")

def make_truth():
    from adcs.config import DEFAULT
    from adcs.environment import FieldTable
    from adcs.orbit import Ephemeris, initial_state, orbital_period
    from adcs.simulate import lvlh_state, run
    P = float(orbital_period(DEFAULT.orbit.a))
    r0, v0 = initial_state(DEFAULT)
    res = run(lvlh_state(r0, v0, offset=np.deg2rad([2,-1,3])), 5*P, "pointing", log_every=1)
    t = res.t
    eph = Ephemeris(r0, v0, t[-1]+10); ft = FieldTable(eph, DEFAULT, t[-1]+10)
    np.savez(OUT, t=t, q=res.q, w=res.w, b=ft.b_eci(t), s=ft.sun_dir_eci(t), lit=ft.sunlit(t), P=P)

def work(seed):
    from adcs.sensors import measure_series
    from adcs.estimation import run_mekf, attitude_errors, nees, MekfConfig
    d = np.load(OUT)
    data = measure_series(d["t"], d["q"], d["b"], d["s"], d["lit"], seed=seed)
    res = run_mekf(data, d["b"], d["s"], MekfConfig())
    err = attitude_errors(d["q"], res.q)
    e = np.full(len(d["t"]), np.nan)
    k0 = res.k_init
    e[k0:] = nees(err[k0:], res.P_att[k0:])
    # bias NEES too
    be = data.gyro_bias - res.b
    return seed, e, err, res.P_att, be, res.std[:,3:]

if __name__ == "__main__":
    if not os.path.exists(OUT):
        t0=time.time(); make_truth(); print("truth", time.time()-t0)
    with Pool(20) as p:
        outs = p.map(work, range(40))
    np.savez(os.path.join(os.path.dirname(__file__), "mc40.npz"),
             nees=np.array([o[1] for o in outs]), err=np.array([o[2] for o in outs]).astype(np.float32),
             Patt=np.array([o[3] for o in outs]).astype(np.float64)[:, :, :, :], be=np.array([o[4] for o in outs]), sb=np.array([o[5] for o in outs]))
    print("done")

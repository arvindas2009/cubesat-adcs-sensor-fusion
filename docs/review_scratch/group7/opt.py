import os, sys
os.environ["OPENBLAS_NUM_THREADS"]="1"
sys.path.insert(0,r"D:/Projects/Coding/ubeSat ADCS + Sensor Fusion")
import numpy as np
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__))
def job(args):
    case, seed = args
    from common import quaternion as quat
    from adcs.sensors import measure_series, sun_vector_from_currents, SunSensorConfig, CSS_NORMALS
    from adcs.estimation import run_mekf, attitude_errors, nees, MekfConfig
    from adcs.config import DEFAULT
    from adcs.orbit import Ephemeris, initial_state
    d=np.load(os.path.join(HERE,"truth_pointing5.npz")); P=float(d["P"])
    n=int(3*P)+1
    t,q,b,s,lit=d["t"][:n],d["q"][:n],d["b"][:n],d["s"][:n],d["lit"][:n]
    data=measure_series(t,q,b,s,lit,seed=seed)
    cfg=MekfConfig()
    rng=np.random.default_rng(100+seed)
    if case.startswith("magbias"):
        bb=rng.normal(size=3); bb*=300e-9/np.linalg.norm(bb)
        data.mag=data.mag+bb
        if case=="magbias300_filter300": cfg=MekfConfig(sigma_mag=300e-9)
    if case=="albedo":
        r0,v0=initial_state(DEFAULT); eph=Ephemeris(r0,v0,t[-1]+10)
        r=eph.at(t)[0]; nad=-r/np.linalg.norm(r,axis=1,keepdims=True)
        nad_b=quat.rotate_inverse(q,nad)
        # crude: albedo 0.3 x view factor ~0.6 on Earth-facing diodes when sub-satellite point sunlit (r.s>0)
        a=0.18*(np.einsum('ij,ij->i',-nad,s)>0)
        cur=data.css_currents+a[:,None]*np.maximum(nad_b@CSS_NORMALS.T,0)
        data.css_currents=cur; data.sun,data.sun_valid=sun_vector_from_currents(cur)
    res=run_mekf(data,b,s,cfg)
    err=attitude_errors(q,res.q); en=np.degrees(np.linalg.norm(err,axis=1))
    k1=t>=P
    e=nees(err[300:],res.P_att[300:])
    nm=res.nis["mag"][1][res.nis["mag"][2]].mean(); ns=res.nis["sun"][1][res.nis["sun"][2]].mean()
    return case,seed,np.sqrt(np.mean(en[k1&lit]**2)),np.sqrt(np.mean(en[k1&~lit]**2)),e.mean(),nm,ns
if __name__=="__main__":
    cases=["base","magbias300","magbias300_filter300","albedo"]
    with Pool(16) as p: out=p.map(job,[(c,s) for c in cases for s in range(4)])
    for c in cases:
        a=np.array([o[2:] for o in out if o[0]==c])
        print("%-22s RMS sunlit %.2f eclipse %.2f deg | NEES %.1f | NIS mag %.1f sun %.2f"%((c,)+tuple(a.mean(0))))

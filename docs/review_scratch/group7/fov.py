import os, sys
os.environ["OPENBLAS_NUM_THREADS"]="1"
sys.path.insert(0,r"D:/Projects/Coding/ubeSat ADCS + Sensor Fusion")
import numpy as np
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__))
def job(seed):
    from common import quaternion as quat
    from adcs.sensors import measure_series, sun_vector_from_currents, CSS_NORMALS
    from adcs.estimation import run_mekf, attitude_errors, nees, MekfConfig
    d=np.load(os.path.join(HERE,"truth_pointing5.npz")); P=float(d["P"]); n=int(3*P)+1
    t,q,b,s,lit=d["t"][:n],d["q"][:n],d["b"][:n],d["s"][:n],d["lit"][:n]
    data=measure_series(t,q,b,s,lit,seed=seed)
    sb=quat.rotate_inverse(q,s); cosi=sb@CSS_NORMALS.T
    rng=np.random.default_rng(seed+7)
    clean=np.where(lit[:,None]&(cosi>=0.5),cosi,0.0)
    cur=clean+rng.normal(0,0.0436,clean.shape)
    data.css_currents=cur; data.sun,data.sun_valid=sun_vector_from_currents(cur)
    sm=data.sun/np.linalg.norm(data.sun,axis=1,keepdims=True)
    angerr=np.degrees(np.arccos(np.clip((sm*sb).sum(1),-1,1)))[data.sun_valid]
    res=run_mekf(data,b,s,MekfConfig())
    err=attitude_errors(q,res.q); en=np.degrees(np.linalg.norm(err,axis=1)); k1=t>=P
    k0=max(res.k_init,0)
    e=nees(err[k0+300:],res.P_att[k0+300:])
    ns=res.nis["sun"][1].mean()
    return np.mean(data.sun_valid[lit]), np.sqrt(np.mean(angerr**2)), np.sqrt(np.mean(en[k1&lit]**2)), np.sqrt(np.mean(en[k1&~lit]**2)), e.mean(), ns
if __name__=="__main__":
    with Pool(4) as p: out=np.array(p.map(job,range(4)))
    print("FOV60: frac lit valid %.3f | CSS-vector RMS angle err %.1f deg | MEKF RMS sunlit %.2f eclipse %.2f | NEES %.0f | NIS sun %.1f"%tuple(out.mean(0)))

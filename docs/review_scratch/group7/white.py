import sys; sys.path.insert(0,r"D:/Projects/Coding/ubeSat ADCS + Sensor Fusion")
import numpy as np
from adcs.sensors import measure_series
from adcs.estimation import MEKF, MekfConfig, triad_init
d=np.load("truth_pointing5.npz"); P=float(d["P"]); n=int(2*P)
t,q,b,s,lit=d["t"][:n],d["q"][:n],d["b"][:n],d["s"][:n],d["lit"][:n]
data=measure_series(t,q,b,s,lit,seed=3); cfg=MekfConfig()
q0,P0=triad_init(data.mag[0],data.sun[0],b[0],s[0],cfg); f=MEKF(q0,np.zeros(3),P0,cfg)
wm=[]
for k in range(1,n):
    f.propagate(data.gyro[k],1.0); r=f.update_mag(data.mag[k],b[k],t[k])
    L=np.linalg.cholesky(r.S); wm.append(np.linalg.solve(L,r.residual))
    if data.sun_valid[k]: f.update_sun(data.sun[k],s[k],t[k])
w=np.array(wm)[300:]
for L in (1,2,10,100):
    c=np.mean([np.corrcoef(w[:-L,i],w[L:,i])[0,1] for i in range(3)])
    print("lag %d whitened mag innovation autocorr %.4f (1/sqrt(N)=%.4f)"%(L,c,1/np.sqrt(len(w))))

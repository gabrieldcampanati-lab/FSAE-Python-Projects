import numpy as np
from scipy.spatial.transform import Rotation
import math

def plane_proj(v, n):
    dp = np.dot(v, n)
    norm = np.dot(n, n)
    proj_n = (dp/norm)*n
    proj = v - proj_n
    return proj

def change_system(v1):
    vf = np.array([0.0,0.0,0.0])
    vf[0] = v1[1]/1000
    vf[1] = v1[2]/1000
    vf[2] = -v1[0]/1000
    return vf

v3 = np.array([0,-519,123])
v7 = np.array([6.986, -504, 283])

vd = v7-v3
c = (203-v3[2])/vd[2]

vf = v3 + c*vd

f7 = np.array([6.985, -490.78, 283])
f8 = np.array([6.985, -298.0171, 109.3688])


r7 = np.array([1540.0, -490.7876, 283])
r8 = np.array([1540.0, -361.045, 129.2736])

r0 = np.array([1540.0,-600,203])

r4 = np.array([123.0, -269.8087, 254.6340])
r5 = np.array([1450.0, -269.8087, 254.6340])

r1 = np.array([1196.0, -300, 163.9306])
r2 = np.array([1460.0, -300, 163.9306])

r6 = r7
r3 = np.array([1540.0, -519.0, 123.0])

r9 = np.array([1600.0, -490.7876, 283.0])

f0 = np.array([0.0, -600.0, 203.0])

f4 = np.array([-115.0, -224.2976, 232.5024])
f5 = np.array([118.0, -224.2976, 232.5024])
f1 = np.array([-151.0, -250.0, 157.7263])
f2 = np.array([135.0, -250.0, 157.7263])
f6 = np.array([4.9850, -490.7876, 283.0])
f3 = np.array([-2.0, -519.0, 123.0])
f10 = np.array([-75.0, -228.7514, 195.8898])
f9 = np.array([-75.0, -515.0000, 202.8712])

of0 = np.array([0.0,-600.0,255.0])
of9 = np.array([87.0,-517.0,196.6536])
of10 = np.array([88.6461,-153.9980,197.6976])

p = change_system(f9-f0)
print("%.4f, %.4f, %.4f"%(p[0],p[1],p[2]))
#print(45/math.radians(200))
r = 12.891
R = 87
print(R/r)
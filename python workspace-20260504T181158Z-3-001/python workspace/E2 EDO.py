import numpy as np
import math
import cmath
import matplotlib.pyplot as plt

E0 = (2*math.pi)
u = 1
b0 = 0
a0 = 1
t0 = 0
t1 = 1
npoints = 10000
w=1*E0*u
Omega = E0*u

def adot(t, b):
    return (-1j)*(E0*cmath.exp(-2j*w*t) + E0.conjugate())*u*b

def bdot(t, a):
    return (-1j)*(E0 + E0.conjugate()*cmath.exp(2j*w*t))*a

def fdot(t, v):
    return (adot(t, v[1]), bdot(t, v[0]))

def RWA(t):
    a = cmath.cos(abs(Omega)*t)
    b = 1j*Omega*cmath.sin(abs(Omega)*t)/abs(Omega)
    return (a, b)

def quad_list(list):
    ret = []
    for e in list:
        ret.append(e.conjugate()*e)
    return ret

def euler_aprox(func, y0, t0, t1, numpoints):
    ret0 = [y0[0]]
    ret1 = [y0[1]]
    t=t0
    dt = (t1-t0)/numpoints
    for i in range(0,numpoints):
        df = func(t, [ret0[i], ret1[i]])
        ret0.append(ret0[i] + dt*df[0])
        ret1.append(ret1[i] + dt*df[1])
        t+=dt
    return [ret0, ret1]


def dif_list(l1, l2):
    s = 0
    for i in range(0, len(l1)):
        s += abs(l2[i]-l1[i])
    return s/npoints
        

ts = np.linspace(t0,t1,npoints+1)
val_rwa = [[],[]]
plt.figure()
plt.xlabel("w/Omega")
plt.ylabel("Difference Sum of Probability amplitude")
w_val = []
dif_a_val = []
dif_b_val = []
for e in ts:
    val_rwa[0].append(RWA(e)[0])
    val_rwa[1].append(RWA(e)[1])
for i in range(0, 10):
    val = euler_aprox(fdot, [a0, b0], t0, t1, npoints)
    dif_a = dif_list(quad_list(val_rwa[0]), quad_list(val[0]))
    dif_b = dif_list(quad_list(val_rwa[1]), quad_list(val[1]))
    w_val.append(w/Omega)
    dif_a_val.append(dif_a)
    dif_b_val.append(dif_b)
    w*=2

plt.plot(w_val, dif_a_val, label = "a")
plt.plot(w_val, dif_b_val, label = "b")
plt.title("Error of RWA (a, b)")
plt.legend()
plt.show()

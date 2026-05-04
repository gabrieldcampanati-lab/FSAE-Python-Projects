import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import channel_module as cm

def load(path):
    data = pd.read_csv(path, header = 0, sep=",")
    return data

def integral(x, y, limits):
    s = 0
    for i in range(limits[0], limits[1]):
        dx = x[i+1]-x[i]
        f = (y[i+1]+y[i])/2
        s += f*dx
    return s

def derivative(dx, dy):
    return dx/dy

def integralArray(x, y, min):
    ret = []
    for i in range(min,len(y)-1):
        ret.append(integral(x, y, [min,i]))
    return ret

def getV(t, ax, theta, interval):
    vxs = 0
    rxs = 0
    
def getVArray(t, ax, theta):
    ret = []
    for i in range(1, len(ax)-2):
        ret.append(getV(t, ax, theta, [0, i]))
    return ret

def toCoord(x1, y1, theta):
    x2 = x1*np.cos(np.deg2rad(theta))
    y2 = y1*np.sin(np.deg2rad(theta))
    return x2, y2
    

def getR(t, ax, ay, az, theta):
    ax, ay = toCoord(ax, ay, theta)
    az = az - 10

    plt.figure()
    plt.plot(t, theta)
    plt.show()
    plt.figure()
    plt.plot(t, ax)
    plt.plot(t, ay)
    plt.plot(t, az)
    plt.show()

    plt.figure()
    plt.plot(ax, ay)
    plt.show()

    vx = integralArray(t, ax, 0)
    vy = integralArray(t, ay, 0)
    vz = integralArray(t, az, 0)

    t.pop(0)
    print(t)

    plt.figure()
    plt.plot(t, vx)
    plt.plot(t, vy)
    plt.plot(t, vz)
    plt.show()

    plt.figure()
    plt.plot(vx, vy)
    plt.show()

    rx = integralArray(t, vx, 1)
    ry = integralArray(t, vy, 1)
    rz = integralArray(t, vz, 1)
    t.pop(1)
    print(t)
    plt.figure()
    plt.plot(t[1:], rx)
    plt.plot(t[1:], ry)
    plt.plot(t[1:], rz)
    plt.show()

    plt.figure()
    plt.plot(rx, ry)
    plt.show()

    return (rx, ry, rz)

def calibrate(ax, ay, az):
    n = [np.mean(ax), np.mean(ay), np.mean(az)]
    n = n/np.linalg.norm(n)
    x1 = [1,0,0]-np.dot(n,[1,0,0])*n
    x1 = x1/np.linalg.norm(x1)
    y1 = np.cross(n, x1)
    return x1, y1, n

def translate(ax, ay, az, x1, y1, z1):
    ax2, ay2, az2 = [], [], []
    for i in range(0, len(ax)):
        v = [ax[i], ay[i], az[i]]
        x = np.dot(v,x1)
        y = np.dot(v,y1)
        z = np.dot(v,z1)
        ax2.append(x)
        ay2.append(y)
        az2.append(z)
    return ax2, ay2, az2



filter = cm.LowPassFilter(60, 5, 12)

data = load("cal.csv")
t = data["t"]/1000
ax = data["ax"]
ax = filter.filter(ax)
ay = data["ay"]
ay = filter.filter(ay)
az = data["az"]
az = filter.filter(az)
print(t[1]-t[0])

plt.figure()
plt.plot(t, ax)
plt.plot(t, ay)
plt.plot(t, az)
plt.show()

x1, y1, z1 = calibrate(ax, ay, az)

data2 = load("log.csv")
t = data2["t"]/1000
ax = data2["ax"]
ax = filter.filter(ax)
ay = data2["ay"]
ay = filter.filter(ay)
az = data2["az"]
az = filter.filter(az)

ax, ay, az = translate(ax, ay, az, x1, y1, z1)

az = az - np.mean(az)

plt.figure()
plt.plot(t, ax)
plt.plot(t, ay)
plt.plot(t, az)
plt.show()

vx = integralArray(t, ax, 0)
vy = integralArray(t, ay, 0)
vz = integralArray(t, az, 0)

plt.figure()
plt.plot(t[1:], vx)
plt.plot(t[1:], vy)
plt.plot(t[1:], vz)
plt.show()
import math
import numpy as np

def getKFromFrequency(f, m):
    a = math.pi*2*f
    return m*(a**2)

def getWheelRate(angle, leverage, k):
    ang = np.deg2rad(angle)
    flr = 1/math.cos(ang[2])
    fm = flr*math.sin(ang[0])*leverage[0]/(math.sin(ang[1])*leverage[1])
    print("fm")
    print(fm)
    return k/(fm*fm)

"""
anglef = [121.61, 93.1, 38.34]
leveragef = [115.3, 87.94]
kf = 132.32

angler = [124.85, 81.64, 51.61]
leverager = [107.51, 110.48]
kr = 78.015
print("front")
print(getWheelRate(anglef, leveragef, kf))
print("rear")
print(getWheelRate(angler, leverager, kr))
"""

print(getKFromFrequency(3.0, 240*0.49))
print(getKFromFrequency(3.2, 240*0.51))
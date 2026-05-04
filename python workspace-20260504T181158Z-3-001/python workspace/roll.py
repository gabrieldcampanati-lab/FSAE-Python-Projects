import math
import numpy as np
import matplotlib.pyplot as plt

GRAVITY_ACCL = 9.81

class Axle:
    def __init__(self, roll_center_height = 0.3, wheel_rate = 60e3, axle_suspended_mass = 150, half_track = 0.6) -> None:
        self.roll_center_height = roll_center_height
        self.wheel_rate = wheel_rate
        self.axle_suspended_mass = axle_suspended_mass
        self.half_track = half_track
    
    def getSpringsRollStiffness(self):
        k = (self.half_track**2)*2*self.wheel_rate*math.pi/180
        return k

def getRollStiffnes(roll_gradient, mass, roll_center_height):
    f = mass*GRAVITY_ACCL
    return f*roll_center_height/roll_gradient

def getRollGradient(k, mass, roll_center_height):
    f = mass*GRAVITY_ACCL
    return (f*roll_center_height)/k

def getRoll(g, roll_gradient):
    return g*roll_gradient

fAxle = Axle(wheel_rate = 50e3, roll_center_height=0.38)
rAxle = Axle(wheel_rate=40e3, roll_center_height=0.38)
rs = rAxle.getSpringsRollStiffness()
fs = fAxle.getSpringsRollStiffness()
print("Springs roll stiffness")
print(rs+fs)
print("Desired Roll Stiffness")
print(getRollStiffnes(0.3, 260, 0.20))
print("Roll Stiffness Distribution")
print("%.2f"%(fs/(rs+fs)))
print("Roll Gradient from Springs")
grad = getRollGradient(rs+fs, 260,0.175)
print(grad)
print("Roll for a 2g curve")
print(getRoll(2,grad))
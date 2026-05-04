import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

def plane_proj(v, n):
    dp = np.dot(v, n)
    norm = np.dot(n, n)
    proj_n = (dp/norm)*n
    proj = v - proj_n
    return proj

class KingPin:
    def __init__(self, kpi = 6, kp_offset = 51.75e-3, caster = 6, caster_offset = -0.4e-3, camber = 0, height = 0.255, toe = 0):
        self.kpi = kpi
        self.kp_offset = kp_offset
        self.caster_offset = caster_offset
        self.caster = caster
        self.camber = camber
        self.height = height
        self.static_wheel_dir = np.array([-math.sin(math.radians(toe))*math.cos(math.radians(camber)), math.cos(math.radians(toe))*math.cos(math.radians(camber)), -math.sin(math.radians(camber))])
        sz = 1/math.sqrt(math.tan(math.radians(caster))**2 + math.tan(math.radians(kpi))**2 + 1)
        sx = -sz*math.tan(math.radians(caster))
        sy = -sz*math.tan(math.radians(kpi))
        self.kp_axis = np.array([sx,sy,sz])
        self.kp_axis /= np.linalg.norm(self.kp_axis)
        self.mecanical_trail = caster_offset + height*math.tan(math.radians(caster))
        self.scrub_radius = kp_offset + height*math.tan(math.radians(kpi))



        return
    
    def flip(self):
        self.kpi*=-1
        self.kp_offset*=-1
        self.kp_axis[1]*=-1
        self.static_wheel_dir[1]*=-1
        return

    def getDynamicCamber(self, steer):
        kpi = math.radians(self.kpi)
        caster = math.radians(self.caster)
        gammak = kpi + np.acos(math.sin(kpi)*np.cos(steer)) - math.pi/2
        gammac = np.acos(math.sin(caster)*np.sin(steer)) - math.pi/2
        gamma = gammak + gammac

        return np.rad2deg(gamma) + self.camber
    
    def getDynamicCamber2(self, steer):
        n = self.getWheelDir(steer)
        camber = math.asin(-n[2])
        return math.degrees(camber)
    
    def getDisplacement(self, steer):
        rot_vec = steer*self.kp_axis
        rot = Rotation.from_rotvec(rot_vec)

        wheel_center = np.array([-self.caster_offset, self.kpi_offset, 0])
        wheel_center = rot.apply(wheel_center)

        camber_dh = self.height*(math.cos(math.radians(self.camber))-math.cos(math.radians(self.getDynamicCamber(math.radians(steer)))))

        return -(wheel_center[2]+camber_dh)

    def getWheelDir(self, steer):
        rot_vec = steer*self.kp_axis
        rot = Rotation.from_rotvec(rot_vec)
        n = rot.apply(self.static_wheel_dir)
        return n
    
    def getTrail(self, steer):
        
        r = [self.mecanical_trail,self.scrub_radius, 0]
        rot_vec = steer*np.array([0,0,1])
        rot = Rotation.from_rotvec(rot_vec)
        ret = rot.apply(r)
        return ret

    def getSteering(self, toe_steer):
        n = self.kp_axis
        ref = self.static_wheel_dir.copy()
        ref_dp = np.dot(ref, n)
        v = np.array([-math.sin(toe_steer), math.cos(toe_steer), 0])
        l = np.dot(v, n)
        c = (l*ref_dp + np.sign(self.static_wheel_dir[1])*n[2]*math.sqrt(l*l - ref_dp*ref_dp + n[2]*n[2]))/(l*l + n[2]*n[2])
        v *= c
        v[2] = (ref_dp - c*l)/n[2]
        proj_v = plane_proj(v, n)
        proj_v/=np.linalg.norm(proj_v)
        proj_ref = plane_proj(ref, n)
        proj_ref/=np.linalg.norm(proj_ref)

        return math.asin(np.dot(np.cross(proj_ref, proj_v),n))

    def getToe(self, steer):
        n = self.getWheelDir(steer)
        ref = self.static_wheel_dir.copy()
        ref[2] = 0
        ref /=np.linalg.norm(ref)
        m = np.array([n[0], n[1], 0])
        m /= np.linalg.norm(m)
        toe = math.asin(np.dot(np.cross(ref, m), [0,0,1]))
        return toe
        
    def getTorque(self, toe, f):

        rot_vec = toe*np.array([0,0,1])
        rot = Rotation.from_rotvec(rot_vec)
        f = rot.apply(f)
        steer = self.getSteering(toe)
        caster = math.radians(self.caster)
        kpi = math.radians(self.kpi)
        wheel_center = np.array([-self.caster_offset, self.kp_offset, 0])
        wheel_dir = self.getWheelDir(steer)

        rot_vec = steer*self.kp_axis
        rot = Rotation.from_rotvec(rot_vec)

        wheel_center = rot.apply(wheel_center)
        contact_point = wheel_center -self.height*math.sqrt(1-wheel_dir[2]**2)*np.array([0,0,1])
        torque_vec = np.cross(contact_point, f)
        return np.dot(torque_vec, self.kp_axis)

    def setOffsetFromLowerTab(self, pos):
        pos = np.array(pos)
        pos += (self.height-pos[2]/self.kp_axis[2])*self.kp_axis
        self.kp_offset = pos[1]
        self.caster = pos[0]
        return

def plotCamberVar(kp_list, range, numpoints):
    st = np.linspace(range[0], range[1], numpoints)
    plt.figure()
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    for kp in kp_list:
        x1 = []
        x = []
        y1 = []
        y2 = []
        for e in st:
            y1.append(kp.getDynamicCamber(np.deg2rad(e)))
            y2.append(kp.getDynamicCamber2(np.deg2rad(-e)))
            x.append(kp.getToe(np.deg2rad(e)))
        #plt.plot(st, y1)
        plt.plot(np.rad2deg(x), y2)
    plt.show()
    return

def getJoeM(kp, toe, f):
    steer = math.radians(toe)
    kpi = math.radians(kp.kpi)
    caster = math.radians(kp.caster)
    trail = kp.getTrail(steer)
    sr = trail[1]
    mt = trail[0]
    mfz = f[2]*sr*math.sin(kpi)*math.sin(steer) - f[2]*sr*math.sin(caster)*math.cos(steer)
    mfy = -f[1]*mt*math.cos(caster)*math.cos(steer)
    mfx = -f[0]*sr*math.cos(kpi)*math.cos(steer)
    return [mfx, mfy, mfz]

def getMFz(kpi, sr0, lh, steer, fz):
    sr = sr0-math.tan(kpi)*lh
    print("sr = %.1f"%(1000*sr))
    return sr*math.sin(kpi)*np.sin(steer)*fz

def printMFz(kpi, sr0, lh, steer, fz, numpoints):
    st = np.linspace(steer[0], steer[1], numpoints)
    y = getMFz(kpi, sr0, lh, st, fz)
    st = np.rad2deg(st)
    plt.plot(st, y)
    print(max(y))
    return

def getBumpForce(bump, fz, height):
    hc = (height-bump)/height
    s =math.sqrt(1 - hc**2)
    fx = fz*s
    return fx

k1 = KingPin(kpi = 6, caster = 7, camber=0.0, height=0.200)
k1.setOffsetFromLowerTab([0,0.081,0.123])
k2 = KingPin(kpi = 6, caster = 7, camber=0.0, height=0.200)
k2.setOffsetFromLowerTab([0,0.081,0.123])
k2.flip()
steer = math.radians(20)
f1 = [-10,750,1000]
f2 = [-10,450,600]
t1 = k1.getTorque(steer, f1)
t2 = k2.getTorque(steer, f2)

"""
"""
fl = [-10, 764, 459]
fr = [-10, 1557, 1040]
x = np.linspace(0,40,1000)
y1 = []
y2 = []
y3 = []
for e in x:
    steer = math.radians(e)
    k1 = KingPin(kpi = 10, caster = 3, camber=0.0, height=0.200)
    k1.setOffsetFromLowerTab([0,0.081,0.123])
    k2 = KingPin(kpi = 10, caster = 3, camber=0.0, height=0.200)
    k2.setOffsetFromLowerTab([0,0.081,0.123])
    k2.flip()
    t1 = k1.getTorque(steer, fl)
    t2 = k2.getTorque(steer, fr)
    y1.append(t1)
    y2.append(t2)
    y3.append(t1+t2)


plt.figure()
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.show()
fl = [-10,0,850]
fr = [-getBumpForce(0.02,850,0.2)-10,0,850]
k1 = KingPin(kpi = 10, caster = 2.5, camber=-0.2, height=0.200)
k1.setOffsetFromLowerTab([0,0.081,0.123])
k2 = KingPin(kpi = 10, caster = 2.5, camber=-0.2, height=0.200)
k2.setOffsetFromLowerTab([0,0.081,0.123])
k2.flip()

#plotCamberVar([k1,k2], [-35, 35], 1000)
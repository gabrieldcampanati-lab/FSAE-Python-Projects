import math
import numpy as np
import matplotlib.pyplot as plt

class LinearSuspension:
    def __init__(self,
                frontWheelRate = 50e3,
                rearWheelRate= 40e3,
                frontRollCenterHeight = 0.1,
                rearRollCenterHeight = 0.125,
                CGHeight = 0.38,
                frontHalfTrack = 0.6,
                rearHalfTrack = 0.6,
                wheelBase = 1.540,
                CGPosition = 0.5,
                frontARB = 0, 
                rearARB = 0,
                frontSwayArmLength = 0.6,
                rearSwayArmLength = 0.6
                ):
        
        self.frontWheelRate = frontWheelRate
        self.rearWheelRate = rearWheelRate
        self.frontRollCenterHeight = frontRollCenterHeight
        self.rearRollCenterHeight = rearRollCenterHeight
        self.CGHeight = CGHeight
        self.frontTrack = 2*frontHalfTrack
        self.rearTrack = 2*rearHalfTrack
        self.frontHalfTrack = frontHalfTrack
        self.rearHalfTrack = rearHalfTrack
        self.wheelBase = wheelBase
        self.CGPosition = CGPosition
        self.frontVerticalStiffness = 2*frontWheelRate
        self.rearVerticalStiffness = 2*rearWheelRate
        self.frontRollStiffness = ((frontHalfTrack)**2)*(frontARB+2*frontWheelRate)*math.pi/180
        self.rearRollStiffness = ((rearHalfTrack)**2)*(rearARB+2*rearWheelRate)*math.pi/180
        self.totalRollStiffness = self.frontRollStiffness+self.rearRollStiffness
        self.frontSwayArmLength = frontSwayArmLength
        self.rearSwayArmLength = rearSwayArmLength

    def getCamberVariation(self, fy):
        phi = self.getRoll(fy)
        z = self.getLateralJacking(fy)
        d1l = -math.radians(phi*(self.frontHalfTrack-self.frontSwayArmLength)/self.frontSwayArmLength) - z["F"]/self.frontSwayArmLength
        d1r = -math.radians(phi*(self.frontHalfTrack-self.frontSwayArmLength)/self.frontSwayArmLength) + z["F"]/self.frontSwayArmLength
        d2l = -math.radians(phi*(self.rearHalfTrack-self.rearSwayArmLength)/self.rearSwayArmLength) - z["R"]/self.rearSwayArmLength
        d2r = -math.radians(phi*(self.rearHalfTrack-self.rearSwayArmLength)/self.rearSwayArmLength) + z["R"]/self.rearSwayArmLength
        return {"FL": d1l, "FR": d1r, "RL": d2l, "RR": d2r}

    def getLateralJacking(self, Fy):
        dff = abs(Fy["FL"] - Fy["FR"])/2
        dfr = abs(Fy["RL"] - Fy["RR"])/2

        return {"F":(4*dff*self.frontRollCenterHeight)/(self.frontVerticalStiffness*self.frontTrack),
                "R":(4*dfr*self.rearRollCenterHeight)/(self.rearVerticalStiffness*self.rearTrack)}

    def getRollCenterHeight(self, fy,  dz = None):
        y1 = fy["FR"]+fy["FL"]
        y2 = fy["RR"]+fy["RL"]
        y = y1+y2
        a1b = (y2/y)*self.wheelBase
        a2b = (y1/y)*self.wheelBase
        if dz is None:
            dz = self.getLateralJacking(fy)

        return (a2b*(self.frontRollCenterHeight+dz["F"]) + a1b*(self.rearRollCenterHeight+dz["R"]))/(a1b+a2b)
    
    def getLateralTrackVariation(self, fy):
        z = self.getLateralJacking(fy)
        z1 = z["F"]
        z2 = z["R"]
        dt1 = -(4*self.frontRollCenterHeight/self.frontTrack)*z1
        dt2 = -(4*self.rearRollCenterHeight/self.rearTrack)*z2
        return {"F": dt1, "R": dt2}
    
    def getRoll(self, fy):
        y1 = fy["FR"]+fy["FL"]
        y2 = fy["RR"]+fy["RL"]
        y = y1+y2
        moment = y*(self.CGHeight - self.getRollCenterHeight(fy))
        return moment/self.totalRollStiffness
    
    def getAxleRoll(self, fy):
        y1 = fy["FR"]+fy["FL"]
        y2 = fy["RR"]+fy["RL"]
        y = y1+y2
        qb = self.getRollCenterHeight(fy)
        frontRoll = (self.rearRollStiffness/self.totalRollStiffness)*(y*(self.CGHeight-qb)/self.rearRollStiffness)
        rearRoll = (self.frontRollStiffness/self.totalRollStiffness)*(y*(self.CGHeight-qb)/self.frontRollStiffness)
        return {"F": frontRoll, "R": rearRoll}
    
    def getAxleLoadTransfer(self, fy):
        y1 = fy["FR"]+fy["FL"]
        y2 = fy["RR"]+fy["RL"]
        y = y1+y2
        dz = self.getLateralJacking(fy)
        qb = self.getRollCenterHeight(fy, dz=dz)
        cg = self.CGHeight + ((1-self.CGPosition)*dz["R"] + self.CGPosition*dz["F"])
        a1 = y*(cg-qb)
        dz1 = (self.frontRollStiffness/self.totalRollStiffness)*a1 + y1*self.frontRollCenterHeight
        dz2 = (self.rearRollStiffness/self.totalRollStiffness)*a1 + y2*self.rearRollCenterHeight
        dz1 /= self.frontTrack
        dz2 /= self.rearTrack
        return {"F": dz1, "R": dz2}
    
    def getRollBalance(self, fy):
        dz = self.getAxleLoadTransfer(fy)
        dz1 = dz["F"]
        dz2 = dz["R"]
        return dz1/(dz1+dz2)
    
    def getCriticalRelaxationTime(self, CGinertia, suspendedMass):
        cg = self.CGHeight
        m1 = CGinertia+suspendedMass*((cg-self.frontRollCenterHeight)**2)
        m2 = CGinertia+suspendedMass*((cg-self.rearRollCenterHeight)**2)
        w1 = math.sqrt(self.frontRollStiffness/m1)
        w2 = math.sqrt(self.rearRollStiffness/m2)
        w = min(w1,w2)
        return 1/w

def test():
    car = LinearSuspension(CGPosition=0.49,
    frontRollCenterHeight=(0.380-0.190),
    frontWheelRate=41e3,
    rearRollCenterHeight=(0.380-0.160),
    rearWheelRate=49e3,
    frontSwayArmLength=0.6,
    rearSwayArmLength=0.6,
    frontARB=0000,
    rearARB=0000)
    print(car.getCriticalRelaxationTime(36, 260))
    fy = {}
    weight = 3000
    u = 2
    x = np.linspace(0,1, 1000)
    rbl1 = []
    rbl2 = []
    for e in x:
        f1 = e*weight
        f2 = (1-e)*weight
        fy["FR"] = f1*car.CGPosition*u
        fy["FL"] = f2*car.CGPosition*u
        fy["RR"] = f1*(1-car.CGPosition)*u
        fy["RL"] = f2*(1-car.CGPosition)*u
        rb = car.getCamberVariation(fy)
        rbl1.append(rb["FL"])
        rbl2.append(rb["FR"])

    x -= 0.5
    rbl1= np.array(rbl1)
    rbl2= np.array(rbl2)
    plt.figure()
    plt.plot(x, np.degrees(rbl1))
    plt.plot(x, np.degrees(rbl2))
    plt.show()


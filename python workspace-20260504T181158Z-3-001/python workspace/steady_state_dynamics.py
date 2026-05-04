import magic_formula as mf
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import font
import raw_data_manager as rdm
import pandas as pd
import channel_module as cm
from scipy.optimize import fsolve
import linear_suspension_model as lsm

GRAVITY_ACCL = 10

WIDTH, HEIGHT = 1280, 720
PADX, PADY = 2, 2
FRAME_BACKGROUND_COLOR = "#D9D9D9"
FRAME_TITLE_FONT_COLOR = "#081E26"
FRAME_FONT_COLOR = "#737373"
WINDOW_BACKGROUND_COLOR = "#081E26"
CANVAS_BACKGROUND_COLOR = "#AAAAAA"

ARROW_THICKNESS = 2
ARROW_LENGTH = 50

WHEEL_WIDTH = 30
POINTING_COLOR = "#00F000"
SLIP_COLOR = "#F00000"
FORCE_COLOR = "#0000F0"

CAR_SCALE = 200
CG_RADIUS = 20
CG_COLOR = "#000000"

FL_COLOR = "#FF0A0A"
FR_COLOR = "#0AFF0A"
RL_COLOR = "#0A0AFF"
RR_COLOR = "#F0F00A"

FORCE_SCALE = 0.1

#---------------------Gemini------------------#
def draw_rotated_rectangle(canvas, center_x, center_y, _width, height, angle_radians, **kwargs):
    """
    Draws a rotated rectangle on a Tkinter canvas.

    :param canvas: The Tkinter Canvas widget.
    :param center_x: X coordinate of the rectangle's center.
    :param center_y: Y coordinate of the rectangle's center.
    :param width: The width of the rectangle.
    :param height: The height of the rectangle.
    :param angle_degrees: The rotation angle in degrees (counter-clockwise).
    :param kwargs: Additional options for create_polygon (e.g., fill, outline).
    """
    # Convert angle from degrees to radians
    cos_angle = math.cos(-angle_radians)
    sin_angle = math.sin(-angle_radians)

    # Calculate half width and half height
    hw, hh = _width / 2, height / 2

    # Define vertices relative to the center before rotation
    # Vertices are typically: top-left, top-right, bottom-right, bottom-left
    points_unrotated = [
        (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
    ]

    # Rotate and translate each vertex
    points_rotated = []
    for x, y in points_unrotated:
        # Apply rotation matrix: x' = x*cos(angle) - y*sin(angle), y' = x*sin(angle) + y*cos(angle)
        x_rot = x * cos_angle - y * sin_angle
        y_rot = x * sin_angle + y * cos_angle
        # Translate to the specified center
        x_translated = center_x + x_rot
        y_translated = center_y + y_rot
        points_rotated.extend([x_translated, y_translated])

    # Draw the rectangle as a polygon
    return canvas.create_polygon(points_rotated, **kwargs)

def draw_circle(canvas, x, y, radius, **kwargs):
    """Helper function to draw a circle by center coordinates and radius."""
    x0 = x - radius
    y0 = y - radius
    x1 = x + radius
    y1 = y + radius
    return canvas.create_oval(x0, y0, x1, y1, **kwargs)
#---------------------Gemini------------------#

class Wheel:
    def __init__(self, tire: mf.MFTire, load = 0, camber = 0, toe = 0, slipAngle = 0, steeringAngle = 0, parity = 1, pos = [0,0]):
        self.pos = pos
        self.tire = tire
        self.load = load
        self.camber = camber
        self.toe = toe
        self.slipAngle = slipAngle
        self.parity = parity
        self.steeringAngle = steeringAngle
        self.pressure = self.tire.p0
        return
    
    def getFy(self):
        fy = self.parity*self.tire.getFy3(self.parity*self.slipAngle, -self.load, self.camber, self.pressure)
        return fy
    
    def drawWheel(self, canvas: tk.Canvas, outline):
        pointingAngle = self.steeringAngle
        force = self.getFy()
        forceVector = {"x": -math.sin(pointingAngle)*force*FORCE_SCALE, "y": math.cos(pointingAngle)*force*FORCE_SCALE}
        draw_rotated_rectangle(canvas, self.pos[0], self.pos[1], WHEEL_WIDTH, WHEEL_WIDTH*2, pointingAngle, fill="black", outline  = outline, width=4)
        canvas.create_line(self.pos[0], self.pos[1], self.pos[0], self.pos[1]-ARROW_LENGTH, arrow=tk.LAST, width = ARROW_THICKNESS/2)
        canvas.create_line(self.pos[0], self.pos[1], self.pos[0]+ARROW_LENGTH*math.sin(-pointingAngle), self.pos[1]-ARROW_LENGTH*math.cos(pointingAngle), arrow=tk.LAST, width = ARROW_THICKNESS, fill = POINTING_COLOR)
        canvas.create_line(self.pos[0], self.pos[1], self.pos[0]+ARROW_LENGTH*math.sin(-(pointingAngle+self.slipAngle)), self.pos[1]-ARROW_LENGTH*math.cos(pointingAngle+self.slipAngle), arrow=tk.LAST, width = ARROW_THICKNESS, fill = SLIP_COLOR)
        canvas.create_line(self.pos[0], self.pos[1], self.pos[0]-forceVector["y"], self.pos[1]-forceVector["x"], arrow=tk.LAST, width = ARROW_THICKNESS, fill = FORCE_COLOR)
        return

class Car:
    def __init__(self, tire: mf.MFTire = None, cgHeight = 0.3, forwardCG = 0.5, wheelBase = 1.530, halfTrack = 0.6, mass = 300, Iz = 180, totalLoadTransferDistribution = 0.5, pos = [0,0], ackermann = 1):
        self.pos = pos
        self.tire = tire
        self.mass = mass
        self.weight = mass*GRAVITY_ACCL
        self.yawMomentOfInertia = Iz
        self.CGHeight = cgHeight
        self.forwardCG = forwardCG
        self.wheelBase = wheelBase
        self.halfTrack = halfTrack
        self.tireStaticCambers = {"FR":-0, "FL": 0, "RR": -0.2, "RL": -0.2}
        self.toes = {"FR":-0.0, "FL": 0.0, "RR": 0.0, "RL": -0.0}
        self.totalLoadTransferDistribution = totalLoadTransferDistribution
        self.updateLoads(0)
        self.lateralG = 0
        self.tangentVelocity = 0
        self.angularVelocity = 0
        self.steeringWheelAngle = 0
        self.wheels = {
            "FR": Wheel(tire, parity = 1, pos = [self.pos[0]+CAR_SCALE*self.halfTrack, self.pos[1]-CAR_SCALE*(1-forwardCG)*wheelBase]),
            "FL": Wheel(tire, parity = -1, pos = [self.pos[0]-CAR_SCALE*self.halfTrack, self.pos[1]-CAR_SCALE*(1-forwardCG)*wheelBase]),
            "RR": Wheel(tire, parity = 1, pos = [self.pos[0]+CAR_SCALE*self.halfTrack, self.pos[1]+CAR_SCALE*forwardCG*wheelBase]),
            "RL": Wheel(tire, parity = -1, pos = [self.pos[0]-CAR_SCALE*self.halfTrack, self.pos[1]+CAR_SCALE*forwardCG*wheelBase])}
        self.cgSlipAngle = 0
        self.steeringCoef = [ackermann, 1]
        self.ackermann = ackermann
        self.suspensionModel = lsm.LinearSuspension()
        self.staticLoads = self.getStaticLoads()
        self.loadTransfer = {"ZF": 0, "ZR": 0, "FL": 0, "FR": 0, "RL": 0, "RR": 0}
        self.loadTransferSS = {"ZF": 0, "ZR": 0, "FL": 0, "FR": 0, "RL": 0, "RR": 0, "w":0}
        return
    
    def updateWheels(self, load, slip, steer):
        for wheel in self.wheels:
            self.wheels[wheel].slipAngle = slip[wheel]
            self.wheels[wheel].load = load[wheel]
            self.wheels[wheel].toe = self.toes[wheel]
            self.wheels[wheel].steeringAngle = steer[wheel]



        return
    
    def draw(self, canvas):
        draw_circle(canvas, self.pos[0], self.pos[1], CG_RADIUS, fill = CG_COLOR)
        canvas.create_line(self.pos[0], self.pos[1], self.pos[0], self.pos[1]-ARROW_LENGTH, arrow=tk.LAST, width = ARROW_THICKNESS/2)
        canvas.create_line(self.pos[0], self.pos[1], self.pos[0]+ARROW_LENGTH*math.sin(-(self.cgSlipAngle)), self.pos[1]-ARROW_LENGTH*math.cos(self.cgSlipAngle), arrow=tk.LAST, width = ARROW_THICKNESS, fill = SLIP_COLOR)
        self.wheels["FL"].drawWheel(canvas, outline = FL_COLOR)
        self.wheels["FR"].drawWheel(canvas, outline = FR_COLOR)
        self.wheels["RL"].drawWheel(canvas, outline = RL_COLOR)
        self.wheels["RR"].drawWheel(canvas, outline = RR_COLOR)
        

    def updateLoads(self, loads):
        self.loadTransfer = loads
        return
    
    def loadTransferSSReset(self):
        self.loadTransferSS = {"ZF": 0, "ZR": 0, "FL": 0, "FR": 0, "RL": 0, "RR": 0, "w":0}
        return

    def getSlipAngles(self, tangentVelocity, angularVelocity, steeringAngles, cgSlipAngle):
        a = (1-self.forwardCG)*self.wheelBase
        b = self.forwardCG*self.wheelBase
        vx = tangentVelocity
        vy = tangentVelocity*math.tan(cgSlipAngle)
        tireSlipAngles = {}
        tireSlipAngles["FR"] = math.atan2((vy+angularVelocity*a),(vx + angularVelocity*self.halfTrack)) - steeringAngles["FR"]
        tireSlipAngles["FL"] = math.atan2((vy+angularVelocity*a),(vx - angularVelocity*self.halfTrack)) - steeringAngles["FL"]
        tireSlipAngles["RR"] = math.atan2((vy-angularVelocity*b),(vx + angularVelocity*self.halfTrack)) - steeringAngles["RR"]
        tireSlipAngles["RL"] = math.atan2((vy-angularVelocity*b),(vx - angularVelocity*self.halfTrack)) - steeringAngles["RL"]
        ptsa = {}
        for e in tireSlipAngles:
            ptsa[e] = math.degrees(tireSlipAngles[e])
        return tireSlipAngles

    def getTireLoads(self, lateralG):
        tireLoads = {"FR": self.forwardCG*self.weight/2,
                    "FL": self.forwardCG*self.weight/2,
                    "RR": (1-self.forwardCG)*self.weight/2,
                    "RL": (1-self.forwardCG)*self.weight/2}
        lateralLoadTransfer = self.CGHeight*lateralG/self.halfTrack

        tireLoads["FR"] *= (1-lateralLoadTransfer*self.totalLoadTransferDistribution)
        tireLoads["FL"] *= (1+lateralLoadTransfer*self.totalLoadTransferDistribution)
        tireLoads["RR"] *= (1-lateralLoadTransfer*(1-self.totalLoadTransferDistribution))
        tireLoads["RL"] *= (1+lateralLoadTransfer*(1-self.totalLoadTransferDistribution))
        s = 0

        for e in tireLoads.values():
            s+=e
        return tireLoads

    def getStaticLoads(self):
        z1 = self.forwardCG*self.mass*GRAVITY_ACCL/2
        z2 = (1-self.forwardCG)*self.mass*GRAVITY_ACCL/2
        return {"FR": z1, "FL": z1, "RR": z2, "RL": z2}

    def loadTransferEquations(self, vars, slipAngles, steer):
        dz1, dz2, dcamber1l, dcamber1r, dcamber2l, dcamber2r = vars

        fy = {
            "FL": -self.tire.getFy3(-slipAngles["FL"], -(self.staticLoads["FL"]-dz1), self.tireStaticCambers["FL"] + dcamber1l, self.tire.p0)*math.cos(steer["FL"]),
            "FR": self.tire.getFy3(slipAngles["FR"], -(self.staticLoads["FR"]+dz1), self.tireStaticCambers["FR"] + dcamber1r, self.tire.p0)*math.cos(steer["FR"]),
            "RL": -self.tire.getFy3(-slipAngles["RL"], -(self.staticLoads["RL"]-dz2), self.tireStaticCambers["RL"] + dcamber2l, self.tire.p0)*math.cos(steer["RL"]),
            "RR": self.tire.getFy3(slipAngles["RR"], -(self.staticLoads["RR"]+dz2), self.tireStaticCambers["RR"] + dcamber2r, self.tire.p0)*math.cos(steer["RR"])
        }

        lt = self.suspensionModel.getAxleLoadTransfer(fy)
        cg = self.suspensionModel.getCamberVariation(fy)

        eq1 = lt["F"] - dz1
        eq2 = lt["R"] - dz2
        eq3 = cg["FL"] - dcamber1l
        eq4 = cg["FR"] - dcamber1r
        eq5 = cg["RL"] - dcamber2l
        eq6 = cg["RR"] - dcamber2r
        return [eq1, eq2, eq3, eq4, eq5, eq6]

    def loadTransferEquationsSS(self, vars, cgSlip, steer, speed):
        dz1, dz2, dcamber1l, dcamber1r, dcamber2l, dcamber2r, w = vars

        slipAngles = self.getSlipAngles(speed, w, steer, cgSlip)
        fy = {
            "FL": -self.tire.getFy3(-slipAngles["FL"], -(self.staticLoads["FL"]-dz1), self.tireStaticCambers["FL"] + dcamber1l, self.tire.p0)*math.cos(steer["FL"]),
            "FR": self.tire.getFy3(slipAngles["FR"], -(self.staticLoads["FR"]+dz1), self.tireStaticCambers["FR"] + dcamber1r, self.tire.p0)*math.cos(steer["FR"]),
            "RL": -self.tire.getFy3(-slipAngles["RL"], -(self.staticLoads["RL"]-dz2), self.tireStaticCambers["RL"] + dcamber2l, self.tire.p0)*math.cos(steer["RL"]),
            "RR": self.tire.getFy3(slipAngles["RR"], -(self.staticLoads["RR"]+dz2), self.tireStaticCambers["RR"] + dcamber2r, self.tire.p0)*math.cos(steer["RR"])
        }

        lt = self.suspensionModel.getAxleLoadTransfer(fy)
        cg = self.suspensionModel.getCamberVariation(fy)
        f=0
        for e in fy.values():
            f+=e
        a = f*math.cos(cgSlip)/self.mass
        v = speed/math.cos(cgSlip)
        if a:
            r = (v**2)/a
            eq7 = (v/r)-w
        else:
            eq7 = -w

        eq1 = lt["F"] - dz1
        eq2 = lt["R"] - dz2
        eq3 = cg["FL"] - dcamber1l
        eq4 = cg["FR"] - dcamber1r
        eq5 = cg["RL"] - dcamber2l
        eq6 = cg["RR"] - dcamber2r
        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]

    def getLoadTransfer(self, slipAngles, steer):
        initialGuess = []
        for e in self.loadTransfer.values():
            initialGuess.append(e)
        solution = fsolve(self.loadTransferEquations, initialGuess, args =(slipAngles, steer))
        return {"ZF": solution[0], "ZR": solution[1], "FL": solution[2], "FR": solution[3], "RL": solution[4], "RR": solution[5]}
    
    def getLoadTransferSS(self, cgSlip, steer, speed):
        initialGuess = []
        for e in self.loadTransferSS.values():
            initialGuess.append(e)
        solution = fsolve(self.loadTransferEquationsSS, initialGuess, args =(cgSlip, steer, speed))
        return {"ZF": solution[0], "ZR": solution[1], "FL": solution[2], "FR": solution[3], "RL": solution[4], "RR": solution[5], "w": solution[6]}

    def getLoads(self, loadTransfer):
        tireLoads = {
            "FL": self.staticLoads["FL"] - loadTransfer["ZF"],
            "FR": self.staticLoads["FR"] + loadTransfer["ZF"],
            "RL": self.staticLoads["RL"] - loadTransfer["ZR"],
            "RR": self.staticLoads["RR"] + loadTransfer["ZR"]
        }
        return tireLoads

    def getSteeringAngles_old(self, steeringWheelAngle):
        al1 = self.steeringCoef[0] + self.steeringCoef[1]*math.radians(steeringWheelAngle)
        if al1 > 1:
            steeringWheelAngle = (1-self.steeringCoef[0])/self.steeringCoef[1]
        elif al1 < -1:
            steeringWheelAngle = -(1-self.steeringCoef[0])/self.steeringCoef[1]
        ar1 = self.steeringCoef[1]*math.radians(steeringWheelAngle)-self.steeringCoef[0]
        if ar1 > 1:
            steeringWheelAngle = (1-self.steeringCoef[0])/self.steeringCoef[1]
        elif ar1<-1:
            steeringWheelAngle = -(1-self.steeringCoef[0])/self.steeringCoef[1]
        al = math.asin(self.steeringCoef[0] + self.steeringCoef[1]*math.radians(steeringWheelAngle)) - math.asin(self.steeringCoef[0])
        ar = math.asin(self.steeringCoef[1]*math.radians(steeringWheelAngle)-self.steeringCoef[0]) - math.asin(-self.steeringCoef[0])
        steeringAngles = {"FR": ar + math.radians(self.toes["FR"]), "FL": al + math.radians(self.toes["FL"]), "RR": math.radians(0 + self.toes["RR"]), "RL": math.radians(0 + self.toes["RL"])}
        #print("steeringAngles:")
        #print(steeringAngles)
        return steeringAngles
    
    def getSteeringAngles(self, steeringWheelAngle):
        ack = self.ackermann*self.halfTrack*self.wheelBase
        steer = math.radians(steeringWheelAngle)
        ar = steer - ack*(steer*steer)
        al = steer + ack*(steer*steer)
        steeringAngles = {"FR": ar + math.radians(self.toes["FR"]), "FL": al + math.radians(self.toes["FL"]), "RR": math.radians(0 + self.toes["RR"]), "RL": math.radians(0 + self.toes["RL"])}
        return steeringAngles
    
    def getCambers(self, transfer):
        ret = {}

        for wheel in self.tireStaticCambers:
            ret[wheel] = -math.radians(self.tireStaticCambers[wheel] + transfer[wheel])
        
        return ret
    
    def getTotalLateralForce(self, tireSlipAngles, tireLoads, tireCambers, steeringAngles, cgSlip):
        total = 0
        for wheel in tireSlipAngles:
            parity = 1 - 2*(wheel[1]=="L")
            tireForce = parity*self.tire.getFy3(parity*tireSlipAngles[wheel], -tireLoads[wheel], tireCambers[wheel], self.tire.p0)
            #print("%s: %.1f"%(wheel, tireForce))
            total += math.cos(steeringAngles[wheel])*tireForce
        return total

    def getLateralForces(self, tireSlipAngles, tireLoads, tireCambers, steeringAngles, cgSlip):
        fy = {}
        for wheel in tireSlipAngles:
            parity = 1 - 2*(wheel[1]=="L")
            tireForce = parity*self.tire.getFy3(parity*tireSlipAngles[wheel], -tireLoads[wheel], tireCambers[wheel], self.tire.p0)
            #print("%s: %.1f"%(wheel, tireForce))
            fy[wheel] = math.cos(steeringAngles[wheel])*tireForce

        return fy

    def getYawMoment(self, tireSlipAngles, tireLoads, tireCambers, steeringAngles):       
        a = (1-self.forwardCG)*self.wheelBase
        b = self.forwardCG*self.wheelBase 
        chassiForces = {}
        for wheel in tireSlipAngles:
            parity = 1 - 2*(wheel[1]=="L")
            tireForce = parity*self.tire.getFy3(parity*tireSlipAngles[wheel], -tireLoads[wheel], tireCambers[wheel], self.tire.p0)
            chassiForces[wheel] = {"x": -math.sin(steeringAngles[wheel])*tireForce, "y": math.cos(steeringAngles[wheel])*tireForce}
        
        momentYF = (chassiForces["FR"]["y"] + chassiForces["FL"]["y"])*a
        momentYR = -(chassiForces["RR"]["y"] + chassiForces["RL"]["y"])*b
        momentXL = -(chassiForces["FL"]["x"]+chassiForces["RL"]["x"])*self.halfTrack
        momentXR = (chassiForces["FL"]["x"]+chassiForces["RL"]["x"])*self.halfTrack

        totalYawMoment = momentYF + momentYR + momentXL + momentXR
        return totalYawMoment
    
    def getLongitudinalLoadTransferCG(self, g):
        pos0 = self.forwardCG*self.wheelBase
        pos = -self.CGHeight*g + pos0
        return pos/self.wheelBase

def system(t, z, speed, steeringAngle, car, output):
    w,theta,phi = z
    cgSlip = phi-theta
    vx = speed
    vy = speed*math.tan(cgSlip)
    a = (1-car.forwardCG)*car.wheelBase
    steer0 = math.atan2((vy+w*a),vx) + math.radians(steeringAngle)
    steer = car.getSteeringAngles(math.degrees(math.sin(steer0)))
    car.steeringWheelAngle = math.degrees(math.sin(steer0))
    slipt = car.getSlipAngles(speed, w, steer, cgSlip)
    loadTransfer = car.getLoadTransfer(slipt, steer)
    loads = car.getLoads(loadTransfer)
    a = car.getTotalLateralForce(slipt, loads, car.getCambers(loadTransfer), steer, cgSlip)/car.mass
    g = -a/GRAVITY_ACCL
    #print("g: ", g)
    #car.updateLoads(g)
    #print("wheel loads: ", car.tireLoads)
    yawMoment = car.getYawMoment(slipt, loads, car.getCambers(loadTransfer), steer)
    wdot = yawMoment/car.yawMomentOfInertia #- 1e-1*w
    thetadot = w
    phidot = a/speed
    out = {"time": t,
           "yawMoment": yawMoment,
           "wdot": wdot, "lateralG": g,
           "carSlip": phi-theta,
           "yawVelocity": w,
           "slipAngles": slipt,
           "FY": car.getLateralForces(slipt, loads, car.getCambers(loadTransfer), steer, cgSlip),
           "FZ": loads,
           "steeringAngles": steer,
           "phidot": phidot,
           "radius": speed/(phidot+1e-15)}
    output.loc[len(output)] = out
    car.updateWheels(loads, slipt, steer)
    car.updateLoads(loadTransfer)
    return [wdot, thetadot, phidot]

def getTimeEvolution(speed, steeringAngle, car, output, t_max = 1, numpoints = 1000):
    z0 = [0.0,math.radians(0),math.radians(0)]
    params = (speed, steeringAngle, car, output)
    t_span = (0, t_max)
    t_eval = np.linspace(t_span[0], t_span[1], numpoints)

    sol = solve_ivp(
        fun = system,
        t_span=t_span,
        y0=z0,
        t_eval=t_eval,
        args=params,
        method="RK45"
    )
    return sol

def plotEvolution(speed, steeringAngle, car, t_max = 1.0, numpoints = 1000):
    columns = ["time", "yawMoment", "wdot", "lateralG", "carSlip", "yawVelocity", "slipAngles", "FY", "FZ", "steeringAngles", "phidot", "radius"]
    output = pd.DataFrame(columns = columns)
    sol = getTimeEvolution(speed, steeringAngle, car, output, t_max = t_max, numpoints = numpoints)
    car.angularVelocity = sol.y[0][-1]
    car.cgSlipAngle = sol.y[2][-1] - sol.y[1][-1]
    car.tangentVelocity = speed
    print(car.cgSlipAngle)
    print("car angular velocity")
    print(car.angularVelocity)
    print("speed")
    print(car.tangentVelocity)
    plt.figure()
    plt.plot(sol.t, sol.y[0])
    plt.plot(sol.t, sol.y[1])
    plt.plot(sol.t, sol.y[2])
    plt.show()
    print(output)
    for e in output.iloc[-1]["slipAngles"]:
        print(e)
        print(math.degrees(output.iloc[-1]["slipAngles"][e]))

    return

def loadTires():
    for tireFolder in rdm.getImmediateSubfolders("./data_files/tires"):
        print("update mf")
        mfPath = "%s/%s/MF"%("./data_files/tires",tireFolder)
        for f in rdm.getImmediateSubfolders(mfPath):
            name = f
            print(name)
            tempTyre = mf.MFTire(name, fyParameters = {}, path = "%s/%s"%(mfPath, name))
            tempTyre.loadLateralParameters(path = "%s/%s"%(mfPath, name))
            tires[name] = tempTyre
            tireSelectionBox['menu'].add_command(label=name, command=tk._setit(selections["tire"], name))
            selections["tire"].set(name)
    return

def plotAcclSteer():
    global car
    car.tire = tires[selections["tire"].get()]
    car.cgSlipAngle = math.radians(10)
    car.angularVelocity = 0
    car.tangentVelocity = 7
    car.steeringWheelAngle = 15
    car.mass = selections["mass"].get()
    car.forwardCG = selections["fowardCG"].get()
    car.CGHeight = selections["cgHeight"].get()
    car.halfTrack = selections["halfTrack"].get()
    car.wheelBase = selections["wheelBase"].get()
    car.totalLoadTransferDistribution = selections["loadTransferDistribution"].get()
    car.steeringCoef[0] = selections["ackermann"].get()
    columns = ["time", "yawMoment", "wdot", "lateralG", "carSlip", "yawVelocity", "slipAngles", "FY", "FZ", "steeringAngles", "phidot", "radius"]
    
    speed = selections["speed"].get()
    steer = selections["steering"].get()
    x = np.linspace(0.5, steer, 100)
    y1 = []
    y2 = []
    fig, ax = plt.subplots(num="lateralg x steering")
    parity = False
    for e in x:
        output = pd.DataFrame(columns = columns)
        sol = getTimeEvolution(speed, e, car, output, t_max = 1.0, numpoints = 250)
        y1.append(output.iloc[-1]["lateralG"])
        y2.append(output.iloc[-1]["radius"])
    
    acclChan = cm.Channel("a", color="C0", visibility=1)
    rChan = cm.Channel("r", color="C1", visibility=1)
    acclChan.setData(y1)
    rChan.setData(y2)
    parity = acclChan.plot(ax, xdata = x, parity = parity)
    parity = rChan.plot(ax, xdata = x, parity = parity)
    car.angularVelocity = sol.y[0][-1]
    car.cgSlipAngle = sol.y[2][-1] - sol.y[1][-1]
    car.tangentVelocity = speed
    ax.set_xlabel("steering angle")
    ax.set_title("lateralg x steering")
    fig.legend()
    fig.show()

def plotTimeEvolution():
    global car
    car.tire = tires[selections["tire"].get()]
    car.cgSlipAngle = math.radians(10)
    car.angularVelocity = 0
    car.tangentVelocity = 7
    car.steeringWheelAngle = 15
    car.mass = selections["mass"].get()
    car.forwardCG = selections["fowardCG"].get()
    car.CGHeight = selections["cgHeight"].get()
    car.halfTrack = selections["halfTrack"].get()
    car.wheelBase = selections["wheelBase"].get()
    car.totalLoadTransferDistribution = selections["loadTransferDistribution"].get()
    car.steeringCoef[0] = selections["ackermann"].get()
    plotEvolution(selections["speed"].get(), selections["steering"].get(), car)
    canvas.delete("all")
    car.draw(canvas)
    loads = car.getTireLoads(2)
    print(loads)
    

    
    return

def plotYawAccl(car, cgSlip, steer, speed):
    steerx = np.linspace(-35, 35, 500)
    slipx = np.linspace(-10, 10, 100)
    channels = []
    for sli in cgSlip:
        slip = math.radians(sli)
        a = []
        ym = []
        for s in steerx:
            st = car.getSteeringAngles(s)
            loadTransfer = car.getLoadTransferSS(slip, st, speed)
            slipt = car.getSlipAngles(speed, loadTransfer["w"], st, slip)
            loads = car.getLoads(loadTransfer)
            accl = car.getTotalLateralForce(slipt, loads, car.getCambers(loadTransfer), st, slip)/car.mass
            accl /= GRAVITY_ACCL
            ymom = car.getYawMoment(slipt, loads, car.getCambers(loadTransfer), st)
            a.append(accl)
            ym.append(ymom)
            car.loadTransferSS = loadTransfer     
        tempChan = cm.Channel("slip: %.1f"%(sli), xdata = a, data=ym, visibility=1, scaleble=False)
        channels.append(tempChan)
        car.loadTransferSSReset()
    for s in steer:
        a = []
        ym = []
        for sli in slipx:
            slip = math.radians(sli)
            #print(sli,slip)
            st = car.getSteeringAngles(s)
            loadTransfer = car.getLoadTransferSS(slip, st, speed)
            slipt = car.getSlipAngles(speed, loadTransfer["w"], st, slip)
            loads = car.getLoads(loadTransfer)
            accl = car.getTotalLateralForce(slipt, loads, car.getCambers(loadTransfer), st, slip)/car.mass
            accl /= GRAVITY_ACCL
            ymom = car.getYawMoment(slipt, loads, car.getCambers(loadTransfer), st)
            a.append(accl)
            ym.append(ymom)
            car.loadTransferSS = loadTransfer
        tempChan = cm.Channel("steer: %.1f"%(s), xdata = a, data=ym, visibility=1, scaleble=False)
        channels.append(tempChan)
        car.loadTransferSSReset()
    """
    """
    fig, ax = plt.subplots(num="YMD (const speed:%1.f)"%speed)
    for i,channel in enumerate(channels):
        channel.color = "C%d"%i
        channel.plot(ax)

    ax.set_xlabel("Lateral Acceleration [g]")
    ax.set_ylabel("Yaw Moment [Nm]")
    ax.set_title("YMD (const speed:%1.f)"%speed)
    fig.legend()
    fig.show()
        


    return

def plotYMD():
    global car
    global selections
    global car
    car.tire = tires[selections["tire"].get()]
    car.cgSlipAngle = math.radians(10)
    car.angularVelocity = 0
    car.tangentVelocity = 7
    car.steeringWheelAngle = 15
    car.mass = selections["mass"].get()
    car.forwardCG = selections["fowardCG"].get()
    car.CGHeight = selections["cgHeight"].get()
    car.halfTrack = selections["halfTrack"].get()
    car.wheelBase = selections["wheelBase"].get()
    car.totalLoadTransferDistribution = selections["loadTransferDistribution"].get()
    car.steeringCoef[0] = selections["ackermann"].get()
    slip = np.linspace(-5,5,10)
    steer = np.linspace(-35,35, 10)
    plotYawAccl(car, slip,steer,selections["speed"].get())
    return

def initializeUserInterface(root):
    global canvas
    global selections
    global tireSelectionBox
    global tires
    global car
    tires = {}

    titleFont = font.Font(family = "Times New Roman", size = 24, slant="italic")
    labelFont = font.Font(family = "Times New Roman", size = 12)

    buttonStyle = ttk.Style()
    buttonStyle.configure('TButton', font = labelFont)
    buttonStyle.configure('TMenubutton', font = labelFont)

    tireSelection = tk.StringVar(value="")
    massSelection = tk.DoubleVar(value = 300)
    forwardCGSelection = tk.DoubleVar(value = 0.5)
    cgHeightSelection = tk.DoubleVar(value = 0.3)
    wheelBaseSelection = tk.DoubleVar(value = 1.560)
    halfTrackSelection = tk.DoubleVar(value = 0.6)
    loadTransferDistributionSelection = tk.DoubleVar(value = 0.5)
    ackermannSelection = tk.DoubleVar(value = 0)
    steeringSelection = tk.DoubleVar(value = 8)
    speedSelection = tk.DoubleVar(value = 10)

    selections = {
        "tire":tireSelection,
        "mass": massSelection,
        "fowardCG": forwardCGSelection,
        "cgHeight": cgHeightSelection,
        "wheelBase": wheelBaseSelection,
        "halfTrack": halfTrackSelection,
        "loadTransferDistribution": loadTransferDistributionSelection,
        "ackermann": ackermannSelection,
        "speed": speedSelection,
        "steering": steeringSelection
    }

    root.geometry("%dx%d"%(WIDTH, HEIGHT))
    root.configure(bg= WINDOW_BACKGROUND_COLOR)

    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)

    configFrame = tk.Frame(root, bg = FRAME_BACKGROUND_COLOR)
    configFrame.grid(row = 0, column = 0, sticky = "nsew", padx=PADX, pady=PADY)

    for i in range(0,19):
        configFrame.rowconfigure(i, weight = 1)

    configFrame.columnconfigure(0, weight = 1)
    configFrame.columnconfigure(1, weight = 1)
    configFrame.columnconfigure(2, weight = 1)

    displayFrame = tk.Frame(root, bg = FRAME_BACKGROUND_COLOR)
    displayFrame.grid(row = 0, column = 1, sticky="nsew", padx=PADX, pady=PADY)
    displayFrame.rowconfigure(0, weight = 1)
    displayFrame.columnconfigure(0, weight = 1)

    canvas = tk.Canvas(displayFrame, width=WIDTH/2-2*PADX, height= HEIGHT-2*PADY, bg=CANVAS_BACKGROUND_COLOR)
    canvas.grid(row=0, column=0, sticky= "nsew", padx= PADX, pady= PADY)

    tireSelectionLabel = ttk.Label(configFrame, text = "Tire Selection", background = "lightblue", font = labelFont)
    tireSelectionLabel.grid(row = 0, column = 0, padx=PADX, pady=PADY, sticky="nsew")
    tireSelectionBox = ttk.OptionMenu(configFrame, tireSelection)
    tireSelectionBox.grid(row = 1, column = 0, padx=PADX, pady=PADY, sticky="nsew")

    massSelectionLabel = ttk.Label(configFrame, text = "Mass [kg]", background = "lightblue", font = labelFont)
    massSelectionLabel.grid(row = 2, column=0, padx=PADX, pady=PADY)
    massSelectionBox = ttk.Spinbox(configFrame, textvariable = massSelection, increment = 1, from_=0, to=500)
    massSelectionBox.grid(row = 3, column=0, padx=PADX, pady=PADY)

    fowardCGSelectionLabel = ttk.Label(configFrame, text = "weight distribution (Foward %)", background = "lightblue", font = labelFont)
    fowardCGSelectionLabel.grid(row = 4, column=0, padx=PADX, pady=PADY)
    fowardCGSelectionBox = ttk.Spinbox(configFrame, textvariable = forwardCGSelection, increment = 0.01, from_=0, to=1)
    fowardCGSelectionBox.grid(row = 5, column=0, padx=PADX, pady=PADY)

    cgHeightSelectionLabel = ttk.Label(configFrame, text = "CG Height", background = "lightblue", font = labelFont)
    cgHeightSelectionLabel.grid(row = 6, column=0, padx=PADX, pady=PADY)
    cgHeightSelectionBox = ttk.Spinbox(configFrame, textvariable = cgHeightSelection, increment = 0.05, from_=0, to=1)
    cgHeightSelectionBox.grid(row = 7, column=0, padx=PADX, pady=PADY)

    wheelBaseSelectionLabel = ttk.Label(configFrame, text = "WheelBase", background = "lightblue", font = labelFont)
    wheelBaseSelectionLabel.grid(row = 8, column=0, padx=PADX, pady=PADY)
    wheelBaseSelectionBox = ttk.Spinbox(configFrame, textvariable = wheelBaseSelection, increment = 0.1, from_=0, to=3)
    wheelBaseSelectionBox.grid(row = 9, column=0, padx=PADX, pady=PADY)

    halfTrackSelectionLabel = ttk.Label(configFrame, text = "halfTrack", background = "lightblue", font = labelFont)
    halfTrackSelectionLabel.grid(row = 10, column=0, padx=PADX, pady=PADY)
    halfTrackSelectionBox = ttk.Spinbox(configFrame, textvariable = halfTrackSelection, increment = 0.1, from_=0, to=2)
    halfTrackSelectionBox.grid(row = 11, column=0, padx=PADX, pady=PADY)

    loadTransferDistributionSelectionLabel = ttk.Label(configFrame, text = "Load Transfer Distribution (Front)", background = "lightblue", font = labelFont)
    loadTransferDistributionSelectionLabel.grid(row = 12, column=0, padx=PADX, pady=PADY)
    loadTransferDistributionSelectionBox = ttk.Spinbox(configFrame, textvariable = loadTransferDistributionSelection, increment = 0.01, from_=0, to=1)
    loadTransferDistributionSelectionBox.grid(row = 13, column=0, padx=PADX, pady=PADY)

    ackermannSelectionLabel = ttk.Label(configFrame, text = "Ackermann Ratio", background = "lightblue", font = labelFont)
    ackermannSelectionLabel.grid(row = 14, column=0, padx=PADX, pady=PADY)
    ackermannSelectionBox = ttk.Spinbox(configFrame, textvariable = ackermannSelection, increment = 0.1, from_=-10, to=10)
    ackermannSelectionBox.grid(row = 15, column=0, padx=PADX, pady=PADY)

    steeringSelectionLabel = ttk.Label(configFrame, text = "Steering", background = "lightblue", font = labelFont)
    steeringSelectionLabel.grid(row = 16, column=0, padx=PADX, pady=PADY)
    steeringSelectionBox = ttk.Spinbox(configFrame, textvariable = steeringSelection, increment = 2, from_=-40, to=40)
    steeringSelectionBox.grid(row = 17, column=0, padx=PADX, pady=PADY)
    
    speedSelectionLabel = ttk.Label(configFrame, text = "Speed", background = "lightblue", font = labelFont)
    speedSelectionLabel.grid(row = 16, column=1, padx=PADX, pady=PADY)
    speedSelectionBox = ttk.Spinbox(configFrame, textvariable = speedSelection, increment = 2, from_=0, to=40)
    speedSelectionBox.grid(row = 17, column=1, padx=PADX, pady=PADY)
    loadTires()

    car = Car(tires[selections["tire"].get()],
              mass = selections["mass"].get(),
              forwardCG = selections["fowardCG"].get(),
              cgHeight = selections["cgHeight"].get(),
              wheelBase = selections["wheelBase"].get(),
              halfTrack = selections["halfTrack"].get(),
              totalLoadTransferDistribution = selections["loadTransferDistribution"].get(),
              ackermann = selections["ackermann"].get(),
              pos = [WIDTH/4, HEIGHT/2])
    
    #-----------------------------Buttons-------------------------#
    plotTimeEvolutionButton = ttk.Button(configFrame, text="Plot Time Evolution", command = plotTimeEvolution)
    plotTimeEvolutionButton.grid(row = 18, column=0, padx=PADX, pady=PADY)

    plotLatGSteerButton = ttk.Button(configFrame, text="Lateral G x Steering Angle", command = plotAcclSteer)
    plotLatGSteerButton.grid(row = 18, column=1, padx=PADX, pady=PADY)

    plotYMDButton = ttk.Button(configFrame, text="YMD", command = plotYMD)
    plotYMDButton.grid(row = 18, column=2, padx=PADX, pady=PADY)

    return

def testModule():

    root = tk.Tk()
    initializeUserInterface(root)
    root.mainloop()
    """
    """
    return
 
testModule()

from dataclasses import dataclass, asdict
from scipy.optimize import least_squares
import numpy as np
import pandas as pd
import index_tables as it
import json
import raw_data_manager as rdm
import channel_module as cm
import math
import glob
import os
import matplotlib.pyplot as plt
import master_builder as mb

U_CORRECTION = 0.7

PURE_LATERAL_COLUMNS = ["SA","FY","FZ","IA","KAPPA","test_type","run_id","FZ_set","IA_set","V","P","TIRE_TEMP", "fz_clamped"]
FY_PARAMETERS_NAMES_STAGE1 = [
                        "pCy1",
                        "pDy1",
                        "pEy1","pEy3",
                        "pKy1",
                        "pHy1",
                        "pVy1"]
FY_PARAMETERS_NAMES_STAGE2 = [
                       "pCy1",
                        "pDy1","pDy2","pDy4", "pDy5",
                        "pEy1","pEy2","pEy3", "pEy6", "pEy7",
                       "pKy1","pKy2",
                        "pHy1","pHy2",
                        "pVy1","pVy2"]
FY_PARAMETERS_NAMES_STAGE3 = [
                        "pCy1",
                        "pDy1","pDy2","pDy3", "pDy4", "pDy5",
                        "pEy1","pEy2","pEy3","pEy4", "pEy5", "pEy6", "pEy7",
                        "pKy1","pKy2","pKy3", "pKy4", "pKy5", "pKy6", "pKy7", "pKy8", "pKy9",
                        "pHy1","pHy2",
                        "pVy1","pVy2","pVy3", "pVy4"]

FY_PARAMETERS_NAMES = ["pCy1",
                        "pDy1","pDy2","pDy3", "pDy4", "pDy5",
                        "pEy1","pEy2","pEy3","pEy4", "pEy5", "pEy6", "pEy7",
                        "pKy1","pKy2","pKy3", "pKy4", "pKy5", "pKy6", "pKy7", "pKy8", "pKy9",
                        "pHy1","pHy2",
                        "pVy1","pVy2","pVy3", "pVy4"
                        "pPy1", "pPy2", "pPy3", "pPy4", "pPy5"
                        ]



@dataclass
class LateralParams:
    P0: float = 83
    FZ0: float = -1500.0
    # Shape factor
    pCy1: float = 1.35
    # Peak (D) load & camber dependence
    pDy1: float = 1
    pDy2: float = 0.1
    pDy3: float = 0.0
    pDy4: float = -0
    pDy5: float = 0.0
    # Curvature (E) dependencies
    pEy1: float = -1.6
    pEy2: float = 0.2
    pEy3: float = 0.0
    pEy4: float = -0
    pEy5: float = 0.0
    pEy6: float = -0.0
    pEy7: float = 0.0
    # Stiffness (Ky) baseline and dependencies -> used to compute B
    pKy1: float = -18.0
    pKy2: float = 1
    pKy3: float = -0
    pKy4: float = 2
    pKy5: float = 0.5
    pKy6: float = 0.0
    pKy7: float = -0.0
    pKy8: float = -0.00
    pKy9: float = -0.00

    pPy1: float = 0
    pPy2: float = 0
    pPy3: float = 0
    pPy4: float = 0
    pPy5: float = 0.0
    
    # Horizontal/vertical shifts
    pHy1: float = 0.0
    pHy2: float = 0.0
    pVy1: float = 0.0
    pVy2: float = 0.0
    pVy3: float = 0.0
    pVy4: float = 0.0


def save_params_to_json(params_dataclass, path):
    """Saves a parameter dataclass to a JSON file."""
    print(f"Saving parameters to {path}...")
    try:
        with open(path, 'w') as f:
            json.dump(asdict(params_dataclass), f, indent=4)
        print("Save complete.")
    except Exception as e:
        print(f"Error saving parameters: {e}")

def saveTyre(tyre, path):
    tyre.save_params_to_json(path=path)
    return

def loadTyre(path, name):
    tyre = MFTire(name, path = path)
    tyre.loadLateralParameters()
    return tyre

def load_params_from_json(ParamsClass, path):
    """Loads parameters from JSON and returns the specific dataclass."""
    print(f"Loading parameters from {path}...")
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        # Ensure FZ0 from the *current data* is used, overriding the saved one
        return ParamsClass(**data)
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return None

def small_angle_stiffness(alpha, Fy, max_abs_alpha=np.deg2rad(1.5)):
    m = np.abs(alpha) < max_abs_alpha
    if m.sum() < 8:
        return np.nan
    # simple least-squares slope through origin
    a = alpha[m]; y = Fy[m]
    return float(np.dot(a, y) / (np.dot(a, a) + 1e-12))

def guess_params(latData, FZ0):
    u = -max(np.abs(latData["FY"]/FZ0))
    D = -max(np.abs(latData["FY"]))
    extremes = latData[np.abs(np.abs(latData["SA"]) - np.abs(latData["SA"]))<0.25]
    ya = np.mean(np.abs(extremes["FY"]/extremes["FZ"]))
    C = 1+(1-(2/math.pi)*math.asin(-ya/u))
    a = np.deg2rad(latData["SA"]).to_numpy()
    fy = latData["FY"].to_numpy()
    K = small_angle_stiffness(a, fy)
    B = K/(C*D + 1e-12)
    xm = latData[latData["FY"] == latData["FY"].min()]
    xm = xm["SA"]
    xm = math.radians(xm)
    E = (B*xm - math.tan(math.pi/(2*C)))/(B*xm - math.atan(B*xm))
    origin = latData[latData["SA"]<0.1]
    y0 = 0
    print("ya = %f, xm = %f, C = %f, y0 = %f, u = %f"%(ya, xm, C, y0, u))
    p0  = p0 = LateralParams(
        FZ0=FZ0,
        pDy1 = u,
        pCy1 = C,
        pKy1 = K/FZ0,
        pEy1 = E,
        pHy1 = 0.0, pVy1 = y0
    )

    return p0

def guess_params_old(lat_df, FZ0):
    # bucket by run (or by FZ_set, IA_set) to sample across conditions
    groups = lat_df.groupby("run_id")
    K_list, D_list, z_list, g_list = [], [], [], []
    for _, g in groups:
        if len(g) < 200:
            continue
        a = g["SA"].to_numpy()
        a = np.deg2rad(a)
        fy = g["FY"].to_numpy()
        FZ = float(np.nanmedian(g["FZ"]))
        IA = float(np.nanmedian(g["IA"]))
        K = small_angle_stiffness(a, fy)
        if not np.isfinite(K):
            continue
        D = float(np.nanmax(np.abs(fy)))
        if D < 100:
            continue
        K_list.append(K); D_list.append(D); z_list.append((FZ - FZ0)/FZ0); g_list.append(IA)
    # Robust medians
    Ky0 = np.nanmedian(K_list) / (FZ0 + 1e-9)
    Dy1 = np.nanmedian([D/(FZ0+1e-9) for D in D_list])
    p0 = LateralParams(
        FZ0=FZ0,
        pDy1 = Dy1,
        pCy1 = 1.0,
        pKy1 = 0.3,
        pEy1 = 1.6,
        pHy1 = 0.0, pVy1 = 0.0
    )
    print("Ky0:%f"%Ky0)
    print(p0)
    print(lat_df)
    return p0

def p_to_vec(p: LateralParams, names):
    return np.array([getattr(p, k) for k in names], dtype=float)

def vec_to_p1(v, FZ0):
    kwargs = dict(zip(FY_PARAMETERS_NAMES_STAGE1, v))
    kwargs["FZ0"] = FZ0

    return LateralParams(**kwargs)

def vec_to_p2(v, FZ0):
    kwargs = dict(zip(FY_PARAMETERS_NAMES_STAGE2, v))
    kwargs["FZ0"] = FZ0

    return LateralParams(**kwargs)

def vec_to_p3(v, FZ0):
    kwargs = dict(zip(FY_PARAMETERS_NAMES_STAGE3, v))
    kwargs["FZ0"] = FZ0

    return LateralParams(**kwargs)

def vec_to_p(v, FZ0):
    kwargs = dict(zip(FY_PARAMETERS_NAMES, v))
    kwargs["FZ0"] = FZ0
    return LateralParams(**kwargs)

def getPureLateral(data):
    for c in PURE_LATERAL_COLUMNS:
        if c not in data.columns: data[c] = np.nan
    # Use pure lateral runs (and κ≈0 as a safety net)
    lat = data[data["test_type"].isin(["lateral_pure","combined"])].copy()
    lat = lat[np.isfinite(lat["SA"]) & np.isfinite(lat["FY"]) & np.isfinite(lat["FZ"]) & np.isfinite(lat["IA"])]
    lat = lat[(np.abs(lat["KAPPA"].fillna(0.0)+1) < 0.01) | (np.abs(lat["KAPPA"].fillna(0.0)))]  # ~1% slip ≈ pure lateral
    return lat
def mf_shape(x, B, C, D, E, SH=0.0, SV=0.0):
    x = np.tan(x)
    xs = x + SH
    return D * np.sin(C * np.arctan(B*xs - E*(B*xs - np.arctan(B*xs)))) + SV


class MFTire:
    def __init__(self, name, path = None, fz0=-1112, fyParameters = None, fxParameters = None, data = None, p = 83, blockGuide = None):
        if path is None:
            path = "./data_files/tires/%s"%name
        if fyParameters is None:
            fyParameters = {}
        self.path = path
        self.data = data
        self.blockGuide = blockGuide
        self.fyParameters = fyParameters
        self.name = name
        self.fz0 = fz0
        self.p0 = p
        self.fyActiveWeight = "W"
        self.fyActiveParameters = "simple"
        if data is None: return
        self.updateFyDataArrays(data)
        self.fyWeights = self.getFyWeights()
        return
    
    def updateFyDataArrays(self, data):
        self.pureLateralData = getPureLateral(data)
        filter = cm.LowPassFilter(100, 2, 3)
        #self.pureLateralData["FY"] = filter.filter(self.pureLateralData["FY"])
        self.fyDataArrays = {"A": self.pureLateralData["SA"].to_numpy(), "FZ": self.pureLateralData["FZ"].to_numpy(), 
                             "GA": self.pureLateralData["IA"].to_numpy(), "FY": self.pureLateralData["FY"].to_numpy(), "T": self.pureLateralData["TIRE_TEMP"], "fz_clamped": self.pureLateralData["fz_clamped"]}
        self.fyDataArrays["A_rad"] = np.deg2rad(self.fyDataArrays["A"])
        self.fyDataArrays["GA_rad"] = np.deg2rad(self.fyDataArrays["GA"])
        self.fyDataArrays["FZ"] = filter.filter(self.fyDataArrays["FZ"])

    def getMu(self, FZ, gamma, P):
        gammaStar = np.sin(gamma)
        p0 = self.fyParameters[self.fz0]["simple"]
        dp0 = (P - p0.P0)/p0.P0
        dfz0 = (FZ - p0.FZ0) / p0.FZ0
        miy0 = (p0.pDy1 + p0.pDy2*dfz0 + p0.pDy4*dfz0*dfz0 + p0.pDy5*dfz0*dfz0*dfz0)*(1+ p0.pPy3*dp0 + p0.pPy4*dp0**2)*(1-p0.pDy3*gammaStar**2)
        Svg = (p0.pVy3 + p0.pVy4*dfz0)*gammaStar
        SV = (p0.pVy1 + p0.pVy2*dfz0) + Svg
        return miy0*U_CORRECTION + SV

    def getMu_old(self, FZ, gamma, P):
        gammaStar = np.sin(gamma)
        fz0 = self.getFzInterpolationInterval(FZ)
        p0 = self.fyParameters[fz0[0]]["simple"]
        p1 = self.fyParameters[fz0[1]]["simple"]
        dp0 = (P - p0.P0)/p0.P0
        dfz0 = (FZ - p0.FZ0) / p0.FZ0
        miy0 = (p0.pDy1 + p0.pDy2*dfz0 + p0.pDy4*dfz0*dfz0 + p0.pDy5*dfz0*dfz0*dfz0)*(1+ p0.pPy3*dp0 + p0.pPy4*dp0**2)*(1-p0.pDy3*gammaStar**2)
        dp1 = (P - p1.P0)/p1.P0
        dfz1 = (FZ - p1.FZ0) / p1.FZ0
        miy1 = (p1.pDy1 + p1.pDy2*dfz1 + p1.pDy4*dfz1*dfz1 + p1.pDy5*dfz1*dfz1*dfz1)*(1+ p1.pPy3*dp1 + p1.pPy4*dp1**2)*(1-p1.pDy3*gammaStar**2)
        if fz0[0] != fz0[1]:
            tfz = -float(fz0[1]-FZ)/float(fz0[0]-fz0[1])
            miy = (math.cos(tfz*math.pi/2)**2)*miy1 + (math.sin(tfz*math.pi/2)**2)*miy0
        else:
            miy = miy1
        Svg = (p0.pVy3 + p0.pVy4*dfz0)*gammaStar
        SV = (p0.pVy1 + p0.pVy2*dfz0) + Svg
        return miy + SV

    def mfFromParameters(self, alpha, FZ, gamma, P, p):
        gammaStar = np.sin(gamma)
        dp = (P - p.P0)/p.P0
        dfz = (FZ - p.FZ0) / p.FZ0
        C = p.pCy1
        miy = (p.pDy1 + p.pDy2*dfz + p.pDy4*dfz*dfz + p.pDy5*dfz*dfz*dfz)*(1+ p.pPy3*dp + p.pPy4*dp**2)*(1-p.pDy3*gammaStar**2)
        D = miy*FZ
        E = (p.pEy1 + p.pEy2*dfz + p.pEy6*dfz*dfz + p.pEy7*dfz*dfz*dfz)*(1+p.pEy5*gammaStar**2 - (p.pEy3+p.pEy4*gammaStar)*np.sign(alpha))
        Ky = p.pKy1*p.FZ0*(1+p.pPy1*dp)*(1-p.pKy3*abs(gammaStar))*np.sin(p.pKy4*np.atan((FZ/p.FZ0)/((p.pKy2 + p.pKy5*gammaStar**2 +1e-9)*1+p.pPy2*dp)))
        B = Ky / (C * (np.abs(D) + 1e-9))
        epslonK = 1e-9
        Kyg0 = FZ*(p.pKy6 + p.pKy7*dfz + p.pKy8*dfz*dfz + p.pKy9*dfz*dfz*dfz)*(1+p.pPy5*dp)
        Svg = FZ*(p.pVy3 + p.pVy4*dfz)*gammaStar
        SV = FZ*(p.pVy1 + p.pVy2*dfz) + Svg
        SH = (p.pHy1 + p.pHy2*dfz) + (Kyg0*gammaStar-Svg)/(Ky + epslonK)
        E = np.where(E>1, 1, E)
        return mf_shape(alpha, B, C, D, E, SH, SV)
    
    def mfFromParameters2(self, alpha, FZ, gamma, P, p):
        gammaStar = np.sin(gamma)
        dp = (P - p.P0)/p.P0
        dfz = (FZ - p.FZ0) / p.FZ0
        C = p.pCy1
        miy = (p.pDy1 + p.pDy2*dfz + p.pDy4*dfz*dfz + p.pDy5*dfz*dfz*dfz)*(1+ p.pPy3*dp + p.pPy4*dp**2)*(1-p.pDy3*gammaStar**2)
        D = miy*FZ*U_CORRECTION
        E = (p.pEy1 + p.pEy2*dfz + p.pEy6*dfz*dfz + p.pEy7*dfz*dfz*dfz)*(1+p.pEy5*gammaStar**2 - (p.pEy3+p.pEy4*gammaStar)*np.sign(alpha))
        Ky = p.pKy1*p.FZ0*(1+p.pPy1*dp)*(1-p.pKy3*abs(gammaStar))*np.sin(p.pKy4*np.atan((FZ/p.FZ0)/((p.pKy2 + p.pKy5*gammaStar**2 +1e-9)*1+p.pPy2*dp)))
        B = Ky / (C * (np.abs(D) + 1e-9))
        epslonK = 1e-9
        Kyg0 = FZ*(p.pKy6 + p.pKy7*dfz + p.pKy8*dfz*dfz + p.pKy9*dfz*dfz*dfz)*(1+p.pPy5*dp)
        Svg = FZ*(p.pVy3 + p.pVy4*dfz)*gammaStar
        SV = FZ*(p.pVy1 + p.pVy2*dfz) + Svg
        SH = (p.pHy1 + p.pHy2*dfz) + (Kyg0*gammaStar-Svg)/(Ky + epslonK)
        E = np.where(E>1, 1, E)
        return mf_shape(alpha, B, C, D, E, SH, SV)

    def getFzInterpolationInterval(self, FZ):
        fz0 = list(it.fzSelectionValues.values())
        if FZ >= fz0[0]: return [fz0[0], fz0[0]]
        if FZ <= fz0[-1]: return [fz0[-1], fz0[-1]]

        for i in range(1,len(fz0)):
            a = fz0[i-1]
            b = fz0[i]
            t1 = FZ-a
            t2 = FZ-b
            if t1 * t2 < 0: return [a, b]
        return [self.fz0,self.fz0]

    def printMFP(self, alpha, FZ, gamma, P):
        p = self.fyParameters[self.fz0]["simple"]
        gammaStar = np.sin(gamma)
        dp = (P - p.P0)/p.P0
        dfz = (FZ - p.FZ0) / p.FZ0
        C = p.pCy1
        miy = (p.pDy1 + p.pDy2*dfz + p.pDy4*dfz*dfz + p.pDy5*dfz*dfz*dfz)*(1+ p.pPy3*dp + p.pPy4*dp**2)*(1-p.pDy3*gammaStar**2)
        D = miy*FZ
        E = (p.pEy1 + p.pEy2*dfz + p.pEy6*dfz*dfz)*(1+p.pEy5*gammaStar**2 - (p.pEy3+p.pEy4*gammaStar)*np.sign(alpha))
        Ky = p.pKy1*p.FZ0*(1+p.pPy1*dp)*(1-p.pKy3*abs(gammaStar))*np.sin(p.pKy4*np.atan((FZ/p.FZ0)/((p.pKy2 + p.pKy5*gammaStar**2 +1e-9)*1+p.pPy2*dp)))
        B = Ky / (C * (np.abs(D) + 1e-9))
        epslonK = 1e-9
        Kyg0 = FZ*(p.pKy6 + p.pKy7*dfz +p.pKy8*dfz*dfz)*(1+p.pPy5*dp)
        Svg = FZ*(p.pVy3 + p.pVy4*dfz)*gammaStar
        SV = FZ*(p.pVy1 + p.pVy2*dfz) + Svg
        SH = (p.pHy1 + p.pHy2*dfz) + (Kyg0*gammaStar-Svg)/(Ky + epslonK)
        print("B:%f"%B)
        print("C:%f"%C)
        print("D:%f"%D)
        print("E:%f"%E)
        print("SH:%f"%SH)
        print("SV:%f"%SV)
        return

    def getFy2(self, alpha, FZ, gamma, P):
        fz0 = self.getFzInterpolationInterval(FZ)
        sa = math.degrees(alpha)

        ph0 = self.fyParameters[fz0[0]]["simple"]
        fh0 = self.mfFromParameters(alpha, FZ, gamma, P, ph0)

        ph1 = self.fyParameters[fz0[1]]["simple"]
        fh1 = self.mfFromParameters(alpha, FZ, gamma, P, ph1)
        if fz0[0] != fz0[1]:
            tfz = -(fz0[1]-FZ)/(fz0[0]-fz0[1])
            az0 = (np.sin(tfz*np.pi/2))**2
            az1 = (np.cos(tfz*np.pi/2))**2
            fh = az1*fh1 + az0*fh0
        else:
            fh = fh1
        
        if abs(sa)>12:
            return fh
        pl0 = self.fyParameters[fz0[0]]["simple"]
        fl0 = self.mfFromParameters(alpha, FZ, gamma, P, pl0)

        pl1 = self.fyParameters[fz0[1]]["simple"]
        fl1 = self.mfFromParameters(alpha, FZ, gamma,  P, pl1)
        if fz0[0]!=fz0[1]:
            fl = az1*fl1 + az0*fl0
        else: fl = fl1
        
        t = abs(sa/12)
        al = (np.cos(t*np.pi/2))**2
        ah = (np.sin(t*np.pi/2))**2
        print(t)
        return al*fl + ah*fh

    def getFy(self, alpha, FZ, gamma, P):
        p = self.fyParameters[self.fz0][self.fyActiveParameters]
        return self.mfFromParameters(alpha, FZ, gamma, P, p)
    def getFy3(self, alpha, FZ, gamma, P):
        p = self.fyParameters[self.fz0][self.fyActiveParameters]
        return self.mfFromParameters2(alpha, FZ, gamma, P, p)
    
    def getCorneringStiffness(self, alpha, FZ, gamma, P, dx):
        f1 = self.getFy2(alpha+dx/2, FZ, gamma, P)
        f2 = self.getFy2(alpha-dx/2, FZ, gamma, P)
        return (f1-f2)/dx
    
    def getCamberStiffness(self, alpha, FZ, gamma, P, dx):
        f1 = self.getFy(alpha, FZ, gamma+dx/2, P)
        f2 = self.getFy(alpha, FZ, gamma-dx/2, P)
        return (f1-f2)/dx

    def getFyWeightsold(self):
        print(self.fyDataArrays["FY"])
        w_small = 1.0 / (1.0 + (self.fyDataArrays["A_rad"]/np.deg2rad(2.0))**2)
        peak = np.maximum(0.0, np.abs(self.fyDataArrays["FY"])/np.nanmax(np.abs(self.fyDataArrays["FY"])))
        W = 100*w_small + 0.1*peak
        weights = {"w_small": w_small, "W": W}
        return weights
    
    def getFyWeights(self):
        print(self.fyDataArrays["FY"])
        w_small = 1.0 / (1.0 + (0.5*self.fyDataArrays["A_rad"]/np.deg2rad(2.0))**2)
        w_big = 1.0 / (1.0 + (0.4*(np.abs(12)-np.abs(self.fyDataArrays["A"])))**2)
        peak = np.maximum(0.0, np.abs(self.fyDataArrays["FY"])/np.nanmax(np.abs(self.fyDataArrays["FY"])))
        W = 1
        W1 = 0.9*w_small + 0.1
        W2 = 0.1+0.9*w_big
        weights = {"w_small": w_small, "W": W, "W1": W1, "W2": W2}
        
        return weights
    
    def residuals1(self, theta):
        p = vec_to_p1(theta, self.fz0)
        self.fyParameters[self.fz0][self.fyActiveParameters] = p
        pred = self.getFy(self.fyDataArrays["A_rad"], self.fyDataArrays["fz_clamped"], self.fyDataArrays["GA_rad"], p.P0)
        return (pred - self.fz0*self.fyDataArrays["FY"]/self.fyDataArrays["FZ"]) * self.fyWeights[self.fyActiveWeight]
    def residuals2(self, theta):
        p = vec_to_p2(theta, self.fz0)
        self.fyParameters[self.fz0][self.fyActiveParameters] = p
        pred = self.getFy(self.fyDataArrays["A_rad"], self.fyDataArrays["FZ"], self.fyDataArrays["GA_rad"], p.P0)
        return (pred - self.fyDataArrays["FY"]) * self.fyWeights[self.fyActiveWeight]
    def residuals3(self, theta):
        fzw = 1/(1+((self.fyDataArrays["FZ"]-self.fz0))**2)
        p = vec_to_p3(theta, self.fz0)
        self.fyParameters[self.fz0][self.fyActiveParameters] = p
        pred = self.getFy(self.fyDataArrays["A_rad"], self.fyDataArrays["FZ"], self.fyDataArrays["GA_rad"], p.P0)
        return (pred - self.fyDataArrays["FY"]) * self.fyWeights[self.fyActiveWeight] #* fzw

    def calculateSubFyParameters(self):
        # --- STAGE 1: Find the SHAPE (fast, with tight bounds) ---

        print("Starting Stage 1: Fitting shape parameters...")
        
        data = self.data
        blocks = self.blockGuide[(self.blockGuide["FZ"] == self.fz0) & (self.blockGuide["IA"] == 0)]
        data = data[data["block_id"].isin(blocks["block_id"])]
        print(" data FZ ")
        print(data["FZ"])
        self.updateFyDataArrays(data)
        self.fyWeights = self.getFyWeights()

        p0 = guess_params(data, self.fz0)
        self.fyParameters[self.fz0][self.fyActiveParameters] = p0
        plat1 = p0
        """
        lo_stage1 = np.array([
                            max(1, p0.pCy1-0.4),#C
                            max(-10.0, p0.pDy1 - 0.8),#D1
                            -2.0, #E1
                            -4.0, #E3
                            0.2,#K1 
                            -0.100,#H1
                            -0.100,#V1
                            ])
        
        hi_stage1 = np.array([
                            min(1.9, p0.pCy1+0.4),#C
                            min(-1.0, p0.pDy1+0.8),#D1
                            1.0, #E1
                            4.00, #E3
                            300.0,#K1
                            0.100,#H1
                            0.100,#V1
                            ])
        theta0_stage1 = p_to_vec(p0, FY_PARAMETERS_NAMES_STAGE1) # p0 comes from guess_params()
        print(theta0_stage1)

        res_stage1 = least_squares(self.residuals1, theta0_stage1, bounds=(lo_stage1, hi_stage1),
                                loss="soft_l1", f_scale=0.5, max_nfev=20000, verbose=2, ftol=1e-9)

        print("\nStage 1 complete. Now starting Stage 2...")

        plat1 = vec_to_p1(res_stage1.x, self.fz0)
        print(plat1)
        self.fyParameters[self.fz0][self.fyActiveParameters] = plat1
        """
        lo_stage2 = np.array([
                            plat1.pCy1 - 0.2,#C
                            plat1.pDy1-1,#D1
                            -3,#D2
                            -2,#D4
                            -0.5,#D5
                            plat1.pEy1-0.2, #E1
                            -4, #E2
                            plat1.pEy3-1.2, #E3
                            -10, #E6
                            -4, #E7
                            plat1.pKy1-00.1,#K1
                            0.00001,#K2
                            -0.100,#H1
                            -0.100,#H2
                            -0.100,#V1
                            -0.010,#V2
                            ])
        
        hi_stage2 = np.array([
                            plat1.pCy1+0.2,#C
                            plat1.pDy1+1,#D1
                            3,#D2
                            2,#D4
                            0.5,#D5
                            plat1.pEy1+0.2, #E1
                            4, #E2
                            plat1.pEy3+1.2, #E3
                            10, #E6
                            4, #E7
                            plat1.pKy1+0.01,#K1
                            10.0,#K2
                            0.100,#H1
                            0.100,#H2
                            0.100,#V1
                            0.010,#V2
                            ])
        """
        """
        theta0_stage2 = p_to_vec(self.fyParameters[self.fz0][self.fyActiveParameters], FY_PARAMETERS_NAMES_STAGE2) # p0 comes from guess_params()
        res_stage2 = least_squares(self.residuals2, theta0_stage2, bounds=(lo_stage2, hi_stage2),
                                loss="soft_l1", f_scale=0.5, max_nfev=200, verbose=2, ftol=1e-9)
        plat2 = vec_to_p2(res_stage2.x, self.fz0)
        plat2.pKy4 = 0.5*math.pi/math.atan(1/plat2.pKy2)
        print(plat2)
        self.fyParameters[self.fz0][self.fyActiveParameters] = plat2
        data = self.data
        blocks = self.blockGuide[(self.blockGuide["IA"] == 0)]
        data = data[data["block_id"].isin(blocks["block_id"])]
        self.updateFyDataArrays(data)
        self.fyWeights = self.getFyWeights()
        print(theta0_stage2)

        print("\nStage 1 complete. Now starting Stage 2 to correct offset...")
        theta0_stage2 = p_to_vec(self.fyParameters[self.fz0][self.fyActiveParameters], FY_PARAMETERS_NAMES_STAGE2) # p0 comes from guess_params()
        res_stage2 = least_squares(self.residuals2, theta0_stage2, bounds=(lo_stage2, hi_stage2),
                                loss="soft_l1", f_scale=0.5, max_nfev=200, verbose=2, ftol=1e-9)
        plat2 = vec_to_p2(res_stage2.x, self.fz0)
        plat2.pKy4 = 0.5*math.pi/math.atan(1/plat2.pKy2)
        print(plat2)
        self.fyParameters[self.fz0][self.fyActiveParameters] = plat2

        lo_stage3 = np.array([
                            plat2.pCy1-0.00,#C
                            plat2.pDy1-0.0,#D1
                            plat2.pDy2-0.0,#D2
                            0.0,#D3
                            plat2.pDy4-0,#D4
                            plat2.pDy5-0.1,#D5
                            plat2.pEy1-10, #E1
                            plat2.pEy2-10, #E2
                            plat2.pEy3-10, #E3
                            -100.0, #E4
                            -100.0, #E5
                            plat2.pEy6-10, #E6
                            plat2.pEy7-10, #E7
                            plat2.pKy1-00.0,#K1
                            plat2.pKy2-0.00,#K2
                            -100.0,#K3
                            1.5,#K4
                            -500,#K5
                            -5, #K6
                            -10,#K7
                            -100,#K8
                            -10,#K9
                            plat2.pHy1-0.0010,#H1
                            plat2.pHy2-0.0010,#H2
                            plat2.pVy1-0.0010,#V1
                            plat2.pVy2-0.0010,#V2
                            -10,#V3
                            -10,#V4
                            ])
        
        hi_stage3 = np.array([
                            plat2.pCy1+0.001,#C
                            plat2.pDy1+0.000001,#D1
                            plat2.pDy2+0.000001,#D2
                            100.0,#D3
                            plat2.pDy4+0.00001,#D4
                            plat2.pDy5+0.1,#D5
                            plat2.pEy1+10, #E1
                            plat2.pEy2+10, #E2
                            plat2.pEy3+10, #E3
                            100, #E4
                            1000, #E5
                            plat2.pEy6+10, #E6
                            plat2.pEy7+10, #E7
                            plat2.pKy1+0.000001,#K1
                            plat2.pKy2+0.0000001,#K2
                            100,#K3
                            10,#K4
                            500,#K5
                            2.0, #K6
                            10.0,#K7
                            100.0,#K8
                            10.0,#K9
                            plat2.pHy1+0.0010,#H1
                            plat2.pHy2+0.0010,#H2
                            plat2.pVy1+0.001,#V1
                            plat2.pVy2+0.001,#V2
                            10,#V3
                            10,#V4
                            ])
        data = self.data
        self.updateFyDataArrays(data)
        self.fyWeights = self.getFyWeights()
        theta0_stage3 = p_to_vec(plat2, FY_PARAMETERS_NAMES_STAGE3) # p0 comes from guess_params()
        print(theta0_stage3)
        res_stage3 = least_squares(self.residuals3, theta0_stage3, bounds=(lo_stage3, hi_stage3),
                                loss="soft_l1", f_scale=0.5, max_nfev=200, verbose=2, ftol=1e-9)

        print("\nStage 1 complete. Now starting Stage 2 to correct offset...")
        plat3 = vec_to_p3(res_stage3.x, self.fz0)
        print(plat3)
        self.fyParameters[self.fz0][self.fyActiveParameters] = plat3
        """
        """
        return

    def calculateFyParameters(self):
        for e in it.fzSelectionValues.values():
            if e != -1112:
                continue
            self.fz0 = e
            self.fyParameters[e] = {}
            self.fyActiveWeight = "W"
            self.fyActiveParameters = "simple"
            self.calculateSubFyParameters()
            """
            print(self.fyParameters)
            self.fyActiveWeight = "W1"
            self.fyActiveParameters = "low"
            self.calculateSubFyParameters()

            self.fyActiveWeight = "W2"
            self.fyActiveParameters = "high"
            self.calculateSubFyParameters()
            """
            """
        plt.figure()
        plt.plot(self.fyDataArrays["A"], self.fyWeights["W1"])
        plt.plot(self.fyDataArrays["A"], self.fyWeights["W2"])
        plt.show()
            """
        return
    
    def saveLateralParameters(self, path = None):
        if path is None: path = self.path
        name = self.name
        name = name.replace(':',"_")
        for fz in it.fzSelectionValues:
            if not fz == -1112: continue

            path2 = "%s/MF/%s/%s"%(path, name, fz)
            path2 = path2.strip("./")
            print(path2)
            os.makedirs(path2, exist_ok = True)
            for e in self.fyParameters[it.fzSelectionValues[fz]]:
                print(self.fyParameters[it.fzSelectionValues[fz]][e])
                save_params_to_json(self.fyParameters[it.fzSelectionValues[fz]][e], "%s/%s.mfp"%(path2, e))
        return
    
    def loadLateralParameters(self, path = None):
        if path is None:
            path = self.path
            path = "%s/MF/%s/"%(path, self.name)
        for fz in it.fzSelectionValues:
            path2 = "%s/%s"%(path, fz)
            pattern = os.path.join(path2, f"*{".mfp"}")
            files = [os.path.basename(file) for file in glob.glob(pattern)]
            self.fyParameters[it.fzSelectionValues[fz]] = {}
            for e in files:
                p = "%s/%s"%(path2, e)
                index = e.strip(".mfp")
                self.fyParameters[it.fzSelectionValues[fz]][index] = load_params_from_json(LateralParams, p)
                print(index)
        return


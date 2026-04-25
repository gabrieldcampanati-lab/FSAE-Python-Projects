import pandas as pd
import numpy as np
from pathlib import Path

import master_builder as mb
import index_tables as it
import raw_data_manager as rdm

# Adjust this to your TTC data folder
TIRE_NAME = "43100 18.0x6.0-10 R20_rim7" 
DATA_ROOT = Path(".datafiles/tires")  # same as in raw_data_manager.py
MASTER_FILE = DATA_ROOT / TIRE_NAME / "masterall.csv"
BLOCK_GUIDE = DATA_ROOT / TIRE_NAME / "blockGuide.csv"

# Choose test conditions
TARGET_PRESSURE_PSI = 14
TARGET_CAMBER_DEG = 0
TARGET_SPEED = 40  # mph or kph depending on TTC (check master_builder)

def load_master():
    df = pd.read_csv(MASTER_FILE, sep=",")
    return df

def filter_longitudinal(df):
    """
    Use the same classification logic as master_builder: 'longitudinalpure'.
    """
    return df[df["testtype"] == "longitudinalpure"].copy()

def filter_conditions(df):
    """
    Filter for desired pressure, camber and speed.
    master_builder snaps these to discrete values via pressureSelectionValues etc.
    """
    # Map 14 psi to its internal value (97) using index_tables
    # In index_tables: pressureSelectionValues: 8 psi -> 55, 10 psi -> 69, 12 psi -> 83, 14 psi -> 97
    pressure_map = it.pressureSelectionValues
    P_target = pressure_map["14 psi"]

    camber_map = it.inclinationAngleSelectionValues
    IA_target = camber_map["0 deg"]

    # If master_builder snaps velocity as well, use a similar map; otherwise use a tolerance
    # For now, we’ll just use a tolerance around TARGET_SPEED
    df_cond = df.copy()
    df_cond = df_cond[df_cond["Pset"] == P_target]
    df_cond = df_cond[df_cond["IAset"] == IA_target]

    # Speed tolerance (adjust as needed)
    df_cond = df_cond[(df_cond["Vset"] >= TARGET_SPEED - 5) &
                      (df_cond["Vset"] <= TARGET_SPEED + 5)]

    return df_cond

def compute_mu_peaks(df):
    """
    Compute peak |mu_x| = |Fx/Fz| vs Fz using load bins.
    FZ in TTC is negative (downwards), so we use abs().
    """
    df = df.copy()
    df["FZ_pos"] = df["FZ"].abs()
    df["mu_x"] = df["FX"] / df["FZ_pos"]

    # Define load bins based on TTC test values
    # index_tables.fzSelectionValues: "1112 N" -> -1112, etc.
    fz_values = np.sort(np.abs(list(it.fzSelectionValues.values)))
    # Create bins halfway between test values
    edges = np.concatenate((
        [fz_values[0] - 0.5 * (fz_values[1] - fz_values[0])],
        0.5 * (fz_values[:-1] + fz_values[1:]),
        [fz_values[-1] + 0.5 * (fz_values[-1] - fz_values[-2])]
    ))
    df["FZ_bin"] = pd.cut(df["FZ_pos"], bins=edges)

    mu_peaks = (
        df.groupby("FZ_bin")
          .apply(lambda g: g["mu_x"].abs().max())
          .reset_index(name="mu_peak")
    )

    mu_peaks["FZ_center"] = mu_peaks["FZ_bin"].apply(
        lambda b: 0.5 * (b.left + b.right)
    )

    # Drop bins with NaN (no data)
    mu_peaks = mu_peaks.dropna(subset=["mu_peak"])

    return mu_peaks

def build_mu_of_Fz(mu_peaks):
    """
    Returns a callable mu(Fz) using linear interpolation.
    """
    from scipy.interpolate import interp1d

    Fz_points = mu_peaks["FZ_center"].values
    mu_points = mu_peaks["mu_peak"].values

    mu_of_Fz = interp1d(
        Fz_points, mu_points,
        kind="linear",
        fill_value="extrapolate"
    )

    def mu_func(Fz):
        # Fz can be vector or scalar, assumed positive
        Fz = np.asarray(Fz)
        return mu_of_Fz(Fz)

    return mu_func

def main():
    df_master = load_master()
    df_long = filter_longitudinal(df_master)
    df_cond = filter_conditions(df_long)

    print("Rows after filtering:", len(df_cond))

    mu_peaks = compute_mu_peaks(df_cond)
    print(mu_peaks[["FZ_center", "mu_peak"]])

    mu_Fz = build_mu_of_Fz(mu_peaks)

    # Example: print μ at 1112 N
    mu_1112 = float(mu_Fz(1112.0))
    print(f"Estimated peak mu at 1112 N: {mu_1112:.3f}")

    # Save to CSV for documentation
    mu_peaks.to_csv(DATA_ROOT / TIRE_NAME / "mu_vs_Fz.csv", index=False)

if __name__ == "__main__":
    main()

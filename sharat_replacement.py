import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def analyze_impedance(file_path):
    # Load data
    with open(file_path, 'r') as f:
        skip = next(i for i, line in enumerate(f) if 'Frequency (Hz)' in line)
    df = pd.read_csv(file_path, skiprows=skip, index_col=False)
    df.columns = df.columns.str.strip()
    
    # Calculate Z components, filter to ≤15 MHz
    f = df['Frequency (Hz)'].values
    z_mag = df['Impedance Magnitude (Ohms)'].values
    theta = np.deg2rad(df["Impedance Phase Degrees (')"].values)
    zr, zi = z_mag * np.cos(theta), z_mag * np.sin(theta)
    
    mask = f <= 15e6
    f, zr, zi = f[mask], zr[mask], zi[mask]
    
    # Fit L-RC on >1 MHz
    m = f > 1e6
    w = lambda freq: 2*np.pi*freq
    
    def lrc(freq, L, R, C):
        wf, wRC = w(freq), w(freq)*R*C
        return np.r_[R/(1+wRC**2), wf*L - wf*R**2*C/(1+wRC**2)]
    
    L, R, C = curve_fit(lrc, f[m], np.r_[zr[m], zi[m]], p0=[1e-7, 300, 1e-11])[0]
    print(f"L-RC: L={L:.2e}, R={R:.1f}, C={C:.2e}")
    
    # Subtract L, refit RC
    zi_corr = zi - w(f)*L
    
    def rc(freq, R, C):
        wRC = w(freq)*R*C
        return np.r_[R/(1+wRC**2), -w(freq)*R**2*C/(1+wRC**2)]
    
    R, C = curve_fit(rc, f[m], np.r_[zr[m], zi_corr[m]], p0=[R, C])[0]
    print(f"RC: R={R:.1f}, C={C:.2e}")
    
    # Extend to 10 GHz (decreasing frequency)
    f_ext = np.logspace(10, np.log10(1.5e7), int((10 - np.log10(1.5e7))*16))
    wRC_ext = w(f_ext)*R*C
    zr_ext, zi_ext = R/(1+wRC_ext**2), -w(f_ext)*R**2*C/(1+wRC_ext**2)
    
    # Save: simulated first, then corrected experimental
    pd.concat([
        pd.DataFrame({'f': f_ext, "z'": zr_ext, "z''": zi_ext}),
        pd.DataFrame({'f': f, "z'": zr, "z''": zi_corr})
    ]).to_csv("processed_impedance_data.txt", sep='\t', index=False)
    
    plt.figure(figsize=(6, 6))
    plt.plot(zr, -zi_corr, 'o', label='Exp (<15MHz)')
    plt.plot(zr_ext, -zi_ext, '-', label='Sim (>15MHz)')
    plt.xlabel("Z' (Ω)"); plt.ylabel("-Z'' (Ω)"); plt.legend(); plt.grid(True)
    plt.show()

analyze_impedance("MgO_10MPa_6mm_RT_10mV_10PPD_after 100mV step.csv")
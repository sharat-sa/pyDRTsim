import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

def analyze(path):
    p = Path(path)
    print(f"Processing: {p.name}")
    
    with open(p) as f: skip = next(i for i, l in enumerate(f) if 'Freq' in l)
    df = pd.read_csv(p, skiprows=skip, index_col=False); df.columns = df.columns.str.strip()
    
    f, z = df['Frequency (Hz)'].values, df['Impedance Magnitude (Ohms)'].values * np.exp(1j * np.deg2rad(df["Impedance Phase Degrees (')"].values))

    # Filter <= 13MHz
    m = f <= 13e6; f, z, w = f[m], z[m], 2*np.pi*f[m]
    
    # Fit L-RC (>1MHz)
    fm = f > 1e6
    def model(w, L, R, C): Z = 1j*w*L + R/(1+1j*w*R*C); return np.r_[Z.real, Z.imag]
    
    (L, R, C), _ = curve_fit(lambda x,L,R,C: model(w[fm],L,R,C), f[fm], np.r_[z[fm].real, z[fm].imag], p0=[1e-7, 300, 1e-11])
    print(f"  L-RC: L={L:.2e}, R={R:.1f}, C={C:.2e}")

    # Refit RC (Corrected)
    z_corr = z - 1j*w*L
    (R, C), _ = curve_fit(lambda x,R,C: model(w[fm],0,R,C), f[fm], np.r_[z_corr[fm].real, z_corr[fm].imag], p0=[R, C])
    print(f"  RC:   R={R:.1f}, C={C:.2e}")

    # Extend & Save
    f_ex = np.logspace(10, np.log10(1.3e7), int((10 - np.log10(1.3e7))*16))
    z_ex = R / (1 + 1j*(2*np.pi*f_ex)*R*C)
    
    pd.concat([pd.DataFrame({'f': f_ex, "z'": z_ex.real, "z''": z_ex.imag}),
               pd.DataFrame({'f': f, "z'": z_corr.real, "z''": z_corr.imag})]).to_csv(p.with_suffix('.txt'), sep='\t', index=False, header=False)

    plt.figure()
    plt.plot(z_corr.real, -z_corr.imag, 'o', label='Exp'); plt.plot(z_ex.real, -z_ex.imag, '-', label='Sim')
    plt.legend(); plt.grid(); plt.savefig(p.with_suffix('.png')); plt.close()

if __name__ == "__main__":
    root = tk.Tk(); root.withdraw()
    files = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])
    for file in files: analyze(file)
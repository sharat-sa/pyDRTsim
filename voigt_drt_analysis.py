import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('pyDRTtools-master')
from pyDRTtools.runs import EIS_object, simple_run

# Define Voigt circuit parameters
R_ohmic = 10  # Ohmic resistance (Ohm)

# List of (R, C) pairs for RC elements in parallel
RC_pairs = [
    (50, 1e-4),  # R1 = 50 Ohm, C1 = 1e-4 F
    # Add more pairs as needed
]

# Frequency range settings
log_freq_min = -2.0  # log10 of min frequency (Hz)
log_freq_max_initial = 3.0  # log10 of initial max frequency (Hz)
points_per_decade = 10  # points per decade

# Number of iterations (how many times to reduce max frequency)
num_iterations = 50  # Adjust as needed

# Arrays to store results
peak_time_constants = []
max_frequencies = []

# Initial max frequency
current_log_max = log_freq_max_initial

for i in range(num_iterations):
    # Calculate number of frequency points
    N_freqs = points_per_decade * int(current_log_max - log_freq_min) + 1
    
    # Generate frequency vector
    freq_vec = np.logspace(log_freq_min, current_log_max, num=N_freqs, endpoint=True)
    
    # Calculate impedance
    Z = R_ohmic * np.ones_like(freq_vec, dtype=complex)
    for R, C in RC_pairs:
        Z += R / (1 + 1j * 2 * np.pi * freq_vec * R * C)
    
    # Create EIS object
    eis = EIS_object(freq_vec, Z.real, Z.imag)
    
    # Run DRT analysis
    simple_run(eis, cv_type='custom', reg_param=0.05)  # Using custom regularization with lambda = 0.05
    
    # Get DRT output
    gamma = eis.gamma
    tau_vec = eis.out_tau_vec
    
    # Find the peak
    idx_max = np.argmax(gamma)
    tau_peak = tau_vec[idx_max]
    
    # Store results
    peak_time_constants.append(tau_peak)
    max_frequencies.append(10**current_log_max)
    
    # Reduce max frequency by 1%
    current_log_max *= 0.99

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(max_frequencies, peak_time_constants, 'o-', markersize=4)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Maximum Frequency (Hz)')
plt.ylabel('Peak Time Constant (s)')
plt.title('Peak Time Constant vs Maximum Frequency')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

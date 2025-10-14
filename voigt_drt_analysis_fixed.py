import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('pyDRTtools-master')
from pyDRTtools.runs import EIS_object, simple_run

# Define Voigt circuit parameters
R_ohmic = 10  # Ohmic resistance (Ohm)

# List of (R, C) pairs for RC elements in parallel -> #######Expand to L, RQ later######
RC_pairs = [
    (500, 1e-5),  # R1 = 50 Ohm, C1 = 1e-4 F
    # Add more pairs as needed
]

# Frequency range settings
log_freq_min = -2.0  # log10 of min frequency (Hz)
log_freq_max_initial = 3.0  # log10 of initial max frequency (Hz)
points_per_decade = 16  # points per decade

# Number of iterations (how many times to reduce max frequency)
num_iterations = 40

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
    
    # Add noise to the impedance ## remove later, modify pyDRT to remove regu ##
    sigma_n_exp = 0.5
    Z += sigma_n_exp * (np.random.normal(0, 1, N_freqs) + 1j * np.random.normal(0, 1, N_freqs))
    
    # Create EIS object
    eis = EIS_object(freq_vec, Z.real, Z.imag)
    
    # Run DRT analysis
    simple_run(eis, cv_type='custom', reg_param=1e-4)  # Using custom regularization with lambda = 1e-4
    
    # Get DRT output
    gamma = eis.gamma
    tau_vec = eis.out_tau_vec
    
    # Find the peak  ##multi-peak later##
    idx_max = np.argmax(gamma)
    tau_peak = tau_vec[idx_max]
    
    # Store results
    peak_time_constants.append(tau_peak)
    max_frequencies.append(10**current_log_max)
    
    # Reduce max frequency by 1% ##check later for point by point removal##
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

# Add markers for characteristic frequency
for i, (R, C) in enumerate(RC_pairs):
    f_char = 1 / (2 * np.pi * R * C)
    t = R*C
    plt.axvline(x=f_char, linestyle='--', color='red', label=f'Characteristic Frequency = {f_char:.2e} Hz')
    plt.axhline(y=t, color='black', label='Actual Time Constant')

plt.legend()
plt.tight_layout()
plt.show()

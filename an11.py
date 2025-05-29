import numpy as np
import matplotlib.pyplot as plt

# === Grid setup ===
x = np.linspace(150, 250, 1000)
t = np.linspace(4.5, 5.5, 400)
dx = x[1] - x[0]
dt = t[1] - t[0]

# === φ(t) trajectory
def phi_exp_structure(t, slope=0.002, t0=4.9):
    return 1.0 + slope * (t - t0)

# === Flat potential
def V_flat_general(phi, V0=1.0, a=0.0001):
    return V0 / (1.0 + a * phi)**2

# === Sech² envelope
def sech2_envelope(phi, p=3.0):
    norm_phi = phi / np.max(phi)
    return 1 / np.cosh(p * (norm_phi - 0.5))**2

# === Main simulation
def run_structure_simulation():
    a = 0.0001
    p = 3.0
    k = 26
    N_pulses = 1000
    sigma_x = 0.08
    noise_eps = 0.001

    phi_t = phi_exp_structure(t)
    dphi_dt = np.gradient(phi_t, dt)
    V_target = V_flat_general(phi_t, a=a)
    H_target = np.sqrt((0.5 * dphi_dt**2 + V_target) / 3)
    envelope = sech2_envelope(phi_t, p=p)

    x_centers = np.linspace(180, 220, N_pulses)
    psi_total = np.zeros_like(x)
    for x0 in x_centers:
        pulse = np.exp(-((x - x0)**2) / (2 * sigma_x**2))
        psi_total += pulse
    psi_total += np.random.normal(0, noise_eps, size=psi_total.shape)
    psi_total /= np.sqrt(np.trapz(psi_total**2, x))

    delta_xt = np.sqrt(V_target)[:, None] * (envelope[:, None] * psi_total[None, :])
    phi_eff = np.trapz(delta_xt, x, axis=1)
    dphi_eff = np.gradient(phi_eff, dt)
    V_eff = V_flat_general(phi_eff, a=a)
    dV_dphi = -2 * a * V_eff / (1 + a * phi_eff)
    H_t = np.trapz(delta_xt**2, x, axis=1)
    rho_t = 0.5 * dphi_eff**2 + V_eff
    friedmann_error = np.abs(H_target**2 - rho_t / 3) / np.maximum(H_target**2, 1e-20)
    epsilon = np.abs(dV_dphi / np.maximum(V_eff, 1e-16))
    w_t = (dphi_eff**2 - V_eff) / (dphi_eff**2 + V_eff + 1e-16)
    mse_phi_eff = np.mean((phi_eff - phi_t)**2)

    print("=== Final Optimized Structure Compression ===")
    print(f"  a = {a}, p = {p}, k = {k}, N = {N_pulses}, σₓ = {sigma_x}, noise = {noise_eps}")
    print(f"  MSE[φ_eff - φ(t)]   = {mse_phi_eff:.6e}")
    print(f"  Max ε(t)           = {np.max(epsilon):.6f}")
    print(f"  Max Friedmann err  = {np.max(friedmann_error) * 100:.4f} %")
    print(f"  Mean w(t)          = {np.mean(w_t):.6f}")

    return t, phi_t, phi_eff, epsilon, w_t, friedmann_error

# === Run
t_vals, phi_t, phi_eff, epsilon_t, w_t, friedmann_error = run_structure_simulation()

# === Plotting
plt.figure(figsize=(10, 4))
plt.plot(t_vals, phi_t, '--', label='φ(t) input')
plt.plot(t_vals, phi_eff, '-', label='φ_eff(t)')
plt.title("φ_eff(t) vs φ(t) — Final Optimized Compression")
plt.xlabel("Time t")
plt.ylabel("Field Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t_vals, epsilon_t, label='ε(t)')
plt.axhline(0.1, color='gray', linestyle='--')
plt.title("ε(t) — Slow-roll Parameter")
plt.xlabel("Time t")
plt.ylabel("ε")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t_vals, w_t, label='w(t)')
plt.axhline(-1, color='gray', linestyle='--')
plt.title("w(t) — Equation of State")
plt.xlabel("Time t")
plt.ylabel("w")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
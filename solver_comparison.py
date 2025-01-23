from target_simulation import Target, Simulation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

#Simulation parameters
T = 100
dt = 0.01
x0, y0 = 0, 0
vx, vy = 1, 1
ax_amp = 3
ay_amp = 3


def simulate_with_scipy(target, dt, T):
    """ using SciPy's solve_ivp as a reference solution."""
    t_span = (0, T)
    t_eval = np.arange(0, T, dt)
    
    initial_state = np.concatenate([target.position, target.velocity])

    def ode_system(t, state):
        pos = state[:2]
        vel = state[2:]
        acc = target.get_acceleration(t, vel)
        return np.concatenate([vel, acc])

    solution = solve_ivp(ode_system, t_span, initial_state, t_eval=t_eval, method='RK45')

    positions = solution.y[:2].T
    velocities = solution.y[2:].T
    energies = 0.5 * (velocities[:, 0]**2 + velocities[:, 1]**2)  # Compute kinetic energy
    

    return positions, energies


# --- Euler ---
target_euler = Target(x0, y0, vx, vy, ax_amp, ay_amp)
sim_euler = Simulation(dt, T, 'euler', target_euler)
positions_euler, energies_euler = sim_euler.run_simulation()

# --- RK4 ---
target_rk4 = Target(x0, y0, vx, vy, ax_amp, ay_amp)
sim_rk4 = Simulation(dt, T, 'rk4', target_rk4)
positions_rk4, energies_rk4 = sim_rk4.run_simulation()

# --- SciPy Reference Solver ---
target_scipy = Target(x0, y0, vx, vy, ax_amp, ay_amp)
positions_scipy, energies_scipy = simulate_with_scipy(target_scipy, dt, T)

# Plotting trajectories
plt.figure(figsize=(8, 6))
plt.plot(positions_euler[:, 0], positions_euler[:, 1], label='Euler', linestyle='-')
plt.plot(positions_rk4[:, 0], positions_rk4[:, 1], label='RK4', linestyle='--')
plt.plot(positions_scipy[:, 0], positions_scipy[:, 1], label='SciPy RK45', linestyle='-.')
plt.xlabel('x position')
plt.ylabel('y position')
plt.legend()
plt.title('Trajectory Comparison: Euler vs RK4 vs SciPy RK45')
plt.grid(True)
plt.show()

# Final positions for numeric comparison
print(f"Final Euler position: {positions_euler[-1]}")
print(f"Final RK4 position:   {positions_rk4[-1]}")
print(f"Final SciPy position: {positions_scipy[-1]}")

# energy growth over time
time_steps = np.arange(len(energies_euler))

plt.figure(figsize=(8, 6))
plt.plot(time_steps, energies_euler, label='Euler Method', linestyle='-')
plt.plot(time_steps, energies_rk4, label='RK4 Method', linestyle='--')
plt.plot(time_steps, energies_scipy, label='SciPy RK45', linestyle='-.')
plt.xlabel('Time Step')
plt.ylabel('Energy')
plt.title('Comparison of Energy Growth Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

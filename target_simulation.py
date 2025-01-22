import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

"""we will construct a moving target in space (and time), with a radar placed at the origin
the radar will be fed signals from the moving target - with some noise - and will through state space modelling 
seek to predict the target's next position"""




class Target():

    def __init__(self, x0, y0, vx, vy, ax_amp, ay_amp = 0):
        self.x0 = x0
        self.y0 = y0
        self.vx = vx
        self.vy = vy
        self.ax_amp = ax_amp
        self.ay_amp = ay_amp

        self.position = np.array([x0, y0], dtype=float) 
        self.positions = None
        self.velocity = np.array([vx, vy], dtype=float) 
        self.velocities = None  # unused for now; for when velocity is non constant


    def get_acceleration(self, time, velocity): 
        drag_coefficient = 0.2
        velocity_magnitude = np.linalg.norm(velocity)
        drag_force = -drag_coefficient * velocity_magnitude * velocity
        return np.array([
            drag_force[0] + self.ax_amp * np.sin(0.5*time),
            drag_force[1] + self.ay_amp * np.cos(0.5*time)
        ])
    

    def setup(self, N):
        self.positions = np.zeros((N, 2))
        self.velocities = np.zeros((N,2))
        self.positions[0] = self.position
        self.velocities[0] = self.velocity
        self.step = 0

    def euler_update(self, dt): 
        time = self.step * dt
        acceleration = self.get_acceleration(time,  self.velocity)
        self.velocity += acceleration * dt  
        self.position += self.velocity * dt  
        self.positions[self.step] = self.position
        self.velocities[self.step] = self.velocity
        self.step += 1




    def rk4_update(self, dt):
        current_step = self.step
        time = current_step * dt
        state = np.concatenate([self.position, self.velocity])

        def state_derivative(t, state_vec):
            pos = state_vec[:2]
            vel = state_vec[2:]
            acc = self.get_acceleration(t, vel)
            return np.concatenate([vel, acc])

        k1 = dt * state_derivative(time, state)
        k2 = dt * state_derivative(time + 0.5*dt, state + 0.5*k1)
        k3 = dt * state_derivative(time + 0.5*dt, state + 0.5*k2)
        k4 = dt * state_derivative(time + dt, state + k3)

        state += (k1 + 2*k2 + 2*k3 + k4) / 6.0

        self.position = state[:2].copy()
        self.velocity = state[2:].copy()
        self.positions[self.step] = self.position
        self.velocities[self.step] = self.velocity
        self.step += 1







    def kinetic_energy(self, mass=1.0):
        speed = np.linalg.norm(self.velocity)
        return 0.5 * mass * speed ** 2




    def state_update(self, dt, method):

        if method == 'euler':
            self.euler_update(dt)
        elif method == 'rk4':
            self.rk4_update(dt)

        else:
            return None


class Simulation():

    def __init__(self, dt, T, method, target):
        self.dt = dt
        self.T = T
        self.N = int(T / dt)
        self.method = method
        self.target = target

        self.target.setup(self.N)
        self.energies = []

    def run_simulation(self):
        times = np.linspace(0, self.T, self.N)

        for _ in range(self.N):
            self.target.state_update(self.dt, self.method)
            self.energies.append(self.target.kinetic_energy())
        return self.target.positions, self.energies






def error_convergence():
    T = 5
    dts = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    x0, y0, vx, vy = 0, 0, 3, 3
    ax_amp = ay_amp = 10

    dt_ref = 1e-6
    target_ref = Target(x0, y0, vx, vy, ax_amp, ay_amp)
    sim_ref = Simulation(dt_ref, T, 'rk4', target_ref)
    pos_ref = sim_ref.run_simulation()[0]

    errors_euler = []
    errors_rk4 = []

    for dt in dts:
        # Euler
        target_euler = Target(x0, y0, vx, vy, ax_amp, ay_amp)
        sim_euler = Simulation(dt, T, 'euler', target_euler)
        pos_euler = sim_euler.run_simulation()[0]
        error_euler = np.sqrt(np.sum((pos_ref[-1] - pos_euler[-1])**2))
        errors_euler.append(error_euler)

        # RK4
        target_rk4 = Target(x0, y0, vx, vy, ax_amp, ay_amp)
        sim_rk4 = Simulation(dt, T, 'rk4', target_rk4)
        pos_rk4 = sim_rk4.run_simulation()[0]
        error_rk4 = np.sqrt(np.sum((pos_ref[-1] - pos_rk4[-1])**2))
        errors_rk4.append(error_rk4)


    plt.figure(figsize=(10,6))
    plt.loglog(dts, errors_euler, 'o-', label='Euler')
    plt.loglog(dts, errors_rk4, 'o-', label='RK4')
    plt.loglog(dts, [dt for dt in dts], '--', label='O(dt)')
    plt.loglog(dts, [dt**4 for dt in dts], '--', label='O(dt⁴)')
    plt.xlabel('dt')
    plt.ylabel('Error (Euclidean distance)')
    plt.legend()
    plt.grid(True)
    plt.title('Error Convergence: Euler vs RK4')
    plt.show()

    return dts, errors_euler, errors_rk4













def main():

    # Simulation parameters
    T = 100
    dt = 0.01
    x0, y0 = 0, 0
    vx, vy = 3, 3
    ax_amp = 10
    ay_amp = 10




    # --- Euler ---
    target_euler = Target(x0, y0, vx, vy, ax_amp, ay_amp)
    sim_euler = Simulation(dt, T, 'euler', target_euler)
    positions_euler, energies_euler = sim_euler.run_simulation()

    # --- RK4 ---
    target_rk4 = Target(x0, y0, vx, vy, ax_amp, ay_amp)
    sim_rk4 = Simulation(dt, T, 'rk4', target_rk4)
    positions_rk4, energies_rk4 = sim_rk4.run_simulation()

    # Plot trajectories
    plt.figure(figsize=(8, 6))
    plt.plot(positions_euler[:, 0], positions_euler[:, 1], label='Euler')
    plt.plot(positions_rk4[:, 0], positions_rk4[:, 1], label='RK4')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.legend()
    plt.title('Comparison of Euler vs. RK4')
    plt.grid(True)
    plt.show()




    error_convergence()



if __name__ == '__main__':
    main()


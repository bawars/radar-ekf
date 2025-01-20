import numpy as np
import matplotlib.pyplot as plt

"""we will construct a moving target in space (and time), with a radar placed at the origin
the radar will be fed signals from the moving target - with some noise - and will through state space modelling 
seek to predict the target's next position"""


# we define our radar position. a given coordinate is of the form (x,y)

# radar_position = np.array([0,0])


class Target():

    def __init__(self, x0, y0, vx, vy, ax_amp, ay_amp = 0):
        self.x0 = x0
        self.y0 = y0
        self.vx = vx
        self.vy = vy
        self.ax_amp = ax_amp
        self.ay_amp = ay_amp

        self.position = np.array([x0, y0], dtype=float)  # current position
        self.positions = None
        self.velocity = np.array([vx, vy], dtype=float)  # constant velocity
        self.velocities = None  # unused for now; for when velocity is non constant


    def get_acceleration(self,dt):
        time = self.step * dt
        self.acceleration = np.array([self.ax_amp * np.sin(time), self.ay_amp * np.sin(time)])
        return self.acceleration

    def setup(self, N):
        self.positions = np.zeros((N, 2))
        self.velocities = np.zeros((N,2))
        self.positions[0] = self.position
        self.velocities[0] = self.velocity
        self.step = 1

    def euler_update(self, dt):  # we start with a simple euler forward algorithm
        acceleration = self.get_acceleration(dt)

        self.velocity += acceleration * dt  # first update velocity, then position below

        self.position += self.velocity * dt
        self.positions[self.step] = self.position
        self.velocities[self.step] = self.velocity
        self.step += 1

    def rk4_update(self, dt):
        def f(state, time):
            pos = state[:2]
            vel = state[2:]
            acceleration = np.array([
                self.ax_amp * np.sin(time),
                self.ay_amp * np.sin(time)
            ])
            dpos = vel
            dvel = acceleration
            return np.concatenate((dpos, dvel))

        time = self.step * dt
        state = np.concatenate((self.position, self.velocity))

        k1 = f(state, time)
        k2 = f(state + 0.5 * dt * k1, time + 0.5 * dt)
        k3 = f(state + 0.5 * dt * k2, time + 0.5 * dt)
        k4 = f(state + dt * k3, time + dt)

        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        self.position = new_state[:2]
        self.velocity = new_state[2:]

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

        self.target.setup(self.N)  # we initialize posiitons array of dimension N here
        self.energies = []

    def run_simulation(self):
        times = np.linspace(0, self.T, self.N)

        for _ in range(self.N - 1):
            self.target.state_update(self.dt, self.method)
            self.energies.append(self.target.kinetic_energy())
        return self.target.positions, self.energies




def error_check():
    trajectories = {}

    analytic_solutions = {}

    errors = []

    x0, y0 = 0, 0
    vx, vy = 1, 1
    ax_amp, ay_amp = 0.5, 0.5
    T = 200
    dt_values = [0.2, 0.1, 0.05, 0.025, 0.001]

    # Lists to store dt and corresponding errors for plotting
    dts = []
    average_errors = []
    max_errors = []
    # dt_values = [0.1]

    for dt in dt_values:
        N = int(T / dt)
        # print(N)
        # euler int
        tar = Target(x0, y0, vx, vy, ax_amp, ay_amp)
        sim = Simulation(dt, T, method='euler', target=tar)
        sim.run_simulation()

        trajectories[dt] = tar.positions.tolist()

        times = np.linspace(0, T, N)
        analytic_positions = []
        for t in times:
            x = x0 + vx * t - (ax_amp * np.cos(t) - ax_amp)
            y = y0 + vy * t - (ay_amp * np.cos(t) - ay_amp)

            analytic_positions.append([x, y])
        analytic_solutions[dt] = analytic_positions

        num = np.array(trajectories[dt])
        analytic = np.array(analytic_solutions[dt])

        error_per_timestep = np.linalg.norm(num - analytic[:len(num)], axis=1)
        average_error = np.mean(error_per_timestep)
        max_error = np.max(error_per_timestep)

        # Store for plotting
        dts.append(dt)
        average_errors.append(average_error)
        max_errors.append(max_error)
        print("Final position (RK4):", tar.positions[-1])
        print("Final velocity (RK4):", tar.velocity)

    # Create log-log plot
    plt.figure(figsize=(10, 6))
    plt.loglog(dts, average_errors, 'bo-', label='Average Error')
    plt.loglog(dts, max_errors, 'ro-', label='Maximum Error')

    # Add reference line (slope = 1)
    reference_x = np.array([min(dts), max(dts)])
    reference_y = reference_x * average_errors[-1] / dts[-1]  # Scale to match our data
    plt.loglog(reference_x, reference_y, 'k--', label='Reference (slope = 1)')

    plt.xlabel('dt')
    plt.ylabel('Error')
    plt.title('Error Scaling with Timestep Size')
    plt.grid(True)
    plt.legend()
    plt.show()
    print(average_errors)

    return dts, average_errors, max_errors


def main():
    # # #run drivers
    # T = 50
    # dt = 0.01
    # x0, y0 = 0, 0
    # vx, vy = 1, 1
    # ax_amp = 3 # adding a small constant acceleration in x
    # ay_amp= 0
    #
    # # # target with acceleration
    # t1 = Target(x0, y0, vx, vy, ax_amp, ay_amp)
    # # Run RK4 simulation
    # sim = Simulation(dt, T, 'rk4', t1)
    # positions, velocities = sim.run_simulation()
    #
    # # Plot the trajectory
    # plt.plot(positions[:, 0], positions[:, 1], 'b.-', label='RK4 Trajectory')
    # plt.xlabel('x position')
    # plt.ylabel('y position')
    # plt.legend()
    # plt.title('Trajectory with Oscillatory Acceleration')
    # plt.grid()
    # plt.show()
    #
    # plt.plot(np.linspace(0, T, len(t1.velocities)), t1.velocities[:, 0], label='Velocity X')
    # plt.plot(np.linspace(0, T, len(t1.velocities)), t1.velocities[:, 1], label='Velocity Y')
    # plt.xlabel('Time')
    # plt.ylabel('Velocity')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # energies = sim.energies
    # # print(f' running sim {sim.method} for its positions: \n {positions}')

    ## error check for euler function; could be expanded later to/for rk4
    error_check()




if __name__ == '__main__':
    main()
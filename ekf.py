from target_simulation import Target, Simulation
from radar import Radar
import numpy as np
import matplotlib.pyplot as plt

class Kalman():
    def __init__(self, target, sim):
        self.dt = sim.dt
        self.target = target
        self.x = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4) * 1000
        self.time = 0.0

    def state_transition(self, state):
        x, y, vx, vy = state.flatten()
        dt = float(self.dt)
        acceleration = self.target.get_acceleration(self.time, np.array([vx, vy]))

        x_new = x + vx * dt
        y_new = y + vy * dt
        vx_new = vx + acceleration[0] * dt
        vy_new = vy + acceleration[1] * dt

        new_state = np.array([[x_new], [y_new], [vx_new], [vy_new]], dtype=float)
        return new_state

    def jacobian(self, state):
        drag = float(self.target.drag)
        turn_rate = float(self.target.turn_rate)
        dt = float(self.dt)

        F_jacobian = np.array([
            [1, 0, dt, 0], 
            [0, 1, 0, dt], 
            [0, 0, 1 - drag * dt, -turn_rate * dt], 
            [0, 0, turn_rate * dt, 1 - drag * dt]
        ], dtype=float)

        return F_jacobian

    def predict(self):
        self.x = self.state_transition(self.x)
        self.time += self.dt
        F_jacobian = self.jacobian(self.x)
        Q = np.array([
            [0.001, 0, 0, 0],  
            [0, 0.001, 0, 0],  
            [0, 0, 0.01, 0],   
            [0, 0, 0, 0.01]    
        ])
        self.P = F_jacobian @ self.P @ F_jacobian.T + Q

    def measurement_update(self, z_cartesian):
        H = np.array([
            [1, 0, 0, 0],  
            [0, 1, 0, 0]
        ], dtype=float)

        R = np.array([
            [3**2, 0], 
            [0, 3**2]
        ], dtype=float)

        y = z_cartesian - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

def mc_kalman(n_runs):
    all_estimated_trajectories = []
    total_rmse = 0

    for _ in range(n_runs):
        kf = Kalman(target, sim)
        kf.x = np.array([[target_trajectory[0][0]], 
                         [target_trajectory[0][1]], 
                         [3],  
                         [3]], dtype=float)

        estimated_positions = []

        for z in cartesian_measurements:
            kf.predict()
            z_measurement = np.array([[z[0]], [z[1]]])
            kf.measurement_update(z_measurement)
            estimated_positions.append((kf.x[0, 0], kf.x[1, 0]))

        all_estimated_trajectories.append(estimated_positions)
        rmse = np.sqrt(np.mean((np.array(estimated_positions) - np.array(true_positions))**2))
        total_rmse += rmse

    avg_rmse = total_rmse / n_runs
    print(f'Average RMSE over {n_runs} runs: {avg_rmse:.6f}')

    all_estimated_trajectories = np.array(all_estimated_trajectories)  # Shape: (n_runs, n_measurements, 2)
    mean_estimated_trajectory = np.mean(all_estimated_trajectories, axis=0)

    return mean_estimated_trajectory, avg_rmse

T = 200
dt = 0.01
x0, y0 = 0, 0
vx, vy = 5, 5
ax_amp, ay_amp = 5, 5
drag = 0.1
turn_rate = 0.3

target = Target(x0, y0, vx, vy, ax_amp, ay_amp, drag, turn_rate)
sim = Simulation(dt, T, method='rk4', target=target)

radar = Radar(position=(0, 0), refresh_rate=0.01, noise=0.5)
target_trajectory, _ = sim.run_simulation()

cartesian_measurements, true_positions = radar.track_target(target_trajectory, sim)

kf = Kalman(target, sim)
kf.x = np.array([[target_trajectory[0][0]], 
                 [target_trajectory[0][1]], 
                 [3],  
                 [3]], dtype=float)


for n in [10, 100, 200, 400]:

    mean_estimated_trajectory, avg_rmse = mc_kalman(n)

    # Calculate normalized RMSE for better scale comparison
    max_pos = np.max(np.linalg.norm(true_positions, axis=1))
    normalized_rmse = np.sqrt(np.mean((np.array(true_positions) - np.array(mean_estimated_trajectory))**2)) / max_pos
    print(f"Normalized RMSE: {normalized_rmse}")

    true_x, true_y = zip(*true_positions)
    mean_estimated_x, mean_estimated_y = zip(*mean_estimated_trajectory)

    plt.figure(figsize=(8, 6))
    plt.plot(true_x, true_y, 'g-', label='True Trajectory', linewidth=2)
    plt.plot(mean_estimated_x, mean_estimated_y, 'bo-', label='Average Estimated Trajectory', linewidth=2)

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Kalman Filter Monte Carlo Simulation (Avg RMSE: {avg_rmse:.4f})')
    plt.legend()
    plt.grid()
    plt.show()






# T = 100
# dt = 0.01
# x0, y0 = 0, 0
# vx, vy = 5, 5
# ax_amp, ay_amp = 0, 0
# drag = 0.2
# turn_rate = 1.0









# target = Target(x0, y0, vx, vy, ax_amp, ay_amp, drag, turn_rate)
# sim = Simulation(dt, T, method='rk4', target=target)

# radar = Radar(position=(0, 0), refresh_rate=0.01, noise=0.2)
# target_trajectory, _ = sim.run_simulation()

# cartesian_measurements, true_positions = radar.track_target(target_trajectory, sim)

# kf = Kalman(target, sim)
# kf.x = np.array([[target_trajectory[0][0]], 
#                  [target_trajectory[0][1]], 
#                  [3],  
#                  [3]], dtype=float)

# estimated_positions = []
# measured_positions = []

# for z in cartesian_measurements:
#     kf.predict()
#     estimated_positions.append((kf.x[0, 0], kf.x[1, 0]))
#     z_measurement = np.array([[z[0]], [z[1]]])
#     kf.measurement_update(z_measurement)
#     measured_positions.append((z[0], z[1]))
#     estimated_positions.append((kf.x[0, 0], kf.x[1, 0]))

# true_x, true_y = zip(*true_positions)
# measured_x, measured_y = zip(*measured_positions)
# estimated_x, estimated_y = zip(*estimated_positions)

# plt.figure(figsize=(8, 6))
# plt.plot(true_x, true_y, 'g-', label='True Trajectory', linewidth=2)
# plt.plot(measured_x, measured_y, 'rx', label='Measured (Noisy)', markersize=8)
# plt.plot(estimated_x, estimated_y, 'bo-', label='Estimated Trajectory', linewidth=2)

# plt.xlabel('X Position')
# plt.ylabel('Y Position')
# plt.title('Kalman Filter Tracking with Radar Data')
# plt.legend()
# plt.grid()
# plt.show()





#####################3
    # def state_transition(self, state):
    #     x, y, vx, vy = state.flatten()
    #     drag = float(self.target.drag)
    #     turn_rate = float(self.target.turn_rate)
    #     dt = float(self.dt)
    #     self.target.ax_amp
    #     ax_amp = float(self.target.ax_amp)
    #     ay_amp = float(self.target.ay_amp)
    #     # Compute thrust components
    #     thrust_x = ax_amp * np.cos(turn_rate * time)
    #     thrust_y = ay_amp * np.sin(turn_rate * time)

    #     x_new = x + vx * dt
    #     y_new = y + vy * dt
    #     vx_new = vx + (-drag * vx - turn_rate * vy) * dt
    #     vy_new = vy + (-drag * vy + turn_rate * vx) * dt

    #     new_state = np.array([[x_new], [y_new], [vx_new], [vy_new]], dtype=float)
    #     return new_state



"""decent params:
    
    1) T = 100
dt = 0.01
x0, y0 = 0, 0
vx, vy = 5, 5
ax_amp, ay_amp = 4, 4
drag = 0.0
turn_rate = 0.1

2)...
"""
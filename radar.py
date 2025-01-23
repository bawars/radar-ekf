"""here we will 'receive' input from our moving target and get its position. we add noise and will feed it to a kalman filter to estimate current and/or future pos"""

import numpy as np
import matplotlib.pyplot as plt
from target_simulation import Target, Simulation


class Radar:
    def __init__(self, position, refresh_rate, noise):
        self.position = np.array(position)
        self.refresh_rate = refresh_rate
        self.noise = noise
        self.measurements = []  # noisy measurements here
        self.curr_time = 0.0  # for measuring elapsed time

    def __repr__(self):
        return (f"Radar(position={self.position}, refresh_rate={self.refresh_rate}, "
                f"noise={self.noise})")

    def measure_distance(self, target_position):
        distance = np.linalg.norm(target_position - self.position)  
        return distance
    


    def measure_angle(self, target_position):
        x_r, y_r = self.position  # unpacking radar position
        
        x, y = target_position  # unpacking target position
        angle = np.arctan2(y - y_r, x - x_r)
        return angle

    def simulate_measurement(self, target_position):
        """
        Simulates a radar measurement by adding Gaussian noise to the calculated distance and angle.
        """
        true_distance = self.measure_distance(target_position)
        true_angle = self.measure_angle(target_position)

        # Adding random noise to the metrics
        noisy_distance = true_distance + np.random.normal(0, self.noise)
        noisy_angle = true_angle + np.random.normal(0, self.noise)

        return noisy_distance, noisy_angle

    def track_target(self, target_trajectory, sim):
        self.measurements = []
        self.curr_time = 0.0
        
        dt = sim.dt
        steps_per_measurement = int(self.refresh_rate / dt)
        
        cartesian_measurements = []  # Store converted Cartesian measurements

        for i in range(0, len(target_trajectory), steps_per_measurement):
            pos = target_trajectory[i]
            #print(f"Processing position: {pos}")
            noisy_distance, noisy_angle = self.simulate_measurement(pos)

            # Convert polar to Cartesian coordinates
            x_meas = self.position[0] + noisy_distance * np.cos(noisy_angle)
            y_meas = self.position[1] + noisy_distance * np.sin(noisy_angle)
            
            cartesian_measurements.append([x_meas, y_meas])

            self.measurements.append({
                "time": self.curr_time,
                "true_pos": pos,
                "noisy_distance": noisy_distance,
                "noisy_angle": noisy_angle,
                "x_meas": x_meas,
                "y_meas": y_meas
            })

            self.curr_time += self.refresh_rate

        return cartesian_measurements, target_trajectory




#############################################
#DRIVERs
T = 10
dt = 0.01
x0, y0 = 0, 0
vx, vy = 3, 3
ax_amp = 4
ay_amp = 4
drag = 0.2
turn_rate = 1


t1 = Target(x0, y0, vx,vy,ax_amp,ay_amp, drag, turn_rate)
sim_euler = Simulation(dt, T, 'euler', t1)
positions_euler, energies_euler = sim_euler.run_simulation()
target_trajectory = positions_euler




#####################################
radar = Radar(position=(0,0), refresh_rate=0.1, noise=0.3)

noisy_data, true_positions = radar.track_target(target_trajectory, sim_euler)




#print(noisy_data)
#print(true_positions)

# # loop to access stored measurements by key
# for idx, measurement in enumerate(noisy_data):
#     dist = measurement["noisy_distance"]
#     angle = measurement["noisy_angle"]
#     true_pos = measurement["true_pos"]
#     # print(f"Step {idx + 1}: Distance = {dist:.2f}, Angle = {np.degrees(angle):.2f} degrees")


# # here we extract true and noisy positions for plotting
# true_x, true_y = zip(*target_trajectory)
# measured_x = [radar.position[0] + m["noisy_distance"] * np.cos(m["noisy_angle"]) for m in noisy_data]
# measured_y = [radar.position[1] + m["noisy_distance"] * np.sin(m["noisy_angle"]) for m in noisy_data]

#print(measured_x[-1], measured_y[-1])

# print(f"Expected measurements: {T/radar.refresh_rate}")
# print(f"Actual measurements: {len(noisy_data)}")

# plt.plot(true_x, true_y, label="True Trajectory", marker="o")
# plt.plot(measured_x, measured_y, label="Noisy Measurements", linestyle="--", marker="x")
# plt.legend()
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.title(f"Radar Tracking Simulation with Refresh Rate: {radar.refresh_rate}")
# plt.grid()
# plt.show()




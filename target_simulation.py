import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

"""we will construct a moving target in space (and time), with a radar placed at the origin
the radar will be fed signals from the moving target - with some noise - and will through state space modelling 
seek to predict the target's next position"""




class Target():
    
    def __init__(self, x0, y0, vx, vy, ax_amp, ay_amp, drag, turn_rate):
        self.x0 = x0
        self.y0 = y0
        self.vx = vx
        self.vy = vy
        self.ax_amp = ax_amp
        self.ay_amp = ay_amp
        self.drag = drag
        self.turn_rate = turn_rate

        self.position = np.array([x0, y0], dtype=float) 
        self.positions = None
        self.velocity = np.array([vx, vy], dtype=float) 
        self.velocities = None  # unused for now; for when velocity is non constant


    # def get_acceleration(self, time, velocity): 
    #     drag_coefficient = 0.2
    #     velocity_magnitude = np.linalg.norm(velocity)
    #     drag_force = -drag_coefficient * velocity_magnitude * velocity
    #     return np.array([
    #         drag_force[0] + self.ax_amp * np.sin(0.5*time),
    #         drag_force[1] + self.ay_amp * np.cos(0.5*time)
    #     ])
    # def get_acceleration(self, time, velocity):
    #     linear_drag_coefficient = 0.2
    #     linear_drag_force = -linear_drag_coefficient * velocity  # linear
    #     return np.array([
    #         linear_drag_force[0] + self.ax_amp * np.sin(time),
    #         linear_drag_force[1] + self.ay_amp * np.cos(time)
    #     ])

    # def get_acceleration(self, time, velocity): 
    #     # Circular motion component
    #     acc_x = -self.turn_rate * velocity[1]
    #     acc_y = self.turn_rate * velocity[0]

    #     # constant acceleration components in fixed directions
    #     ax_external = self.ax_amp
    #     ay_external = self.ay_amp

    #     # Final acceleration with drag
    #     return np.array([acc_x + ax_external, acc_y + ay_external]) - self.drag * velocity
    
    def get_acceleration(self, time, velocity): 
        # control input (thrust)
        thrust_x = self.ax_amp * np.cos(self.turn_rate * time)
        thrust_y = self.ay_amp * np.sin(self.turn_rate * time)

        # dircular motion component
        acc_x = -self.turn_rate * velocity[1]
        acc_y = self.turn_rate * velocity[0]

        # drag force opposing velocity
        drag_force = self.drag * velocity

        # final acceleration combining control, circular motion, and drag
        return np.array([thrust_x + acc_x, thrust_y + acc_y]) - drag_force




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
    

    def animate_trajectory(self, positions_euler, positions_rk4, save_path = None):
        fig, ax = plt.subplots(figsize=(8, 6))
        line_euler, = ax.plot([], [], 'b--', label='Euler', alpha=0.7, linewidth=2)  # Blue dashed line
        line_rk4, = ax.plot([], [], 'r:', label='RK4', alpha=0.8, linewidth=2)      # Red dotted line
        
        ax.set_xlim(-50, 150)
        ax.set_ylim(-30, 70)
        ax.grid(True)
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.legend()
        
        def update(frame):
            line_euler.set_data(positions_euler[:frame, 0], positions_euler[:frame, 1])
            line_rk4.set_data(positions_rk4[:frame, 0], positions_rk4[:frame, 1])
            
            # Dynamic limits
            if frame > 1:
                x_min = min(positions_euler[:frame,0].min(), positions_rk4[:frame,0].min())
                x_max = max(positions_euler[:frame,0].max(), positions_rk4[:frame,0].max())
                y_min = min(positions_euler[:frame,1].min(), positions_rk4[:frame,1].min())
                y_max = max(positions_euler[:frame,1].max(), positions_rk4[:frame,1].max())
                
                x_pad = 0.1 * (x_max - x_min)
                y_pad = 0.1 * (y_max - y_min)
                
                ax.set_xlim(x_min - x_pad, x_max + x_pad)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)
            
            return line_euler, line_rk4
        
        anim = FuncAnimation(fig, update, frames=len(positions_euler), 
                            interval=1e-10, blit=True, repeat=False)
        


        if save_path:
            try:
                print(f"Attempting to save animation to {save_path}...")
                if save_path.endswith('.mp4'):
                    writer = FFMpegWriter(fps=60)
                elif save_path.endswith('.gif'):
                    writer = PillowWriter(fps=60)
                else:
                    raise ValueError("Unsupported file format. Use .mp4 or .gif")
                
                anim.save(save_path, writer=writer)
                print(f"Animation successfully saved to {save_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
        else:
            plt.show()





def error_convergence():
    T = 5
    dts = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    x0, y0, vx, vy = 0, 0, 3, 3
    ax_amp = ay_amp = 10
    drag = 0.2
    turn_rate = 1

    dt_ref = 1e-3
    target_ref = Target(x0, y0, vx, vy, ax_amp, ay_amp, drag, turn_rate)
    sim_ref = Simulation(dt_ref, T, 'rk4', target_ref)
    pos_ref = sim_ref.run_simulation()[0]

    errors_euler = []
    errors_rk4 = []

    for dt in dts:
        # Euler
        target_euler = Target(x0, y0, vx, vy, ax_amp, ay_amp, drag, turn_rate)
        sim_euler = Simulation(dt, T, 'euler', target_euler)
        pos_euler = sim_euler.run_simulation()[0]
        error_euler = np.sqrt(np.sum((pos_ref[-1] - pos_euler[-1])**2))
        errors_euler.append(error_euler)

        # RK4
        target_rk4 = Target(x0, y0, vx, vy, ax_amp, ay_amp, drag, turn_rate)
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
    vx, vy = 5, 10
    ax_amp, ay_amp = 0, 0
    drag = 0.01
    turn_rate = 0.0



    # --- Euler ---
    target_euler = Target(x0, y0, vx, vy, ax_amp, ay_amp,drag, turn_rate)
    sim_euler = Simulation(dt, T, 'euler', target_euler)
    positions_euler, energies_euler = sim_euler.run_simulation()

    # --- RK4 ---
    target_rk4 = Target(x0, y0, vx, vy, ax_amp, ay_amp,drag, turn_rate)
    sim_rk4 = Simulation(dt, T, 'rk4', target_rk4)
    positions_rk4, energies_rk4 = sim_rk4.run_simulation()

    plt.figure(figsize=(8, 6))
    plt.plot(positions_euler[:, 0], positions_euler[:, 1], label='Euler')
    plt.plot(positions_rk4[:, 0], positions_rk4[:, 1], label='RK4')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.legend()
    plt.title('Comparison of Euler vs. RK4')
    plt.grid(True)
    plt.show()

    #print(positions_rk4[:5])
    #velocities_euler = target_euler.velocities
    #print(velocities_euler)
    # plt.figure(figsize=(8, 6))
    # plt.plot(velocities_euler, label='Euler')
    # plt.xlabel('x vel')
    # plt.ylabel('y vel')
    # plt.legend()
    # plt.title('velocities')
    # plt.grid(True)
    # plt.show()

    #sim_euler.animate_trajectory(positions_euler, positions_rk4)
    #sim_euler.animate_trajectory(positions_euler, positions_rk4, save_path="/home/johndoe/trajectory.gif")





    error_convergence()



if __name__ == '__main__':
    main()


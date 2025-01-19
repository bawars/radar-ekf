import numpy as np
import matplotlib.pyplot as plt

"""we will construct a moving target in space (and time), with a radar placed at the origin
the radar will be fed signals from the moving target - with some noise - and will through state space modelling 
seek to predict the target's next position"""


# we define our radar position. a given coordinate is of the form (x,y)

#radar_position = np.array([0,0])


class Target():

    def __init__(self, x0,y0, vx, vy, ax, ay=0):
        self.x0 = x0
        self.y0 = y0
        self.vx = vx
        self.vy = vy

        self.position = np.array([x0,y0], dtype = float) #current position
        self.positions = None
        #self.velocities = np.array([[vx, vy]])  # unused for now; for when velocity is non constant
        self.velocity = np.array([vx,vy], dtype = float)    # constant velocity
        self.acceleration = np.array([ax, 0], dtype = float)
        self.step = 1 # indexing

    def setup(self, N):
        self.positions = np.zeros((N,2))
        self.positions[0] = self.position
        self.step = 1



    def euler_update(self, dt): # we start with a simple euler forward algorithm
        
        self.velocity += self.acceleration*dt   # first update velocity, then position below

        self.position += self.velocity*dt
        self.positions[self.step] = self.position
        self.step += 1
        

    def rk4_update(self, dt):
        pass

    def state_update(self, dt, method):
        if method == 'euler':
            self.euler_update(dt)
        elif method == 'rk4':
            self.rk4_update(dt)

        else:
            return None
        
        


class Simulation():
    
    def __init__(self, dt, T, method, target):
        self.dt=dt
        self.T = T
        self.N = int(T/dt)
        self.method = method
        self.target = target
        
        
        self.target.setup(self.N)       # we initialize posiitons array of dimension N here


    def run_simulation(self):

        times = np.linspace(0, self.T, self.N)

        for _ in range(self.N-1):
            self.target.state_update(self.dt, self.method)
        return self.target.positions





        
def error_check():
    trajectories = {}


    analytic_solutions = {}
    
    errors = []

    x0, y0 = 0, 0
    vx, vy = 1, 1
    ax = 0.5
    T = 20
    dt_values = [0.2, 0.1, 0.05, 0.025]


        # Lists to store dt and corresponding errors for plotting
    dts = []
    average_errors = []
    max_errors = []
    #dt_values = [0.1]

    for dt in dt_values:
        N = int(T/dt)
        #print(N)
        # euler int
        tar = Target(x0,y0,vx,vy, ax)
        sim = Simulation(dt, T, method='euler', target=tar)
        sim.run_simulation()

        trajectories[dt] = tar.positions.tolist()



        times = np.linspace(0, T, N+1)
        analytic_positions = []
        for t in times:
            # x(t) = x0 + v0x*t + (1/2)*ax*t^2
            x = x0 + vx*t + 0.5*ax*t**2
            # y(t) = y0 + vy*t (no acceleration in y)
            y = y0 + vy*t
            analytic_positions.append([x,y])
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

    # Create log-log plot
    plt.figure(figsize=(10,6))
    plt.loglog(dts, average_errors, 'bo-', label='Average Error')
    plt.loglog(dts, max_errors, 'ro-', label='Maximum Error')
    
    # Add reference line (slope = 1)
    reference_x = np.array([min(dts), max(dts)])
    reference_y = reference_x * average_errors[-1]/dts[-1]  # Scale to match our data
    plt.loglog(reference_x, reference_y, 'k--', label='Reference (slope = 1)')
    
    plt.xlabel('dt')
    plt.ylabel('Error')
    plt.title('Error Scaling with Timestep Size')
    plt.grid(True)
    plt.legend()
    plt.show()

    return dts, average_errors, max_errors


def main():
    # run properties
    T = 10
    dt = 0.1
    x0, y0 = 0, 0
    vx, vy = 1, 1
    ax = 0.5  # adding a small constant acceleration in x

    # target with acceleration
    t1 = Target(x0, y0, vx, vy, ax)
    sim = Simulation(dt, T, 'euler', t1)
    positions = sim.run_simulation()

    # positions = np.array(positions)
    # plt.figure(figsize=(10,6))
    # plt.plot(positions[:,0], positions[:,1], 'b.-', label='Trajectory')
    # plt.xlabel('x position')
    # plt.ylabel('y position')
    # plt.title('Target Motion with Constant x-acceleration')
    # plt.grid(True)
    # plt.legend()
    # plt.show()


    # print("\nFirst few positions:")
    # print(positions[:5])
    # print("\nLast few positions:")
    # print(positions[-5:])
    
    error_check()
    
    #sim.run_simulation()

#    error_check(sim,dt)

if __name__ == '__main__':
    main()
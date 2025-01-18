import numpy as np
import matplotlib.pyplot as plt

"""we will construct a moving target in space (and time), with a radar placed at the origin
the radar will be fed signals from the moving target - with some noise - and will through state space modelling 
seek to predict the target's next position"""


# we define our radar position. a given coordinate is of the form (x,y)

radar_position = np.array([0,0])


class Target():

    def __init__(self, x,y, vx, vy, N):
        self.position = np.array([x,y])
        self.positions = np.zeros((N,2))    # aggregated positions
        self.positions[0] = [x,y]
        #self.velocities = np.array([[vx, vy]])  # we will start with constant velocity for now
        self.velocity = np.array([vx,vy])
        self.step = 1 # indexing



    def euler_update(self, dt):
        # we need to update component-wise wrt x and y
        
        x_new = self.position[0] + self.velocity[0]*dt
        y_new = self.position[1] + self.velocity[1]*dt
        
        self.position =np.array([x_new, y_new])
        self.positions[self.step] = self.position
        self.step += 1
        
        #self.positions = np.vstack((self.positions, self.position))
        #print(self.positions)


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
    
    def __init__(self, dt, T, method):
        self.dt=dt
        self.T = T
        self.N = int(T/dt)
        self.method = method


    def run_simulation(self, target):

        errors = []
        times = np.linspace(0, self.T, self.N)

        for step in range(self.N - 1):
            target.state_update(self.dt, self.method)

            # analytical solution at this time is


        return target.positions



        



# def run_simulation(target, dt, T, method):
#     N = T/dt
#     #print(N)
#     for step in range(int(N)):
#         target.state_update(dt, method)

#     return target.positions







def error_check(sim, dt, T, x0, y0, vx, vy):
    ### we need tocomapre to analytical solution for euler method


        # analytivally we have x1 = x0 + vx*t
        x_an  = x0 + vx*T
        y_an = y0 + vy*T
        r_anal= np.array([x_an,y_an])
        #print(x_analytical, y_analytical)

        #positions = run_simulation(target, dt, T, 'euler')
        target = Target(x0,y0,vx,vy, sim.N)
        positions = sim.run_simulation(target)

        x_num, y_num = positions[-1]
        #print(f'numerical: {x_num, y_num}')
        #print(f'analytical: {x_an, y_an}')

        error = np.sqrt((x_num-x_an)**2 + (y_num - y_an)**2)
        print(f"Error at T={T}: {error}")
        return error
    #error = np.sqrt((x_num - r_anal[0])**2 +(r)





def main():
    # run properties
    T = 100
    dt = 0.001

    x0, y0 = 0, 0
    vx, vy = 1, 1


    #print(run_simulation(t1,dt,'euler',T))
    #t1 = Target(x0,y0,vx,vy)
    #sim.run_simulation(t1)

    time_points = [10, 25, 50, 100]
    for current_T in time_points:
        sim = Simulation(dt,current_T,'euler')
        error_check(sim, dt, current_T, x0, y0, vx, vy)


if __name__ == '__main__':
    main()
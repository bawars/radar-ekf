from target_simulation import Target, Simulation
import numpy as np

class Kalman():
    def __init__(self, target, sim):
        self.dt = sim.dt
        self.target = target
        self.x = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4) * 1000

    def state_transition(self, state):
        x, y, vx, vy = state.flatten()
        drag = float(self.target.drag)
        turn_rate = float(self.target.turn_rate)
        dt = float(self.dt)

        x_new = x + vx * dt
        y_new = y + vy * dt
        vx_new = vx + (-drag * vx - turn_rate * vy) * dt
        vy_new = vy + (-drag * vy + turn_rate * vx) * dt

        new_state = np.array([[x_new], [y_new], [vx_new], [vy_new]], dtype=float)
        return new_state

    def jacobian(self, state):
        drag = float(self.target.drag)
        turn_rate = float(self.target.turn_rate)
        dt = float(self.dt)

        F_jacobian = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1 - dt * drag, -dt * turn_rate],
            [0, 0, dt * turn_rate, 1 - dt * drag]
        ], dtype=float)

        return F_jacobian

    def predict(self):
        self.x = self.state_transition(self.x)
        F_jacobian = self.jacobian(self.x)
        noise = 1e-1
        Q = noise * np.eye(4)
        self.P = F_jacobian @ self.P @ F_jacobian.T + Q

target = Target(x0=0, y0=0, vx=3, vy=3, ax_amp=0, ay_amp=0, drag=0.2, turn_rate=1)
sim = Simulation(dt=0.1, T=10, method='euler', target=target)
kf = Kalman(target, sim)

kf.x = np.array([[0], [0], [3], [3]], dtype=float)

predicted_state = kf.state_transition(kf.x)
print("Predicted state:")
print(predicted_state)

F_jacobian = kf.jacobian(kf.x)
print("Jacobian matrix:")
print(F_jacobian)

kf.predict()
print("Predicted state after predict step:")
print(kf.x)
print("Predicted covariance after predict step:")
print(kf.P)

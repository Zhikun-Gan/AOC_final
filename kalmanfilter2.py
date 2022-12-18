import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class KF:
    def __init__(self, timesteps):
        dt = timesteps
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([0, 0., 0, 0.])
        self.kf.F = np.array([[1., dt, 0, 0],
                    [0., 1., 0, 0],
                    [0, 0, 1., dt],
                    [0., 0, 0, 1.]])
        self.kf.H = np.array([[1, 0., 0, 0],
                            [0, 0, 1, 0]])
        self.kf.P = np.eye(4)*1
        self.kf.R = np.array([[0.005, 0],
                    [0, 0.005]])
        self.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.000001, block_size=2)

    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.update(measured)
        self.kf.predict()
        x = self.kf.x[0]
        y = self.kf.x[2]
        result = np.array([x, y])
        return result

if __name__ == '__main__':

    def os_mearsurement(start, end, num, sigma=0.1, type_='line'):
        if type_ == 'line':
            os_p_move = np.linspace(start, end, num)
        for i in range(num):
            os_p_move[i, :] += np.random.randn(2) * sigma
        return os_p_move

    os_move = os_mearsurement([-2, -5],[-1, 6],1000,sigma=0.001)
    f = KF(0.2)
    print('\n')
    for pt in os_move:
        z = pt
        print('measured states', z, '\n')
        pre = f.predict(z[0],z[1])
        print('predicted states: ', pre)
        # print('post_p: ', f.P)
    print('end')

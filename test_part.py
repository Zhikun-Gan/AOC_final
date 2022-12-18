import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

a = np.linspace([2, 3], [2, 3], 10)
b = np.repeat(np.array([[5,6]]),10,axis=0)
c = np.concatenate((a,b),axis=0)
d = np.size(c,axis=0)
print("end")
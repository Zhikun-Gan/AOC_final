import numpy as np
import matplotlib.pyplot as plt
from kalmanfilter import KalmanFilter


kf = KalmanFilter()

os_p_move = np.stack((np.linspace([0, 0], [10, -20], 40),np.linspace([2, 5], [-1, -2], 40)), axis=0)
os_r_move = [0.5, 0.5]
os_predict_1 = np.zeros((2, 40))
for k in range(40):
    predicted = kf.predict(os_p_move[0, k, 0],os_p_move[0, k, 1])
    os_predict_1[:, k] = np.concatenate((predicted[0], predicted[1]), axis=0)
    # os_predict_1[1, k] = predicted[1]

fig, axs = plt.subplots()
# plt.axis([-10, 10, -10, 10])
axs.axis('equal')
axs.plot(os_p_move[0,:,0], os_p_move[0,:,1],'m')
axs.plot(os_predict_1[0,:], os_predict_1[1,:],'--g')



plt.show()
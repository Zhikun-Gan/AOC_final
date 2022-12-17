#https://pysource.com/2021/10/29/kalman-filter-predict-the-trajectory-of-an-object/
import cv2
import numpy as np


class KalmanFilter:
    # def __init__(self,timesteps):
    # kf = cv2.KalmanFilter(4, 2)
    # kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    # kf.transitionMatrix = np.array([[1, 0, 4, 0], [0, 1, 0, 4], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    def __init__(self,timesteps):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, timesteps, 0], [0, 1, 0, timesteps], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y =predicted[0], predicted[1]
        result=np.array([x,y])
        return result[:,0]


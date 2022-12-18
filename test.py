import math
import numpy as np
import matplotlib.pyplot as plt
from DDP_moving_obstacles import DDP
# from kalmanfilter import KalmanFilter
from kalmanfilter2 import KF
import time
T1 = time.time()
DDP = DDP()
T2 = 0
def observe(kfs,measures,is_new,radius):
    predicts={}
    for i in kfs.keys():
        if(is_new[i]!=True):
            predicts[i]=kfs[i].predict(measures[i][0],measures[i][1])
            radius[i]=is_new[i]
        else:
            # kfs[i].kf.x = np.array([measures[i][0],0,measures[i][1],0])
            predicts[i]=measures[i]
            is_new[i]=radius[i]
            radius[i]=radius[i]
    return predicts,is_new,radius

def update_constraints(k,os_p_move,os_r_move,change):
    measures={}
    is_new={}
    radius={}
    if(k==0):
        for i in range(np.shape(os_p_move)[0]-1):
            is_new[i]=True
            radius[i]=os_r_move[i]
            measures[i]=os_p_move[i,k,:]
    # elif(k==change):
    #     is_new[2] = True
    #     radius[2] = os_r_move[2]
    #     measures[2] = os_p_move[2, k, :]
    return measures,is_new,radius

def expandArray(original, length):
    newArray=np.zeros((np.shape(original)[0],np.shape(original)[1]*length,np.shape(original)[2]))
    for i in range(np.shape(original)[1]-1):
        newArray[:,i*length:i*length+length,:]=np.linspace(original[:,i,:],original[:,i+1,:],length).transpose(1,0,2)
    i=np.shape(original)[1]-1
    dist=(original[:,i,:]-original[:,i-1,:])/length
    newArray[:,i*length:,:]=np.linspace(original[:,i,:],original[:,i,:]+dist*length,length).transpose(1,0,2)
    return newArray
def os_mearsurement(start, end, num, sigma=0.1, type_='line'):
    if type_ == 'line':
        os_p_move=np.linspace(start,end,num)
    for i in range(num):
        os_p_move[i,:] += np.random.randn(2)*sigma

    return os_p_move
def fun(TF,update_rate,control_rate):  # update_rate is the rate of measurment like 5Hz, 10Hz, 20Hz

    # n = DDP.N_seg
    n=int(control_rate/update_rate)
    N_all = int(TF*control_rate)
    k_range=int(TF*update_rate)
    num_measurment = int(TF*update_rate)
    # generate random measurements of moving obstacle
    os_p_move=np.zeros((3,num_measurment,2))
    os_p_move[0,:,:] = os_mearsurement([-2, -5],[-1, 6],num_measurment,sigma=0.001)
    os_p_move[1,:,:] = os_mearsurement([2, 10], [-2, -5], num_measurment, sigma=0.001)
    os_p_move[2, math.ceil(num_measurment/2):, :] = os_mearsurement([3, -6], [-4, 7], num_measurment-math.ceil(num_measurment/2),sigma= 0.001)
    os_r_move = [0.25, 2, 2]
    # predicted positions of moving obstacles
    os_p_predicted = np.zeros_like(os_p_move)
    # static obstacles
    os_p = [[-2.5, 2], [-1, 0]]
    os_r = [0.25, 0]
    # initial state
    x0 = np.array([-10, -5, 1.2, 0, 0])
    xs = np.zeros((np.size(x0), N_all+1))
    xs[:, 0] = x0
    # dicts of obstacles measurement
    kfs={}
    measures = {}
    is_new = {}
    radius = {}
    for i in range(np.shape(os_p_move)[0] - 1):
        is_new[i] = True
        radius[i] = os_r_move[i]
        measures[i] = os_p_move[i, 0, :]
    for k in range(0, k_range):
        t = TF - k / update_rate

        # measures,is_new,radius=update_constraints(k,os_p_move,os_r_move,math.ceil(k_range/2))
        for i in is_new.keys():
            if is_new[i]==True:
                # kfs[i]=KalmanFilter(control_rate/update_rate)
                kfs[i] = KF(control_rate / update_rate)

        for i in range(2):
            measures[i]=os_p_move[i,k,:]

        predicts,is_new,radius=observe(kfs, measures, is_new, radius)
        steps_left=round(t*control_rate)
        os_p_m=np.zeros((len(predicts),steps_left,2))
        os_r_m = np.zeros(len(predicts))

        for i in range(len(predicts)):
            keys = list(measures.keys())
            os_p_predicted[i,k,:]=predicts[keys[i]]
            first_seg = np.linspace(measures[keys[i]],predicts[keys[i]],int(control_rate/update_rate))
            remained_seg = np.repeat(np.array([predicts[keys[i]]]),steps_left-int(control_rate/update_rate), axis=0)
            os_p_m[i,:,:]= np.concatenate((first_seg, remained_seg), axis=0)
            os_r_m[i]=radius[keys[i]]

        xs_, N = DDP.trajectory(t, update_rate, control_rate, x0, os_p, os_r, os_p_m, os_r_m)
        # fig, axs = plt.subplots()
        # plt.plot(xs_[0,:], xs_[1, :])
        # plt.show()
        # time.sleep(0.5)
        xs[:, k*n+1: (k+1)*n+1] = xs_[:, 1:n+1]
        if k < k_range-1:
            x0=xs_[:, n+1]

    global T2
    T2 = time.time()
    # print figures
    fig, axs = plt.subplots()
    for j in range(len(os_r)):  # plot the position of static obstacles
        circle = plt.Circle(os_p[j], os_r[j], color='r', linewidth=2,
                        fill=False)
        axs.add_patch(circle)
    i_ob = 0
    for k in range(np.size(xs,1)):  # plot  agent's trajectory and the position of moving obstacles
        # axs.axis('equal')
        plt.axis([-15, 15, -10, 15])
        plt.scatter(xs[0, k], xs[1, k])
        plt.pause(1 / update_rate)
        if k%n == 0 and i_ob < np.size(os_p_move,1):
            for j in range(len(os_r_move)-1):
                circle_true = plt.Circle(os_p_move[j,i_ob,:], os_r_move[j], color='r', linewidth=2,
                              fill=False)
                circle_predicted = plt.Circle(os_p_predicted[j,i_ob,:], os_r_move[j], color='b', linewidth=2,
                              fill=False)
                axs.add_patch(circle_true)
                axs.add_patch(circle_predicted)
            i_ob += 1

        if (k+1)%n == 0 and i_ob < np.size(os_p_move,1):
            for j in range(len(os_r_move)-1):
                axs.patches.pop()
                axs.patches.pop()

    plt.show()

fun(TF=3,update_rate=5,control_rate=30)
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
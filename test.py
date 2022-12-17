import math

import numpy as np
import matplotlib.pyplot as plt
from DDP_moving_obstacles import DDP
from kalmanfilter import KalmanFilter

DDP = DDP()


def observe(kfs,measures,is_new,radius):
    predicts={}
    for i in kfs.keys():
        if(is_new[i]!=True):
            predicts[i]=kfs[i].predict(measures[i][0],measures[i][1])
            radius[i]=is_new[i]
        else:
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

def fun(TF,update_rate,control_rate):

    # n = DDP.N_seg
    n=int(control_rate/update_rate)
    N_all = int(TF*control_rate)
    k_range=int(TF*update_rate)


    # os_p_move = np.stack((np.linspace([-1, 6], [-2, -5], N_all+n), np.linspace([2, -5], [1, 10], N_all+n)), axis=0)
    os_p_move=np.zeros((3,k_range,2))

    #generate random measurements of moving obstacle
    dist1=(np.array([-2, -5])-np.array([-1, 6]))/k_range
    dist2=(np.array([1, 10])-np.array([2, -5]))/k_range
    dist3 = (np.array([3, -6]) - np.array([-4, 7])) / math.ceil(k_range/2)

    os_p_move[0,0,:]=np.array([-2, -5])
    os_p_move[1, 0, :] = np.array([1, 10])
    os_p_move[2, math.ceil(k_range/2), :] = np.array([3, -6])

    for i in range(1,k_range):
        os_p_move[0,i,:]=os_p_move[0,i-1,:]+dist1+np.random.randn(2)*0.1
        os_p_move[1, i, :] = os_p_move[1, i - 1, :] + dist2 + np.random.randn(2)*0.1
    for i in range(math.ceil(k_range/2)+1,k_range):
        os_p_move[2, i, :] = os_p_move[2, i - 1, :] + dist3 + np.random.randn(2) * 0.1

    # print(os_p_move[0,:,:])
    os_r_move = [0.25, 2,2]
    os_p_predicted = np.zeros_like(os_p_move)  # predicted positions of moving obstacles

    os_p = [[-2.5, 2], [-1, 0]]
    os_r = [0.25, 0]
    x0 = np.array([-10, -5, 1.2, 0, 0])  # initial state
    xs = np.zeros((np.size(x0), N_all+1))
    xs[:, 0] = x0
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
                kfs[i]=KalmanFilter(control_rate/update_rate)

        for i in range(2):
            measures[i]=os_p_move[i,k,:]

        predicts,is_new,radius=observe(kfs, measures, is_new, radius)
        steps_left=int(t*control_rate)
        os_p_m=np.zeros((len(predicts),steps_left,2))
        os_r_m = np.zeros(len(predicts))

        for i in range(len(predicts)):
            keys = list(measures.keys())
            dist=(predicts[keys[i]]-measures[keys[i]])/n
            os_p_predicted[i,k,:]=predicts[keys[i]]
            # print(np.shape(predicts[keys[i]]))
            # print(np.shape(measures[keys[i]]))
            #
            # print(np.shape(dist))
            # print(np.shape(np.linspace(measures[keys[i]],measures[keys[i]]+steps_left*dist,steps_left)))
            os_p_m[i,:,:]=np.linspace(measures[keys[i]],measures[keys[i]]+steps_left*dist,steps_left)
            os_r_m[i]=radius[keys[i]]

        xs_, N = DDP.trajectory(t, update_rate, control_rate, x0, os_p, os_r, os_p_m, os_r_m)
        xs[:, k*n+1: (k+1)*n+1] = xs_[:, 1:n+1]
        x0=xs_[:, n]

    os_p_move = np.zeros((3, N_all, 2))

    # generate random measurements of moving obstacle
    dist1 = (np.array([-2, -5]) - np.array([-1, 6])) / N_all
    dist2 = (np.array([1, 10]) - np.array([2, -5])) / N_all

    os_p_move[0, 0, :] = np.array([-2, -5])
    os_p_move[1, 0, :] = np.array([1, 10])

    for i in range(1,N_all):
        os_p_move[0,i,:]=os_p_move[0,i-1,:]+dist1+np.random.randn(2)*0.1
        os_p_move[1, i, :] = os_p_move[1, i - 1, :] + dist2 + np.random.randn(2)*0.1
    os_p_predicted=expandArray(os_p_predicted,n)

    fig, axs = plt.subplots()
    # plt.axis([-10, 10, -10, 15])

    for j in range(len(os_r)):
        circle = plt.Circle(os_p[j], os_r[j], color='r', linewidth=2,
                        fill=False)
        axs.add_patch(circle)
    size = np.size(os_p_move,1)
    for k in range(np.size(os_p_move,1)):
        axs.axis('equal')
        # plt.axis([-15, 15, -10, 15])


        plt.scatter(xs[0, k], xs[1, k])
        for j in range(len(os_r_move)):
            circle_true = plt.Circle(os_p_move[j,k,:], os_r_move[j], color='r', linewidth=2,
                          fill=False)
            circle_predicted = plt.Circle(os_p_predicted[j,k,:], os_r_move[j], color='b', linewidth=2,
                          fill=False)
            axs.add_patch(circle_true)
            axs.add_patch(circle_predicted)
        plt.pause(DDP.h)
        if k != np.size(os_p_move,1)-1:
            for j in range(len(os_r_move)):
                axs.patches.pop()
                axs.patches.pop()
    plt.show()

fun(TF=3,update_rate=5,control_rate=20)
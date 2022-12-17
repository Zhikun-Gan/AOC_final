import numpy as np
import matplotlib.pyplot as plt
from ddp_traj import ddp_traj
from ddp_cost import ddp_cost
from ddp import ddp
import math
# Optimal control example of a simple car model with disk obstacles
# Control bounds  were added

class DDP:

  def __init__(self,t=1,N=40):

    # time-step and # of segments
    self.tf = t
    self.t_seg = t
    self.N = int(N)
    self.N_seg = int(N)
    self.h = self.tf/self.N  # dt

    # system mass
    self.m = 2

    # cost function specification
    self.Q = np.diag([1, 1, 0, 0, 0])
    self.R = np.diag([0.1, 0.1])
    self.Pf = np.diag([100, 100, 0, 0, 0])

    self.mu = 1
    # set obstacles and penalty coefficient
    self.os_p = [[-2.5, 2], [-1, 0]]
    self.os_r = [1, 0]
    # self.os_p_move = []
    # self.os_p_move.append(np.linspace([0, 3], [1, -1.3], self.N))
    # self.os_p_move.append(np.linspace([2, 5], [-1, -2], self.N))
    self.os_p_move = np.stack((np.linspace([0, 3], [1, 0], self.N), np.linspace([2, 5], [-1, -2], self.N)), axis=0)
    self.os_r_move = [2, 2]
    self.ko_x = 5
    self.ko_u = 1

    # initial state
    self.x0 = np.array([-3, -1, 1.2, 0, 0])

    # control bound
    self.u_bd = np.array([50, 30])  # absolute control boundary

    # goal position
    self.goal = np.array([1, 1, 0, 0, 0])

  def f(self, k, x, u):

    h = self.h
    c = np.cos(x[2])
    s = np.sin(x[2])
    v = x[3]
    w = x[4]

    A = np.array([[1, 0, -h * s * v, h * c, 0],
                  [0, 1, h * c * v, h * s, 0],
                  [0, 0, 1, 0, h],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])

    B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [h, 0],
                  [0, h]])

    x = np.array([x[0] + h * c * v, x[1] + h * s * v, x[2] + h * w, v + h * u[0],
                  w + h * u[1]])

    return x, A, B

  def L(self, k, x, u):

    if k < self.N:
      # print(np.shape(x))
      # print(np.shape(u))
      L = self.h * 0.5 * (np.transpose(x-self.goal)@self.Q@(x-self.goal) + np.transpose(u)@self.R@u)

      Lx = self.h * self.Q @ (x-self.goal)
      Lxx = self.h * self.Q
      Lu = self.h * self.R @ u
      Luu = self.h * self.R
    else:
      L = np.transpose(x-self.goal) @ self.Pf @ (x-self.goal) * 0.5
      Lx = self.Pf @ (x-self.goal)
      Lxx = self.Pf
      Lu = np.zeros(self.m)
      Luu = np.zeros((self.m, self.m))

    if k < self.N and hasattr(self, 'u_bd'):
      for i in range(len(u)):
        c_1 = np.abs(u[i])-self.u_bd[i]
        c_u = max(c_1, 0)
        v_u = np.array([np.sign(u[i]), 0])

        L = L + self.ko_u/2*c_u**2
        Lu = Lu + self.ko_u*v_u*c_u
        Luu = Luu + self.ko_u*v_u*np.transpose(v_u)*c_u/abs(c_1)

    if hasattr(self, 'os_r'):

      for i in range(len(self.os_r)):
        g = x[:2] - self.os_p[i]
        c_2 = self.os_r[i] - np.linalg.norm(g)
        v_x = -g/np.linalg.norm(g)
        c_x = max(c_2, 0)

        L = L + self.ko_x/2.0*c_x**2
        Lx[:2] = Lx[:2] + self.ko_x*v_x*c_x
        Lxx[:2, :2] = Lxx[:2, :2] + self.ko_x*v_x@np.transpose(v_x)*c_x/np.abs(c_2)

    if k < self.N and hasattr(self, 'os_r_move'):
      for i in range(len(self.os_r_move)):
        g = x[:2] - self.os_p_move[i, k, 0:2]
        c_2 = self.os_r_move[i] - np.linalg.norm(g)
        v_x = -g / np.linalg.norm(g)
        c_x = max(c_2, 0)

        L = L + self.ko_x / 2.0 * c_x ** 2
        Lx[:2] = Lx[:2] + self.ko_x * v_x * c_x
        Lxx[:2, :2] = Lxx[:2, :2] + self.ko_x * v_x @ np.transpose(v_x) * c_x / np.abs(c_2)

    return L, Lx, Lxx, Lu, Luu

  def trajectory(self, tf, update_rate,control_rate,x0, os_p, os_r, os_p_m, os_r_m):

    # self.N = tf//self.t_seg*self.N_seg
    self.N = int(tf*control_rate)
    self.tf = tf
    self.x0 = x0
    self.os_p = os_p
    self.os_p_move = os_p_m
    self.os_r = os_r
    self.os_r_move = os_r_m
    # initial control sequence

    us = np.concatenate((np.tile([[0.1], [0.05]], (1, self.N // 2)),
                         np.tile([[-0.1], [-0.05]], (1, self.N // 2))), axis=1) / 2

    if(np.shape(us)[1]!=self.N):
      us = np.concatenate((us, np.tile([[-0.1], [-0.05]], (1, 1))/2), axis=1)
      # print(np.shape(us)[1])
      # print(self.N)
      # print(tf)
      # print(control_rate)
    for i in range(50):
      dus, V, Vn, dV, a = ddp(us, self)
      # update control
      us = us + dus
      self.a = a
      xs = ddp_traj(us, self)
      # axs[0].plot(xs[0,:], xs[1,:], '-g')
      self.ko_x = self.ko_x * 1.5
      self.ko_u = self.ko_u * 1.1
    return xs, self.N
if __name__ == '__main__':

  prob = DDP()

  # initial control sequence
  us = np.concatenate((np.tile([[0.1], [0.05]], (1, prob.N//2)),
    np.tile([[-0.1], [-0.05]], (1, prob.N//2))), axis=1)/2



  fig, axs = plt.subplots(1, 2)
  for j in range(len(prob.os_r)):
    circle = plt.Circle(prob.os_p[j], prob.os_r[j], color='r', linewidth=2, 
      fill=False)
    axs[0].add_patch(circle)
  axs[0].axis('equal')

  for i in range(50):
    dus, V, Vn, dV, a = ddp(us, prob)
    # update control
    us = us + dus
    prob.a = a
    xs = ddp_traj(us, prob)
    # axs[0].plot(xs[0,:], xs[1,:], '-g')
    prob.ko_x = prob.ko_x*1.5
    prob.ko_u = prob.ko_u*1.1
  # xs = prob.trajectory(1)
  axs[0].plot(xs[0,:], xs[1,:], '-m')

  axs[1].plot(np.arange(0, prob.tf, prob.h), us[0, :], color='b')
  axs[1].plot(np.arange(0, prob.tf, prob.h), us[1, :], color='g')
  axs[1].axhline(prob.u_bd[0], color='b', linestyle='--')
  axs[1].axhline(prob.u_bd[1], color='g', linestyle='--')
  axs[1].axhline(-prob.u_bd[0], color='b', linestyle='--')
  axs[1].axhline(-prob.u_bd[1], color='g', linestyle='--')
  axs[1].set_xlabel("sec.")
  axs[1].legend(["u_1", "u_2", 'u1_boundary', 'u2_boundary'], fontsize=8)

  plt.show()

  fig, axs = plt.subplots()
  # plt.axis([-10, 10, -10, 10])
  axs.axis('equal')
  for j in range(len(prob.os_r)):
    circle = plt.Circle(prob.os_p[j], prob.os_r[j], color='r', linewidth=2,
                        fill=False)
    axs.add_patch(circle)

  for k in range(prob.N):
    plt.scatter(xs[0, k], xs[1, k])
    for j in range(len(prob.os_r_move)):
      circle = plt.Circle(prob.os_p_move[j][k], prob.os_r_move[j], color='r', linewidth=2,
                          fill=False)
      axs.add_patch(circle)
    plt.pause(0.2)
    if k != prob.N-1:
      for j in range(len(prob.os_r_move)):
        axs.patches.pop()


  plt.show()
import numpy as np
import matplotlib.pyplot as plt
from trajopt_sqp import trajopt_sqp
# import car_ekf_estimation as car_est
# Optimal control example of a simple car model with disk obstacles


class Problem:

  def __init__(self):

    # time horizon and segments
    self.tf = 10.0
    self.N = 32
    self.h = self.tf / self.N

    # cost function parameters
    self.Q = np.diag([0, 0, 0, 0, 0])
    self.R = np.diag([1, 5])
    self.Qf = np.diag([5, 5, 1, 1, 1])

    # initial state
    self.x0 = np.array([-5, -2, -1.2, 0, 0])

    # add disk obstacles
    self.os_p = [[0.5,0.5],[-0.25,-0.25]]
    self.os_r = [1,2]
    self.os_p_move = []
    self.os_p_move.append(np.linspace([0,3],[1,0],self.N))
    self.os_p_move.append(np.linspace([2,5],[-1,-2],self.N))
    self.os_r_move = [2,2]

  def f(self, k, x, u):
    # car dynamics and jacobians

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
    # car cost (just standard quadratic cost)

    L = self.h * 0.5 * (np.transpose(x) @ self.Q @ x +
                        np.transpose(u) @ self.R @ u)
    Lx = self.h * self.Q @ x
    Lxx = self.h * self.Q
    Lu = self.h * self.R @ u
    Luu = self.h * self.R

    return L, Lx, Lxx, Lu, Luu

  def Lf(self, x):
    # car final cost (just standard quadratic cost)

    L = np.transpose(x) @ self.Qf @ x * 0.5
    Lx = self.Qf @ x
    Lxx = self.Qf

    return L, Lx, Lxx

  def con(self, k, x, u):
    numCon=len(self.os_r)
    numMov=len(self.os_r_move)
    constraints=np.zeros(numCon+numMov)
    for i in range(numCon):
      constraints[i]=self.os_r[i]**2-((x[0]-self.os_p[i][0])**2+(x[1]-self.os_p[i][1])**2)
    for i in range(numMov):
      constraints[numCon+i] = self.os_r_move[i] ** 2 - ((x[0] - self.os_p_move[i][k,0]) ** 2 + (x[1] - self.os_p_move[i][k,1]) ** 2)
    return constraints

  # def con(self, k, x, u):
  #   numCon=len(self.os_r)
  #   constraints=np.zeros(numCon)
  #   x1,_,_=self.f(k,x,u)
  #   D=x1[:2]-x[:2]
  #   l=np.linalg.norm(D)
  #   for i in range(numCon):
  #     A=self.os_p[i]-x[:2]
  #     temp=min(A @ D / l, l)
  #     f=x[:2]+max(min(A@D/l,l),0)/l*D
  #     constraints[i]=self.os_r[i]-np.linalg.norm(f-self.os_p[i])
  #   return constraints

  def conf(self, k, x):
    numCon = len(self.os_r)
    constraints = np.zeros(numCon)
    for i in range(numCon):
      constraints[i] = self.os_r[i] ** 2 - ((x[0] - self.os_p[i][0]) ** 2 + (x[1] - self.os_p[i][1]) ** 2)
    return constraints

  def traj(self, us):

    N = us.shape[1]

    xs = np.zeros((5, N + 1))
    xs[:, 0] = self.x0
    for k in range(N):
      xs[:, k + 1], _, _ = self.f(k, xs[:, k], us[:, k])

    return xs

  def plot_traj(self, xs, us):

    # plot state trajectory
    self.axs[0].plot(xs[0, :], xs[1, :], '-b')
    self.axs[0].axis('equal')
    self.axs[0].set_xlabel('x')
    self.axs[0].set_ylabel('y')

    # plot control trajectory
    self.axs[1].lines.clear()
    self.axs[1].plot(np.arange(0, self.tf, self.h), us[0, :], '-b')
    self.axs[1].plot(np.arange(0, self.tf, self.h), us[1, :], '-r')
    self.axs[1].relim()
    self.axs[1].autoscale_view()

    # drawing updated values
    fig.canvas.draw()

    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    fig.canvas.flush_events()


if __name__ == '__main__':

  prob = Problem()

  # initial control sequence
  us = np.concatenate((np.ones((2, prob.N // 2)) * 0.02,
                       np.ones((2, prob.N // 2)) * -0.02), axis=1)

  xs = prob.traj(us)

  plt.ion()
  fig, axs = plt.subplots(1, 2)
  prob.fig = fig
  prob.axs = axs

  # plot obstacles
  for j in range(len(prob.os_r)):
    circle = plt.Circle(prob.os_p[j], prob.os_r[j], color='r', linewidth=2,
                        fill=False)
    axs[0].add_patch(circle)



  plt.show()

  for j in range(len(prob.os_r_move)):
    for k in range(prob.N):
      circle = plt.Circle(prob.os_p_move[j][k], prob.os_r_move[j], color='r', linewidth=2,
                          fill=False)
      axs[0].add_patch(circle)
  axs[0].axis('equal')
  axs[0].set_xlabel("x")
  axs[0].set_ylabel("y")

  # plot initial trajectory
  prob.plot_traj(xs, us)
  axs[1].legend(["u_0", "u_1"])
  axs[1].set_xlabel('sec.')

  xs, us, cost = trajopt_sqp(xs, us, prob)

  plt.ioff()
  prob.plot_traj(xs, us)


  axs[0].plot(xs[0, :], xs[1, :], color="lime", linewidth=3)
  axs[0].axis('equal')
  axs[0].set_xlabel('x')
  axs[0].set_ylabel('y')
  plt.show()

  # xs, us, cost = trajopt_sqp(xs, us, prob)

  fig, axs = plt.subplots()
  plt.axis([-10, 10, -10, 10])
  for j in range(len(prob.os_r)):
    circle = plt.Circle(prob.os_p[j], prob.os_r[j], color='r', linewidth=2,
                        fill=False)
    axs.add_patch(circle)

  for k in range(prob.N):
    plt.scatter(xs[0, k],xs[1,k])
    for j in range(len(prob.os_r_move)):
      circle = plt.Circle(prob.os_p_move[j][k], prob.os_r_move[j], color='r', linewidth=2,
                          fill=False)
      axs.add_patch(circle)
    plt.pause(0.5)
    for j in range(len(prob.os_r_move)):
      axs.patches.pop()
  plt.show()
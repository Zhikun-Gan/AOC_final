from ast import While
from re import L
import numpy as np
from ddp_traj import ddp_traj
 
def diff_def(x, xn):
  # default state difference
  dx = xn - x
  return dx

def fd(k, x, u, e, S):
  # compute numerically the jacobians A=fx, B=fu of a given function 
  # S.f(k, x, u)

  f = S.f(k, x, u)[0]

  n = np.size(x)
  m = np.size(u)

  En = np.eye(n)
  Em = np.eye(m)

  A = np.zeros((n, n))
  B = np.zeros((n, m))

  for j in range(n):
    A[:, j] = (S.f(k, x + e*En[:, j], u)[0] - f) / e

  for j in range(m):
    B[:, j] = (S.f(k, x, u + e*Em[:,j])[0] - f) / e

  return A, B

def ddp(us, S):

  # Second-order numerical optimal control. The code computes
  # the optimal control adjustment for a given dynamical system
  #
  # params:
  # us - m-N matrix with discrete controls
  # S - problem data:
  #     S.L  : handle to the cost function
  #     S.f  : handle to the discrete dynamics functions
  #     S.mu : regularizing constant (default is 0)
  #     S.a  : initial step size (default is 1)
  #     S.diff : difference function (default is minus, i.e. vector space)
  #
  # return:
  #   dus: m-N matrix containing computed optimal change in control
  #   V: current value function
  #   Vn: new value function
  #   dV: predicted change in value function
  #   a: computed step-size along control search direction
  #
  #
  # Note: this implementation is most closely related to second-order 
  # metehods known as stage-wise Newton (SN) - Bertsekas, 2003 
  # and differential-dynamic-programming (DDP), Mayne, 1966. 
  # In this implementation second-order terms in the dynamics 
  # are ignored which corresponds to the linear-quadratic-subproblem
  # (LQS) approach (see also iterative-LQR (Todorov et al)).
  #
  # Disclaimer: the code is for education purposes only
  #
  # Author: Marin Kobilarov marin(at)jhu.edu

  if not (hasattr(S, "diff") and callable(S.diff)):
    S.diff = lambda x, xn: diff_def(x, xn)

  if not hasattr(S, 'mu'):
    S.mu = 0

  if not hasattr(S, 'mu0'):
    S.mu0 = 1e-3

  if not hasattr(S, 'dmu0'):
    S.dmu0 = 2
    
  if not hasattr(S, 'mumax'):
    S.mumax = 1e6

  if not hasattr(S, 'a'):
    S.a = 1

  if not hasattr(S, 'amin'):
    S.amin = 1e-32

  if not hasattr(S, 'n'):
    S.n = np.size(S.x0)

  if not hasattr(S, 'info'):
    S.info = 0

  n = S.n
  m = us.shape[0]
  N = us.shape[1]

  Ps = np.zeros((n, n, N+1))
  vs = np.zeros((n, N+1))

  cs = np.zeros((m, N))
  Ds = np.zeros((m, n, N))

  dus = np.zeros(np.shape(us))

  # integrate trajectory and get terminal cost
  xs = ddp_traj(us, S)
  L, Lx, Lxx, Lu, Luu = S.L(N, xs[:, -1], [])
  
  # initialize
  V = L
  v = Lx
  P = Lxx

  dV = np.zeros(2)

  Ps[:, :, N] = P
  vs[:, N] = v

  for k in range(N-1, -1, -1):

    x = xs[:, k]
    u = us[:, k]

    xn, A, B = S.f(k, x, u)

    if np.size(A) == 0 or np.size(B) == 0:
      A, B = fd(k, x, u, 1e-6, S) 

    L, Lx, Lxx, Lu, Luu = S.L(k, x, u)
    V = V + L

    Qx = Lx + np.transpose(A)@v
    Qu = Lu + np.transpose(B)@v
    Qxx = Lxx + np.transpose(A)@P@A
    Quu = Luu + np.transpose(B)@P@B
    Qux = np.transpose(B)@P@A

    mu = S.mu
    dmu = 1

    while(True):
      Quum = Quu + mu*np.eye(m)

      try:
        F = np.linalg.cholesky(Quum)

        # this is the standard quadratic rule specified by Tassa and Todorov
        dmu = min(1.0/S.dmu0, dmu/S.dmu0)
        if (mu*dmu > S.mu0):
          mu = mu*dmu
        else:
          mu = S.mu0
        
        if S.info:
          print("[I] Ddp::Backward: reduce mu=", mu, "at k=", k)
        break

      except:
        pass

      dmu = max(S.dmu0, dmu*S.dmu0)
      mu = max(S.mu0, mu*dmu)
      
      if S.info:
        print("[I] Ddp::Backward: increased mu=", mu, "at k=", k)

      if (mu > S.mumax):
        print("[W] Ddp::Backward: mu=", mu, "exceeded maximum")
        break

    if (mu > S.mumax):
      break

    # control law is du = c + D*dx
    cD = np.linalg.lstsq(-F, np.linalg.lstsq(np.transpose(F), np.concatenate(
      (np.transpose([Qu]), Qux), axis=1), rcond=None)[0], rcond=None)[0]
    c = cD[:, 0]
    D = cD[:, 1:]
    
    v = Qx + np.transpose(D) @ Qu
    P = Qxx + np.transpose(D) @ Qux
    
    dV = dV + np.array([np.transpose(c)@Qu, np.transpose(c)@Quu@c*0.5])

    vs[:, k] = v
    Ps[:, :, k] = P

    cs[:, k] = c
    Ds[:, :, k] = D

  s1 = 0.1
  s2 = 0.5
  b1 = 0.25
  b2 = 2.0

  a = S.a

  # measured change in V
  dVm = np.spacing(1)

  while dVm > 0:

    # variation
    dx = np.zeros(n)

    # varied x
    xn = S.x0

    # new measured cost
    Vn = 0

    for k in range(N):

      u = us[:, k]

      c = cs[:, k]
      D = Ds[:, :, k]

      du = a*c + D@dx
      un = u + du

      Ln, Lx, Lxx, Lu, Luu = S.L(k, xn, un)

      xn, A, B = S.f(k, xn, un)

      dx = S.diff(xs[:, k+1], xn)

      Vn = Vn + Ln

      dus[:, k] = du

    L, Lx, Lxx, Lu, Luu = S.L(N, xn, [])
    Vn = Vn + L

    dVm = Vn - V

    if dVm > 0:
      a = b1 * a
      if S.info:
        print("[I] Ddp: decrasing a=", a)

      if a < S.amin:
        break

      continue

    dVp = np.transpose(np.array([a, a*a]))@dV

    if (dVp != 0):
      r = dVm/dVp

      if r < s1:
        a = b1*a
      else:
        if r >= s2:
          a = b2*a
      if (S.info):
        print("[I] ddp: decreasing a=", a)

  return dus, V, Vn, dV, a
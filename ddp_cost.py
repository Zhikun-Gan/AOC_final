import numpy as np

def ddp_cost(xs, us, S):

  N = us.shape[1]
  J = 0

  for k in range(N+1):
    if k < N:
      L, Lx, Lxx, Lu, Luu = S.L(k, xs[:, k], us[:, k])
    else:
      L, Lx, Lxx, Lu, Luu = S.L(N, xs[:, k], [])
    J = J + L

  return J
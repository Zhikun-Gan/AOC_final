import numpy as np

def ddp_traj(us, S):

  N = us.shape[1]

  xs = np.zeros((np.size(S.x0), N+1))
  xs[:, 0] = S.x0
  for k in range(N):
    xs[:, k+1], _, _ = S.f(k, xs[:, k], us[:, k])

  return xs

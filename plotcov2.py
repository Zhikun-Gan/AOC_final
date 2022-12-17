import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def plotcov2(x, cov, ax, n_std=3.0, facecolor='none', **kwargs):

  pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
  # Using a special case to obtain the eigenvalues of this
  # two-dimensional dataset.
  ell_radius_x = np.sqrt(1 + pearson)
  ell_radius_y = np.sqrt(1 - pearson)
  ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                    facecolor=facecolor, edgecolor='blue', **kwargs)

  # Calculating the standard deviation of x from
  # the squareroot of the variance and multiplying
  # with the given number of standard deviations.
  scale_x = np.sqrt(cov[0, 0]) * n_std

  # calculating the standard deviation of y ...
  scale_y = np.sqrt(cov[1, 1]) * n_std

  transf = transforms.Affine2D() \
      .rotate_deg(45) \
      .scale(scale_x, scale_y) \
      .translate(x[0], x[1])

  ellipse.set_transform(transf + ax.transData)
  return ax.add_patch(ellipse)

from numpy import pi, cos, sin, sqrt, arange
import matplotlib.pyplot as pp


def create_points(num_pts, R):
    indices = arange(0, num_pts, dtype=float) + 0.5
    r = sqrt(indices / num_pts) * R
    theta = pi * (1 + 5**0.5) * indices

    X, Y = r * cos(theta), r * sin(theta)

    return X, Y


num_pts = 1000
R = 0.1
X, Y = create_points(num_pts, R)

pp.scatter(X, Y)
pp.show()
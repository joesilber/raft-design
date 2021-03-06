import math
import numpy as np
from numpy.polynomial import Polynomial
from astropy.table import Table
from scipy.spatial.transform import Rotation  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
import matplotlib.pyplot as plt
import os
from datetime import datetime

timestamp_fmt = '%Y%m%dT%H%M'
timestamp = datetime.now().astimezone().strftime(timestamp_fmt)

# polynomial fits of DESI focal surface asphere, as functions of radius
# c.f. DESI-0530-v18
# Z (mm) ... distance from origin CS5 parallel to optical axis
# S (mm) ... integrated distance along surface from optical axis
# N (deg) ... nutation angle (angle of positioner or fiducial local central axis)
# CRD (deg) ... chief ray deviation
Z = Polynomial([-2.33702E-05, 6.63924E-06, -1.00884E-04, 1.24578E-08, -4.82781E-10, 1.61621E-12, -5.23944E-15, 2.91680E-17, -7.75243E-20, 6.74215E-23])
S = Polynomial([9.95083E-06, 9.99997E-01, 1.79466E-07, 1.76983E-09, 7.24320E-11, -5.74381E-13, 3.28356E-15, -1.10626E-17, 1.89154E-20, -1.25367E-23])
N = Polynomial([1.79952E-03, 8.86563E-03, -4.89332E-07, -2.43550E-08, 9.04557E-10, -8.12081E-12, 3.97099E-14, -1.07267E-16, 1.52602E-19, -8.84928E-23])
CRD = Polynomial([0, 3.4019e-3, -2.8068e-5, 4.4307e-7, -2.4009e-9, 5.1158e-12, -3.9825e-15])
sphR = 4978  # mm, approx spherical radius of curvature

# taken from desimeter/xy2qs.py on 2022-05-19
def s2r(s):
    '''Convert radial distance along focal surface to polar coordinate r.'''
    # fitted on desimodel/data/focalplane/fiberpos.ecsv
    # residuals are < 0.4 microns
    s = s if isinstance(s, np.ndarray) else np.array(s)
    c = np.array([-2.60833797e-03,  6.40671681e-03, -5.64913181e-03,  6.99354170e-04, -2.13171265e-04,  1.00000009e+00,  9.75790364e-07])
    pol = np.poly1d(c)
    r = 400.*pol(s/400.)
    return r

# raft geometry inputs
B = 80.0  # mm, base of raft triangle
L = 657.0  # mm, length of raft from origin (at center fiber tip) to rear
g = 2.0  # mm, gap between triangles at rear

# raft outline
h1 = B * 3**0.5 / 2  # height from base of triangle to opposite tip
h2 = B / 3**0.5 / 2 # height from base of triangle to center
h3 = h1 - h2  # height from center of triangle to tip
basic_raft_x = [-B/2,  0, B/2]
basic_raft_x += [basic_raft_x[0]]
basic_raft_y = [-h2, h3, -h2]
basic_raft_y += [basic_raft_y[0]]
basic_raft_z = [0]*len(basic_raft_x)
basic_raft_x += basic_raft_x
basic_raft_y += basic_raft_y
basic_raft_z += [-L]*len(basic_raft_z)

# nominal spacing between triangles
spacing_rear = g + 2*h2
spacing_front = spacing_rear * sphR / (sphR - L)
crd_margin = lambda radius: sphR * math.radians(abs(CRD(radius + spacing_front) - CRD(radius)))

# wedge envelope geometry
envelope_r_max = 416  # mm, max allowable mechanical envelope
envelope_angle = 72  # deg
outer_radius_plot_angles = np.radians(np.linspace(0, envelope_angle, 36))
envelope_x = [0] + [envelope_r_max * math.cos(a) for a in outer_radius_plot_angles]
envelope_x += [envelope_x[-1]] + [0]
envelope_y = [0] + [envelope_r_max * math.sin(a) for a in outer_radius_plot_angles]
envelope_y += [envelope_y[-1]] + [0]
envelope_z = [0]*len(envelope_x)
envelope_x += envelope_x
envelope_y += envelope_y
envelope_z += [-L]*len(envelope_z)

# table structure for raft positions and orientations
t = Table(names=['x', 'y',  'z', 'radius', 'S', 'precession', 'nutation', 'spin'])
def fill_cols(m):
    '''Fill in other columns, knowing x and y'''
    m['radius'] = math.hypot(m['x'], m['y'])
    m['z'] = Z(m['radius'])
    m['S'] = S(m['radius'])
    m['precession'] = np.rad2deg(np.arctan2(m['y'], m['x']))
    m['nutation'] = N(m['radius'])

# pattern row 1
t.add_row({'x': 68, 'y': 56, 'spin': 180})
t.add_row({'x': 116, 'y': 28, 'spin': 0})
t.add_row({'x': 166, 'y': 56, 'spin': 180})
t.add_row({'x': 216, 'y': 28, 'spin': 0})
t.add_row({'x': 266, 'y': 56, 'spin': 180})
t.add_row({'x': 318, 'y': 28, 'spin': 0})
t.add_row({'x': 375, 'y': 56, 'spin': 180})

# pattern row 2
t.add_row({'x': 75, 'y': 112, 'spin': 0})
t.add_row({'x': 124, 'y': 140, 'spin': 180})
t.add_row({'x': 173, 'y': 112, 'spin': 0})
t.add_row({'x': 224, 'y': 140, 'spin': 180})
t.add_row({'x': 277, 'y': 114, 'spin': 0})
t.add_row({'x': 337, 'y': 140, 'spin': 180})

# pattern row 3
t.add_row({'x': 102, 'y': 196, 'spin': 0})
t.add_row({'x': 154, 'y': 224, 'spin': 180})
t.add_row({'x': 205, 'y': 196, 'spin': 0})
t.add_row({'x': 261, 'y': 224, 'spin': 180})
t.add_row({'x': 316, 'y': 196, 'spin': 0})

# pattern row 4
t.add_row({'x': 142, 'y': 284, 'spin': 0})
t.add_row({'x': 200, 'y': 314, 'spin': 180})
t.add_row({'x': 254, 'y': 288, 'spin': 0})

for row in t:
    fill_cols(row)

# counter-act precessions
t['spin'] -= t['precession']

# print stats and write table
t.pprint_all()
n_rafts = len(t)
n_robots = n_rafts*72
basename = f'{timestamp}_desi2_layout_{n_rafts}rafts_{n_robots}robots'
filename = basename + '.csv'
t.write(filename, overwrite=True)
print(f'Saved table to {os.path.abspath(filename)}')

# plot rafts
fig = plt.figure(figsize=plt.figaspect(1)*2, dpi=200, tight_layout=True)
ax = fig.add_subplot(projection='3d', proj_type='ortho')
outlines = []
for row in t:
    basic = np.transpose([basic_raft_x, basic_raft_y, basic_raft_z])
    r = Rotation.from_euler('ZYZ',(row['precession'], row['nutation'], row['spin']), degrees=True)
    rotated = r.apply(basic)
    translated = rotated + [row['x'], row['y'], row['z']]
    f = np.transpose(translated)
    ax.plot(f[0], f[1], f[2], '-')

# plot envelope
ax.plot(envelope_x, envelope_y, envelope_z, 'k--')

# from: https://newbedev.com/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to-x-and-y
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

set_axes_equal(ax)
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_zlabel('z (mm)')
ax.set_box_aspect([1, 1, 1])
ax.set_proj_type('ortho')
ax.azim = -114
ax.elev = 23

num_text = f'{n_rafts} rafts --> {n_robots} robots'
plt.title(f'{timestamp}\n{num_text}')

filename = basename + '.png'
plt.savefig(filename)
print(f'Plotted {num_text}.')
print(f'Saved plot to {os.path.abspath(filename)}')


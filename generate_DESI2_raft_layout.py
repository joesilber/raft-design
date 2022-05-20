import math
import numpy as np
from numpy.polynomial import Polynomial
from astropy.table import Table
from scipy.spatial.transform import Rotation  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

# polynomial fits of DESI focal surface asphere, as functions of radius
# c.f. DESI-0530-v18
# Z (mm) ... distance from origin CS5 parallel to optical axis
# S (mm) ... integrated distance along surface from optical axis
# N (deg) ... nutation angle (angle of positioner or fiducial local central axis)
Z = Polynomial([-2.33702E-05, 6.63924E-06, -1.00884E-04, 1.24578E-08, -4.82781E-10, 1.61621E-12, -5.23944E-15, 2.91680E-17, -7.75243E-20, 6.74215E-23])
S = Polynomial([9.95083E-06, 9.99997E-01, 1.79466E-07, 1.76983E-09, 7.24320E-11, -5.74381E-13, 3.28356E-15, -1.10626E-17, 1.89154E-20, -1.25367E-23])
N = Polynomial([1.79952E-03, 8.86563E-03, -4.89332E-07, -2.43550E-08, 9.04557E-10, -8.12081E-12, 3.97099E-14, -1.07267E-16, 1.52602E-19, -8.84928E-23])

# raft geometry inputs
B = 80.0  # mm, base of raft triangle
L = 657.0  # mm, length of raft from origin (at center fiber tip) to rear

# raft outline
h1 = B * 3**0.5 / 2  # height from base of triangle to opposite tip
h2 = B / 3**0.5  # height from base of triangle to center
h3 = h1 - h2  # height from center of triangle to tip
basic_raft_x = np.array([B/2,  0, -B/2, B/2])
basic_raft_y = np.array([-h3, h2,  -h3, -h3])
basic_raft_z = np.zeros_like(basic_raft_x)

# positions and spin angles
t = Table(names=['x',   'y',  'z', 'radius', 'S', 'precession', 'nutation', 'spin'],
         )
seed0 = {'x': 68.5, 'y': 56.0, 'spin': 180.0}
t.add_row(seed0)

t['radius'] = (t['x']**2 + t['y']**2)**0.5
t['z'] = Z(t['radius'])
t['S'] = S(t['radius'])
t['precession'] = np.rad2deg(np.arctan2(t['x'], t['y']))
t['nutation'] = N(t['radius'])
t['spin'] -= t['precession']  # counter-act precession

t.pprint_all()

# plot rafts
outlines = []
for row in t:
    basic = np.transpose([basic_raft_x, basic_raft_y, basic_raft_z])
    r = Rotation.from_euler('zyz', [row['precession'], row['nutation'], row['spin']], degrees=True)
    rotated = r.apply(basic)
    translated = rotated + [row['x'], row['y'], row['z']]
    print('')
    print('rotated', rotated)
    print('')
    print('translated', translated)
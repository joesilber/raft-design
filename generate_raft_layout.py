#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Generates a layout of triangular raft modules on a focal surface. Layout is saved
as a csv table, as well as illustrated with an approximately to-scale 3D plot.
Contact: Joe Silber, jhsilber@lbl.gov
'''

import math
import numpy as np
from numpy.polynomial import Polynomial
from astropy.table import Table
from scipy.spatial.transform import Rotation  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import os
from datetime import datetime
import fitcircle
import argparse

timestamp_fmt = '%Y%m%dT%H%M'
timestamp = datetime.now().astimezone().strftime(timestamp_fmt)

# GEOMETRY OF FOCAL SURFACE DESIGNS
# ---------------------------------
# INPUTS:
#   Z (mm) ... distance from origin parallel to optical axis
#   CRD (deg) ... chief ray deviation (rotation from x-axis toward z-axis)
#   vigR (mm) ... nominal vignette radius (i.e. size of focal surface)
#   file ... optional alternate to Z, CRD --- table to read and interpolate
#   z_sign ... optional, to change direction of +z
# DERIVED:
#   S (mm) ... integrated distance along surface from optical axis
#   NORM (deg) ... normal angle (rotation from x-axis toward z-axis, i.e. in negative direction about y-axis)
#   NUT (deg) ... nutation angle, equivalent to chief ray. NUT = -(NORM + CRD). (rotation from z-axis toward x-axis, i.e. in positive direction about y-axis) 
focal_surfaces = {
    'MM1536_cfg1_2021-09-10':
        {'description': 'MegaMapper 1536 config 1, 2021-09-21',
        'file': 'MM1536_cfg1_2021-09-10.csv',
        'z_sign': -1,
        'vigR': 613.2713,
        },
    'DESI':
        {'description': 'DESI Echo22 corrector, c.f. DESI-0530-v18',
        'Z': Polynomial([-2.33702E-05, 6.63924E-06, -1.00884E-04, 1.24578E-08, -4.82781E-10, 1.61621E-12, -5.23944E-15, 2.91680E-17, -7.75243E-20, 6.74215E-23]),
        'CRD': Polynomial([0, 3.4019e-3, -2.8068e-5, 4.4307e-7, -2.4009e-9, 5.1158e-12, -3.9825e-15]),
        'vigR': 406.,
        },
    }
focsurf_numbers = {i: name for i, name in enumerate(focal_surfaces)}

# command line argument parsing
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--focal_surface', type=int, default=0, help=f'select focal surface design by number, valid options are {focsurf_numbers}')
parser.add_argument('-b', '--raft_tri_base', type=float, default=80.0, help='mm, length of base edge of a raft triangle')
parser.add_argument('-l', '--raft_length', type=float, default=657.0, help='mm, length of raft from origin (at center fiber tip) to rear')
parser.add_argument('-g', '--raft_rear_gap', type=float, default=2.0, help='mm, gap between triangles at rear')
parser.add_argument('-c', '--raft_chamfer', type=float, default=8.6, help='mm, chamfer at triangle tips')
userargs = parser.parse_args()

# set up geometry functions
focsurf_number = focsurf_numbers[userargs.focal_surface]
focsurf = focal_surfaces[focsurf_number]
if all(label in focsurf for label in {'Z', 'CRD'}):
    R2Z = focsurf['Z']  # should be a function accepting numpy array argument for radius, returning z
    R2CRD = focsurf['CRD']  # should be a function accepting numpy array argument for radius, returning chief ray deviation
elif 'file' in focsurf:
    t = Table.read(focsurf['file'], comment='#')
    R2Z = interpolate.interp1d(t['R'], t['Z'], kind='cubic')
    if 'CRD' in t:
        R2CRD = interpolate.interp1d(t['R'], t['CRD'])
    else:
        R2CRD = Polynomial([0])  # in the absence of chief ray deviation information
        print(f'WARNING: no chief ray deviation defined, letting CRD(R)=0')
else:
    assert False, 'unrecognized geometry input data'
vigR = focsurf['vigR']  # should be a scalar
if 'z_sign' in focsurf:
    _R2Z = R2Z
    R2Z = lambda x: np.sign(focsurf['z_sign']) * _R2Z(x)
r = np.linspace(0, vigR, 10000)
z = R2Z(r)
dr = np.diff(r)
dz = np.diff(z)
dzdr = dz / dr
ds = (1 + dzdr**2)**0.5 * dr
s = np.cumsum(ds)
s = np.insert(s, 0, 0.)  # first value of path length is 0
Z2R = interpolate.interp1d(z, r)
R2S = interpolate.interp1d(r, s)
S2R = interpolate.interp1d(s, r)
norm = np.degrees(np.arctan(dzdr))
R2NORM = interpolate.interp1d(r[:-1], norm)
NORM2R = interpolate.interp1d(norm, r[:-1])
crd = R2CRD(r)
nut = - (norm + crd[:-1])
CRD2R = interpolate.interp1d(crd, r)
R2NUT = interpolate.interp1d(r[:-1], nut)
NUT2R = interpolate.interp1d(nut, r[:-1])
circlefit_data = np.transpose([np.append(r, -r), np.append(z, z)])
rz_ctr, sphR = fitcircle.FitCircle().fit(circlefit_data)  # best-fit sphere to (r, z)
concavity = np.sign(rz_ctr[1])  # +1 --> concave, -1 --> convex

# basic raft outline
RB = userargs.raft_tri_base
RL = userargs.raft_length
RH = userargs.raft_chamfer
h1 = RB * 3**0.5 / 2  # height from base of triangle to opposite tip
h2 = RB / 3**0.5 / 2 # height from base of triangle to center
h3 = h1 - h2  # height from center of triangle to tip
raft_profile_x = [-RB/2,  0, RB/2]
raft_profile_y = [-h2, h3, -h2]
raft_profile_z = [0, 0, 0]
raft_profile = np.transpose([raft_profile_x, raft_profile_y, raft_profile_z])

# offset to average out the defocus of all the robots on a raft
raft_targetable_area = RB**2/2 - 3*RC**2/2
above_below_equal_area_radius = (raft_targetable_area/2 / math.pi)**0.5  # i.e. for a circle centered on raft that contains same area inside as in the rest of the raft
avg_focus_offset = above_below_equal_area_radius**2 / sphR / 2
avg_focus_offset *= concavity  # apply sign for concave vs convex focal surface

basic_raft_x += [basic_raft_x[0]]
basic_raft_y = [-h2, h3, -h2]
basic_raft_y += [basic_raft_y[0]]
basic_raft_z = [0]*len(basic_raft_x)
basic_raft_x += basic_raft_x
basic_raft_y += basic_raft_y
basic_raft_z += [RL]*len(basic_raft_z)


class Raft:
    '''Represents a single triangular raft.'''
    
    def __init__(self, x=0, y=0, spin=0):
        '''
        x ... x location of center of front triangle
        y ... y location of center of front triangle
        spin0 ... rotation of triangle, *not* including precession compensation
        '''
        self.x = x
        self.y = y
        self.spin = spin

    @property
    def r(self):
        '''radial position of center of raft at front'''
        return math.hypot(self.x, self.y)

    @property
    def precession(self):
        '''angular position about the z-axis, same as precession'''
        return math.atan2(self.y, self.x)

    @property
    def nutation(self):
        '''angle w.r.t. z-axis (i.e. matches chief ray at center of raft)'''
        return R2NUT(R)
    
    @property
    def spin(self)
        '''rotation about raft's local z-axis, *including* compensation for
        precession (since raft orientation is defined by a 3-2-3 Euler rotation)'''
        return self.spin0 - precession

    @property
    def front_poly(self):
        '''polygon of raft profile at front (i.e. at focal surface)'''
        poly = raft_profile + [0, 0, avg_focus_offset]
        return self._place_poly(poly)

    @property
    def rear_poly(self):
        '''polygon of raft profile at rear (i.e. at connectors bulkhead, etc)'''
        poly = raft_profile + [0, 0, avg_focus_offset - RL]
        return self._place_poly(poly)

    @property
    def poly3d(self):
        '''intended for 3D plotting, includes front and rear closed polygons'''
        front = self.front_poly.tolist()
        rear = self.rear_poly.tolist()
        poly3d = front + [front[0]]
        for i in range(len(rear) - 1):
            poly3d += [rear[i], rear[i+1], front[i+1]]
        return poly3d

    def front_gap(self, other_raft):
        '''Returns min distance and perpendicular unit vector from closest segment on this
        raft's front polygon toward corresponding closest point on "other" raft.'''
        return Raft.poly_gap(other.front_poly, self.front_poly)

    def rear_gap(self, other_raft):
        '''Returns min distance and perpendicular unit vector from closest segment on this
        raft's front polygon toward corresponding closest point on "other" raft.'''
        return Raft.poly_gap(other.rear_poly, self.rear_poly)

    def _place_poly(self, poly):
        '''Transform a polygon (N x 3) from the origin to the raft's center position on the
        focal surface. The polygon is first rotated such that a vector (0, 0, 1) becomes
        its final orientation when placed at the corresponding radius, and such that a point
        (0, 0, 0) will land on the focal surface.'''
        rot = Rotation.from_euler('ZYZ', (self.precession, self.nutation, self.spin), degrees=True)
        rotated = rot.apply(poly)
        translated = rotated + [self.x, self.y, R2Z(self.r)]
        return translated

    @staticmethod
    def poly_gap(poly1, poly2):
        '''Returns a vector for closest distance between two polygons. This is calculated
        from segments defined by poly2 to vertices defined by poly1. If the two polygons
        overlap, returns None. The input polygons should be Nx2 arrays of (x, y) or Nx3
        arrays of (x, y, z) vertices. Returns minimum distance magnitude and unit vector
        pointing perpendicularly from the nearest poly2 segment to poly1 point.'''
        if Raft.polygons_collide(poly1, poly2):
            return None
        test_pts = [np.array(pt) for pt in poly1]
        segment_pts = [(poly2[i], poly2[i+1]) for i in range(len(poly2) - 1)]
        segment_pts += [(poly2[-1] - poly2[0])]  # close the polygon with last segment
        min_dist = math.inf
        min_vec = None
        for seg in segment_pts:
            s0 = np.array(seg[0])
            s1 = np.array(seg[1])
            seg_vec = s1 - s0
            seg_mag = np.sqrt(np.sum(np.power(seg_vec, 2)))
            seg_unit = seg_vec / seg_mag
            for pt in test_pts:
                dist = np.dot(pt - s0, seg_unit)
                if dist < min_dist:
                    min_dist = dist
                    min_vec_start_pt = s0 + min_dist*seg_unit
                    min_vec = pt - min_vec_start_pt
        min_vec_unit = min_vec / min_dist
        return min_dist, min_vec_unit

    @staticmethod
    def polygons_collide(poly1, poly2):
        """Check whether two closed polygons collide.

        poly1 ... Nx2 array of the 1st polygon's vertices
        poly2 ... Nx2 array of the 2nd polygon's vertices

        Returns True if the polygons intersect, False if they do not.

        The algorithm is by detecting intersection of line segments, therefore the case of
        a small polygon completely enclosed by a larger polygon will return False. Not checking
        for this condition admittedly breaks some conceptual logic, but this case is not
        anticipated to occur given the DESI petal geometry, and speed is at a premium.
        """
        for i in range(len(poly1) - 1):
            for j in range(len(poly2) - 1):
                if Raft.segments_intersect(poly1[i], poly1[i+1], poly2[j], poly2[j+1]):
                    return True
        return False

    @staticmethod
    def segments_intersect(A1, A2, B1, B2):
        """Checks whether two 2d line segments intersect. The endpoints for segments
        A and B are each a pair of (x, y) coordinates.
        """
        dx_A = A2[0] - A1[0]
        dy_A = A2[1] - A1[1]
        dx_B = B2[0] - B1[0]
        dy_B = B2[1] - B1[1]
        delta = dx_B * dy_A - dy_B * dx_A
        if delta == 0.0:
            return False  # parallel segments
        s = (dx_A * (B1[1] - A1[1]) + dy_A * (A1[0] - B1[0])) / delta
        t = (dx_B * (A1[1] - B1[1]) + dy_B * (B1[0] - A1[0])) / (-delta)
        return (0 <= s <= 1) and (0 <= t <= 1)


# nominal spacing between triangles
spacing_rear = userargs.raft_rear_gap + 2*h2
spacing_front = spacing_rear * sphR / (sphR - L)
crd_margin = lambda radius: sphR * math.radians(abs(R2CRD(radius + spacing_front) - R2CRD(radius)))

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

# generate raft instances
rafts = []
for row in t:
    rafts.append(Raft(x=row['x'], y=row['y'], spin=row['spin']))

# print stats and write table
t.pprint_all()
n_rafts = len(rafts)
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


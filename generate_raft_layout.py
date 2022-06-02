#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Generates a layout of triangular raft modules on a focal surface. Layout is saved
as a csv table, as well as illustrated with an approximately to-scale 3D plot.
Contact: Joe Silber, jhsilber@lbl.gov
'''

import time
start_time = time.perf_counter()

import math
import numpy as np
from numpy.polynomial import Polynomial
from astropy.table import Table
from scipy.spatial.transform import Rotation  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
import scipy.interpolate as interpolate
from scipy import optimize
import matplotlib.pyplot as plt
import os
from datetime import datetime
import argparse


timestamp_fmt = '%Y%m%dT%H%M'
timestamp = datetime.now().astimezone().strftime(timestamp_fmt)
interp1d = lambda x, y: interpolate.interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')

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
focsurfs_index = {i: name for i, name in enumerate(focal_surfaces)}

# command line argument parsing
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--focal_surface_number', type=int, default=0, help=f'select focal surface design by number, valid options are {focsurfs_index}')
parser.add_argument('-b', '--raft_tri_base', type=float, default=80.0, help='mm, length of base edge of a raft triangle')
parser.add_argument('-l', '--raft_length', type=float, default=657.0, help='mm, length of raft from origin (at center fiber tip) to rear')
parser.add_argument('-g', '--raft_gap', type=float, default=3.0, help='mm, minimum gap between rafts')
parser.add_argument('-c', '--raft_chamfer', type=float, default=8.6, help='mm, chamfer at triangle tips')
parser.add_argument('-w', '--wedge', type=float, default=360.0, help='deg, angle of wedge envelope, argue 360 for full circle')
parser.add_argument('-xo', '--x_offset', type=float, default=0.0, help='mm, x offset the seed of raft pattern (note base*sqrt(3)/2 often useful)')
parser.add_argument('-yo', '--y_offset', type=float, default=0.0, help='mm, y offset the seed of raft pattern (note base/sqrt(3)/2 often useful)')
userargs = parser.parse_args()

# set up geometry functions
focsurf_name = focsurfs_index[userargs.focal_surface_number]
focsurf = focal_surfaces[focsurf_name]
CRD2R_undefined = False
if all(label in focsurf for label in {'Z', 'CRD'}):
    R2Z = focsurf['Z']  # should be a function accepting numpy array argument for radius, returning z
    R2CRD = focsurf['CRD']  # should be a function accepting numpy array argument for radius, returning chief ray deviation
elif 'file' in focsurf:
    t = Table.read(focsurf['file'], comment='#')
    R2Z = interp1d(t['R'], t['Z'])
    if 'CRD' in t:
        R2CRD = interp1d(t['R'], t['CRD'])
    else:
        R2CRD = Polynomial([0])  # in the absence of chief ray deviation information
        CRD2R_undefined = True
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
Z2R = interp1d(z, r)
R2S = interp1d(r, s)
S2R = interp1d(s, r)
norm = np.degrees(np.arctan(dzdr))
R2NORM = interp1d(r[:-1], norm)
NORM2R = interp1d(norm, r[:-1])
crd = R2CRD(r)
nut = - (norm + crd[:-1])
if not CRD2R_undefined:
    CRD2R = interp1d(crd, r)
R2NUT = interp1d(r[:-1], nut)
NUT2R = interp1d(nut, r[:-1])

# best-fit sphere
calc_sphR = lambda z_ctr: (r**2 + (z - z_ctr)**2)**0.5
def calc_sphR_err(z_ctr):
    sphR_test = calc_sphR(z_ctr)
    errors = sphR_test - np.mean(sphR_test)
    scalar_error = np.sum(np.power(errors, 2))
    return scalar_error
typical_fov = 3.0  # deg
z_guess = np.sign(np.mean(z)) * np.max(r) / np.radians(typical_fov/2)
result = optimize.least_squares(fun=calc_sphR_err, x0=z_guess)
z_ctr = float(result.x)
sphR = abs(z_ctr)
is_convex = np.sign(z_ctr) < 1  # convention where +z is toward the fiber tips

# basic raft outline
RB = userargs.raft_tri_base
RL = userargs.raft_length
RC = userargs.raft_chamfer
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
avg_focus_offset *= -1 if is_convex else +1

_raft_id_counter = 0
class Raft:
    '''Represents a single triangular raft.'''
    
    def __init__(self, x=0, y=0, spin0=0):
        '''
        x ... [mm] x location of center of front triangle
        y ... [mm] y location of center of front triangle
        spin0 ... [deg] rotation of triangle, *not* including precession compensation
        '''
        global _raft_id_counter
        self.id = _raft_id_counter
        _raft_id_counter += 1
        self.x = x
        self.y = y
        self.spin0 = spin0
        self.neighbors = []

    @property
    def r(self):
        '''radial position [mm] of center of raft at front'''
        return math.hypot(self.x, self.y)
    
    @property
    def z(self):
        '''z position [mm] of center of raft at front'''
        offset_correction = avg_focus_offset * math.cos(math.radians(self.nutation))
        return R2Z(self.r) + offset_correction

    @property
    def precession(self):
        '''angular position [deg] about the z-axis, same as precession'''
        return math.degrees(math.atan2(self.y, self.x))

    @property
    def nutation(self):
        '''angle [deg] w.r.t. z-axis (i.e. matches chief ray at center of raft)'''
        return R2NUT(self.r)
    
    @property
    def spin(self):
        '''rotation [deg] about raft's local z-axis, *including* compensation for
        precession (since raft orientation is defined by a 3-2-3 Euler rotation)'''
        return self.spin0 - self.precession

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
        poly3d += [rear[i+1], rear[0]]
        return poly3d

    def front_gap(self, other_raft):
        '''Returns min distance and perpendicular unit vector from closest segment on this
        raft's front polygon toward corresponding closest point on "other" raft.'''
        return Raft.poly_gap(other_raft.front_poly, self.front_poly)

    def rear_gap(self, other_raft):
        '''Returns min distance and perpendicular unit vector from closest segment on this
        raft's front polygon toward corresponding closest point on "other" raft.'''
        return Raft.poly_gap(other_raft.rear_poly, self.rear_poly)

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
        '''Returns a magnitude and direction unit vector for closest distance between two polygons.
        This is calculated from segments defined by poly2 to vertices defined by poly1. If the two
        polygons overlap, returns (None, None). The input polygons should be Nx2 arrays of (x, y)
        or Nx3 arrays of (x, y, z) vertices. The returned unit vector points perpendicularly from
        the nearest poly2 segment toward the corresponding nearest poly1 point.'''
        if Raft.polygons_collide(poly1, poly2):
            return None, None
        test_pts = [np.array(pt) for pt in poly1]
        segment_pts = [(poly2[i], poly2[i+1]) for i in range(len(poly2) - 1)]
        segment_pts += [(poly2[-1], poly2[0])]  # close the polygon with last segment
        min_dist = math.inf
        min_vec = None
        for seg in segment_pts:
            s0 = np.array(seg[0])
            s1 = np.array(seg[1])
            seg_vec = s1 - s0
            seg_mag = np.sqrt(seg_vec.dot(seg_vec))
            seg_unit = seg_vec / seg_mag
            for pt in test_pts:
                s2_mag = np.dot(pt - s0, seg_unit)
                if s2_mag < 0:
                    gap_vec_start = s0
                elif s2_mag > seg_mag:
                    gap_vec_start = s1
                else:
                    s2 = s2_mag * seg_unit
                    gap_vec_start = s2
                gap_vec = pt - gap_vec_start
                gap_mag = np.sqrt(gap_vec.dot(gap_vec))
                if gap_mag < min_dist:
                    min_dist = gap_mag
                    min_vec_unit = gap_vec / gap_mag
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


# generate grid of raft center points
# (based on two sets of staggered equilateral triangles)
if is_convex:
    # JHS 2022-05-31: This estimation for front gap is simple and conservative.
    # A more efficient algorithm would use the appropriate delta_crd for each
    # raft's particular radial position, rather than uniformly assuming worst-case
    # chief ray deviation. Should definitely address this before producing any
    # final layout for a *convex* focal surface (such as DESI's Echo22 corrector).
    coarse_r = np.arange(0, vigR, RB)
    coarse_crd = R2CRD(coarse_r)
    delta_crd_rad = np.radians(np.diff(coarse_crd))
    max_delta_crd_rad = max(delta_crd_rad)
    delta_gap_rad = (userargs.raft_gap + RB) / (sphR - RL)
    front_gap = (max_delta_crd_rad + delta_gap_rad) * sphR - RB
else:
    delta_gap_rad = userargs.raft_gap / sphR
    bidirectional_total_angle = math.acos(math.cos(delta_gap_rad)**2)
    front_gap = sphR * bidirectional_total_angle
gap_expansion_factor = 2.0  # over-expands the nominal pattern, with goal of assuring no initial overlaps (iterative nudging will contract this later)
spacing_x = RB + front_gap / (math.sqrt(3)/2) * gap_expansion_factor
spacing_y = spacing_x * math.sqrt(3)/2
half_width_count = math.ceil(vigR / spacing_x) + 1
rng = range(-half_width_count, half_width_count+1)
natural_grid = {'x': [], 'y': [], 'spin0': []}
for j in rng:
    x = [spacing_x*i + userargs.x_offset for i in rng]
    if j % 2:
        x = [u + spacing_x/2 for u in x]
    y = [spacing_y * j + userargs.y_offset]*len(x)

    # upward pointing triangles
    natural_grid['x'] += x
    natural_grid['y'] += y
    natural_grid['spin0'] += [0]*len(x)

    # downward pointing triangles
    natural_grid['x'] += [u + spacing_x/2 for u in x]
    natural_grid['y'] += [v + spacing_y/3 for v in y]
    natural_grid['spin0'] += [180]*len(x)

# flatten grid from its natural, implicit, curved space of focal surface
# to cartesian (where focal surface shape as function of radius applies)
q = np.arctan2(natural_grid['y'], natural_grid['x'])
s = np.hypot(natural_grid['x'], natural_grid['y'])
r = S2R(s)
grid = {'x': r * np.cos(q),
        'y': r * np.sin(q),
        'spin0': natural_grid['spin0'],
        }

# vignette & wedge envelope plottable geometry
a = np.radians(np.linspace(0, userargs.wedge, 100))
envelope_x = vigR * np.cos(a)
envelope_y = vigR * np.sin(a)
not_full_circle = abs(userargs.wedge) < 360
if not_full_circle:
    envelope_x = np.append(envelope_x, [0, vigR])
    envelope_y = np.append(envelope_y, [0, 0])
envelope_z = np.zeros_like(envelope_x)

# vignette & wedge raft selection
raft_position_radii = np.hypot(grid['x'], grid['y'])
remove = raft_position_radii > vigR
if not_full_circle:
    raft_position_angles = np.degrees(np.arctan2(grid['y'], grid['x']))
    remove |= raft_position_angles > max(0, userargs.wedge)
    remove |= raft_position_angles < min(0, userargs.wedge)
keep = np.logical_not(remove)
for key in grid:
    grid[key] = np.array(grid[key])[keep]

# table structure for raft positions and orientations
t = Table(grid)
other_cols = ['z', 'radius', 'precession', 'nutation', 'spin', 'id']
for col in other_cols:
    t[col] = [0]*len(t)

# generate raft instances
rafts = []
for row in t:
    raft = Raft(x=row['x'], y=row['y'], spin0=row['spin0'])
    rafts += [raft]
    row['id'] = raft.id

# determine neighbors
for raft in rafts:
    dist = np.hypot(t['x'] - raft.x, t['y'] - raft.y)
    neighbor_selection = dist < spacing_x * 1.2  # may be conservatively inclusive, but that's ok, not too costly
    neighbor_selection &= raft.id != t['id']  # skip self
    neighbor_selection_ids = np.flatnonzero(neighbor_selection)
    raft.neighbors = [r for r in rafts if r.id in neighbor_selection_ids]

# function to assess all gaps to neighbors
def calc_gaps():
    gaps = {'id': [], 'raft': [], 'min_gap_front': [], 'min_gap_rear': []}
    for raft in rafts:
        gaps_front = []
        gaps_rear = []
        for neighbor in raft.neighbors:
            gap_front, dir_gap_front = raft.front_gap(neighbor)
            gap_rear, dir_gap_rear = raft.rear_gap(neighbor)
            gaps_front += [gap_front]
            gaps_rear += [gap_rear]
        gaps['min_gap_front'] += [0 if None in gaps_front else min(gaps_front)]
        gaps['min_gap_rear'] += [0 if None in gaps_rear else min(gaps_rear)]
        gaps['id'] += [raft.id]
        gaps['raft'] += [raft]
        gaps_table = Table(gaps)
    return gaps_table

### break out the neighbor calcs so can just do a set of neighbors
### include main table in the program so that it can be updated

# iteratively squeeze the pattern for more optimal close-packing
# max_iters = 10 * len(rafts)
# goal_delta = 0.1  # mm
# for iter in range(max_iters):
#     gaps = calc_gaps()
#     biggest_gap = gaps.sort(['min_gap_front', 'min_gap_rear'], reverse=True)
#     if delta < goal_delta:
#         break

# print stats and write table

gap_timer = time.perf_counter()
gaps = calc_gaps()
print(f'Calculated gaps for {len(rafts)} rafts in {time.perf_counter() - gap_timer} sec.')

gaps.sort('id')
t.sort('id')
for key in ['min_gap_front', 'min_gap_rear']:
    t[key] = gaps[key]
for raft in rafts:
    row_idx = int(np.flatnonzero(t['id'] == raft.id))
    row = t[row_idx]
    row['radius'] = raft.r
    row['z'] = raft.z
    row['precession'] = raft.precession
    row['nutation'] = raft.nutation
    row['spin'] = raft.spin
    row['id'] = raft.id
neighbor_ids = []
for raft in rafts:
    neighbor_ids += ['-'.join(str(n.id) for n in raft.neighbors)]
t['neighbor_ids'] = neighbor_ids
t.pprint_all()
n_rafts = len(rafts)
n_robots = n_rafts*72
basename = f'{timestamp}_{focsurf_name}_{n_rafts}rafts_{n_robots}robots'
filename = basename + '.csv'
t.write(filename, overwrite=True)
print(f'Saved table to {os.path.abspath(filename)}')
for key in (k for k in t.colnames if 'gap' in k):
    statfuncs = {'min': min, 'max': max, 'median': np.median, 'mean': np.mean, 'rms': lambda a: np.sqrt(np.sum(np.power(a, 2))/len(a))}
    print(f'\nFor "{key}" column:')
    for name, func in statfuncs.items():
        print(f'  {name:>6} = {func(t[key]):.3f}')
    print('')

# plot rafts
max_rafts_to_plot = math.inf  # limit plot complexity, sometimes useful in debugging
fig = plt.figure(figsize=plt.figaspect(1)*2, dpi=200, tight_layout=True)
ax = fig.add_subplot(projection='3d', proj_type='ortho')
outlines = []
for i, raft in enumerate(rafts):
    if i >= max_rafts_to_plot:
        break
    f = np.transpose(raft.poly3d)
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

num_text = f'{n_rafts} rafts --> {n_robots} robots'
plt.title(f'{timestamp}\n{num_text}')

views = [(-114, 23), (-90, 90), (0, 0), (-90, 0), (-80, 52), (-61, 14)]
for i, view in enumerate(views):
    ax.azim = view[0]
    ax.elev = view[1]
    filename = f'{basename}_view{i}.png'
    plt.savefig(filename)
    print(f'Saved plot to {os.path.abspath(filename)}')

print(f'Completed in {time.perf_counter() - start_time:.1f} sec')
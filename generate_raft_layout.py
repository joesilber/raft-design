#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Generates a layout of triangular raft modules on a focal surface. Layout is saved
as a csv table, as well as illustrated with an approximately to-scale 3D plot.
Contact: Joe Silber, jhsilber@lbl.gov
'''

import time
start_time = time.perf_counter()

from datetime import datetime
import math
import numpy as np
from numpy.polynomial import Polynomial
from astropy.table import Table
from scipy.spatial.transform import Rotation  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
import scipy.interpolate as interpolate
from scipy import optimize
import matplotlib.pyplot as plt
import os
import argparse
import simple_logger

timestamp_fmt = '%Y%m%dT%H%M'
timestamp = datetime.now().astimezone().strftime(timestamp_fmt)
interp1d = lambda x, y: interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')

logdir = os.getcwd()
logname = f'{timestamp}_generate_raft_layout.log'
logpath = os.path.join(logdir, logname)
logger, _, _ = simple_logger.start_logger(logpath)

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
    'MM1536-cfg1-20210910':
        {'description': 'MegaMapper 1536 config 1, 2021-09-21',
        'file': 'MM1536-cfg1-20210910.csv',
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
parser.add_argument('-c', '--raft_chamfer', type=float, default=2.5, help='mm, chamfer at triangle tips')
parser.add_argument('-ic', '--instr_chamfer', type=float, default=8.5, help='mm, chamfer to instrumented area of raft')
parser.add_argument('-iw', '--instr_wall', type=float, default=0.3, help='mm, shield wall thickness to instrumented area of raft')
parser.add_argument('-w', '--wedge', type=float, default=60.0, help='deg, angle of wedge envelope, argue 360 for full circle')
parser.add_argument('-o', '--offset', type=str, default='hex', help='argue "hex" to do a 6-raft ring at the middle of the focal plate, or "tri" to center one raft triangle there')
parser.add_argument('-v', '--max_vignette_cases', type=int, default=5, help='maximum number of cases to plot, varying the vignette radius')
transform_template = {'id':-1, 'dx':0.0, 'dy':0.0, 'dspin':0.0}
transform_keymap = {'dx': 'x', 'dy': 'y', 'dspin': 'spin0'}
example_mult_transform_args = '-t "{\'id\':1, \'dx\':0.5}" -t "{\'id\':2, \'dx\':-1.7}"'
parser.add_argument('-t', '--transforms', action='append', help=f'specify custom transformations for specific rafts (in mm and deg), formatted like {transform_template}. The \'id\' key references a specific raft to be adjusted, which presumably you know from inspecting results a previous, otherwise-identical, run of this same code. To adjust multiple rafts, just repeat the command, like: {example_mult_transform_args}. Hint: you must enclose each dict in " at the command line, and use \' around keys')
userargs = parser.parse_args()
logger.info(f'User inputs: {userargs}')

# validate the custom transform input
import json
if not userargs.transforms:
    userargs.transforms = []
user_transforms = [json.loads(x.replace("'", '"')) for x in userargs.transforms]
simple_logger.assert2(all(isinstance(x, dict) for x in user_transforms), 'not all elements of transforms input are dicts, check that the input is a list of dicts as shown --help')
simple_logger.assert2(all(all(key in transform_template for key in x) for x in user_transforms), f'not all keys recognized in transforms input. valid keys are {transform_template.keys()}')
simple_logger.assert2(all(isinstance(x['id'], int) and x['id'] >= 0 for x in user_transforms), 'not all raft ids in transforms input are ints >= 0')
simple_logger.assert2(all(all(isinstance(val, (int, float)) for key, val in x.items() if key != 'id') for x in user_transforms), 'not all transform values are ints or floats')

# set up geometry functions
focsurf_name = focsurfs_index[userargs.focal_surface_number]
focsurf = focal_surfaces[focsurf_name]
logger.info(f'Focal surface name: {focsurf_name}')
logger.info(f'Focal surface parameters: {focsurf}')
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
        logger.warning('no chief ray deviation defined, letting CRD(R)=0')
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
nut = -(norm + crd[:-1])
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
logger.info(f'Best-fit sphere radius = {sphR:.3f} mm, is_convex = {is_convex}')

# basic raft outline
RB = userargs.raft_tri_base
RL = userargs.raft_length
RC = userargs.raft_chamfer
h1 = RB * 3**0.5 / 2  # height from base of triangle to opposite tip
h2 = RB / 3**0.5 / 2 # height from base of triangle to center
h3 = RB / 3**0.5  # height from center of triangle to tip
CB = RC * 2 / 3**0.5  # chamfer base length
for key, val in {'length (RL)': RL, 'triangle base (RB)': RB, 'triangle height (h1)': h1,
                 'triangle base to center (h2)': h2, 'triangle center to tip (h3)': h3,
                 'corner chamfer height (RC)': RC, 'corner chamfer base (CB)': CB}.items():
    logger.info(f'Raft geometry {key.upper()} = {val:.3f}')
raft_profile_x = [RB/2-CB,  RB/2-CB/2,    CB/2,    -CB/2,  -RB/2+CB/2,   -RB/2+CB]
raft_profile_y = [    -h2,      CB-h2,   h3-CB,    h3-CB,       CB-h2,        -h2]
raft_profile_z = [0.0]*len(raft_profile_x)
raft_profile = np.transpose([raft_profile_x, raft_profile_y, raft_profile_z])
logger.info(f'Raft profile polygon: {raft_profile.tolist()}')

# special function used for projecting from a coordinate like "S", but at rear
# of raft, to the corresponding point at the focal surface (in convex case)
r2 = r - RL * np.sin(np.radians(R2NUT(r)))
z2 = R2Z(r) - RL * np.cos(np.radians(R2NUT(r)))
dr2 = np.diff(r2)
dz2 = np.diff(z2)
dzdr2 = dz2 / dr2
ds2 = (1 + dzdr2**2)**0.5 * dr2
s2 = np.cumsum(ds2)
s2 = np.insert(s2, 0, 0.)
rearS_to_frontR = interp1d(s2, r)
rearR_to_frontR = interp1d(r2, r)
frontR_to_rearS = interp1d(r, s2)
frontR_to_rearR = interp1d(r, r2)

# single raft instrumented area
instr_base = RB - userargs.instr_wall * 2 * 3**0.5
instr_chamfer_adjusted_for_wall = userargs.instr_chamfer - 2 * userargs.instr_wall
instr_chamfer_base = instr_chamfer_adjusted_for_wall * 2 / 3**0.5
instr_chamfer_area = instr_chamfer_base**2 * 3**.5 / 4
instr_triangle_area = instr_base**2 * 3**.5 / 4
instr_area_per_raft = instr_triangle_area - 3 * instr_chamfer_area
logger.info(f'Instrumented area for a single raft = {instr_area_per_raft:.3f} mm^2')

# offset to average out the defocus of all the robots on a raft
above_below_equal_area_radius = (instr_area_per_raft/2 / math.pi)**0.5  # i.e. for a circle centered on raft that contains same area inside as in the rest of the raft
avg_focus_offset = above_below_equal_area_radius**2 / sphR / 2
avg_focus_offset *= -1 if is_convex else +1
logger.info(f'Focus offset (to average out the defocus of all the robots on a raft) = {avg_focus_offset:.4f} mm')

_raft_id_counter = 0
class Raft:
    '''Represents a single triangular raft.'''
    
    def __init__(self, x=0., y=0., spin0=0.):
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
        return float(R2Z(self.r)) + offset_correction

    @property
    def precession(self):
        '''angular position [deg] about the z-axis, same as precession'''
        return math.degrees(math.atan2(self.y, self.x))

    @property
    def nutation(self):
        '''angle [deg] w.r.t. z-axis (i.e. matches chief ray at center of raft)'''
        return float(R2NUT(self.r))
    
    @property
    def spin(self):
        '''rotation [deg] about raft's local z-axis, *including* compensation for
        precession (since raft orientation is defined by a 3-2-3 Euler rotation)'''
        return float(self.spin0 - self.precession)

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

    @property
    def max_front_vertex_radius(self):
        '''maximum distance from the z-axis of any point in the 3d raft polygon'''
        return max(self._front_vertex_radii)

    @property
    def min_front_vertex_radius(self):
        '''minimum distance from the z-axis of any point in the 3d raft polygon'''
        return min(self._front_vertex_radii)

    @property
    def _front_vertex_radii(self):
        '''return distances of all points in the 3d raft polygon from the z-axis'''
        all_points = np.transpose(self.front_poly)
        return np.hypot(all_points[0], all_points[1])

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
        polygons overlap, returns a default value of (0, np.array([1,0,0])). The input polygons should
        be Nx2 arrays of (x, y) or Nx3 arrays of (x, y, z) vertices. The returned unit vector points
        perpendicularly from the nearest poly2 segment toward the corresponding nearest poly1 point.'''
        if Raft.polygons_collide(poly1, poly2):
            return 0, np.array([1, 0, 0])
        test_pts = [np.array(pt) for pt in poly1]
        segment_pts = [(poly2[i], poly2[i+1]) for i in range(len(poly2) - 1)]
        segment_pts += [(poly2[-1], poly2[0])]  # close the polygon with last segment
        min_dist = math.inf
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
spacing_x = RB + userargs.raft_gap * math.sqrt(3)
spacing_y = spacing_x * math.sqrt(3)/2
if userargs.offset == 'hex':
    offset_x = spacing_x / 2
    offset_y = spacing_x / 3**0.5 / 2
elif userargs.offset == 'tri':
    offset_x = 0.0
    offset_y = 0.0
else:
    assert False, f'user argument "{userargs.offset}" for offset not recognized'
logger.info(f'Initial hex grid spacing_x = {spacing_x:7.3f}, offset_x = {offset_x:7.3f}')
logger.info(f'Initial hex grid spacing_y = {spacing_y:7.3f}, offset_y = {offset_y:7.3f}')
half_width_count = math.ceil(vigR / spacing_x) + 2
rng = range(-half_width_count, half_width_count + 1)
natural_grid = {'x': [], 'y': [], 'spin0': []}
for j in rng:
    x = [spacing_x*i + offset_x for i in rng]
    if j % 2:
        x = [u + spacing_x/2 for u in x]
    y = [spacing_y*j + offset_y]*len(x)

    # upward pointing triangles
    natural_grid['x'] += x
    natural_grid['y'] += y
    natural_grid['spin0'] += [0.]*len(x)

    # downward pointing triangles
    natural_grid['x'] += [u + spacing_x/2 for u in x]
    natural_grid['y'] += [v + spacing_y/3 for v in y]
    natural_grid['spin0'] += [180.]*len(x)
    
# flatten grid from its natural, implicit, curved space of focal surface
# to cartesian (where focal surface shape as function of radius applies)
q = np.arctan2(natural_grid['y'], natural_grid['x'])
s = np.hypot(natural_grid['x'], natural_grid['y'])
if is_convex:  # natural grid lies on the rear surface of rafts array
    r = rearS_to_frontR(s)
else: # natural grid lies on the front surface of rafts array
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
remove = raft_position_radii > vigR + spacing_x
if not_full_circle:
    raft_position_angles = np.degrees(np.arctan2(grid['y'], grid['x']))
    remove |= raft_position_angles > max(0, userargs.wedge)
    remove |= raft_position_angles < min(0, userargs.wedge)
keep = np.logical_not(remove)
for key in grid:
    grid[key] = np.array(grid[key])[keep]

# table structure for raft positions and orientations
t = Table(grid)
t['radius'] = np.hypot(t['x'], t['y'])
t.sort('radius')  # not important, just a trick to give the raft ids some sort of readability, when they are auto-generated below during raft instantiation
other_cols = {'z': float, 'precession': float, 'nutation': float, 'spin': float, 'id': int,
              'max_front_vertex_radius': float, 'min_front_vertex_radius': float}
for col, typecast in other_cols.items():
    t[col] = [typecast(0)]*len(t)

# generate raft instances
rafts = []
for row in t:
    raft = Raft(x=row['x'], y=row['y'], spin0=row['spin0'])
    rafts += [raft]
    row['id'] = raft.id
id_rafts = {raft.id: raft for raft in rafts}  # for lookup convenience

# apply custom transforms
user_transformed_rafts = set()
for transform in user_transforms:
    id = transform['id']
    simple_logger.assert2(id in id_rafts, f'no raft found with id {id}')
    raft = id_rafts[id]
    for key, delta in transform.items():
        if key in transform_keymap:
            adjust_key = transform_keymap[key]
            old = getattr(raft, adjust_key)
            new = old + delta
            setattr(raft, adjust_key, new)
            t[adjust_key][id] = new
            user_transformed_rafts.add(raft)
skip_interference_checks = any(user_transformed_rafts)
if skip_interference_checks:
    logger.warning('Turned off automatic raft interference checks, since user has input custom raft position shifts.')

# determine neighbors
neighbor_selection_radius = spacing_x / math.sqrt(3) * 1.1
for raft in rafts:
    if is_convex:
        r_others = np.array([other.r for other in rafts])  # "others" here will actually include self, but that's ok, will filter out later
        q_others = np.radians([other.precession for other in rafts])
        q_this = np.radians(raft.precession)
        r2_others = frontR_to_rearR(r_others)
        x2_others = r2_others * np.cos(q_others)
        y2_others = r2_others * np.sin(q_others)
        r2_this = frontR_to_rearR(raft.r)
        x2_this = r2_this * np.cos(q_this)
        y2_this = r2_this * np.sin(q_this)
        dist = np.hypot(x2_others - x2_this, y2_others - y2_this)
    else:
        dist = np.hypot(t['x'] - raft.x, t['y'] - raft.y)
    neighbor_selection = dist < neighbor_selection_radius
    neighbor_selection &= raft.id != t['id']  # skip self
    neighbor_selection_ids = np.flatnonzero(neighbor_selection)
    raft.neighbors = [r for r in rafts if r.id in neighbor_selection_ids]
neighbor_counts = {raft.id: len(raft.neighbors) for raft in rafts}
too_many_neighbors = {id: count for id, count in neighbor_counts.items() if count > 3}
if not skip_interference_checks:
    assert not(too_many_neighbors), f'non-physical number of neighbors detected. RAFT_ID:COUNT = {too_many_neighbors}'

# gap assessment
gap_mag_keys = ['min_gap_front', 'min_gap_rear', 'max_gap_front', 'max_gap_rear']
gap_vec_keys = ['min_gap_front_vec', 'min_gap_rear_vec', 'max_gap_front_vec', 'max_gap_rear_vec']
def calc_gaps(rafts, return_type='table'):
    '''Calculates nearest gaps to neighbors for all rafts in argued collection.
    Arg return_type may be 'dict' or 'table' (for astropy table).'''
    gaps = {}
    for key in ['id'] + gap_mag_keys + gap_vec_keys:
        gaps[key] = []
    if not isinstance(rafts, (list, set, tuple)):
        rafts = [rafts]
    for raft in rafts:
        mags_front, mags_rear = [], []
        vecs_front, vecs_rear = [], []
        for neighbor in raft.neighbors:
            mag_front, vec_front = raft.front_gap(neighbor)
            mag_rear, vec_rear = raft.rear_gap(neighbor)
            mags_front += [mag_front]
            mags_rear += [mag_rear]
            vecs_front += [vec_front]
            vecs_rear += [vec_rear]
        for name, func in {'min': np.argmin, 'max': np.argmax}.items():
            front_idx = func(mags_front)
            rear_idx = func(mags_rear)
            gaps[f'{name}_gap_front'] += [mags_front[front_idx]]
            gaps[f'{name}_gap_rear'] += [mags_rear[rear_idx]]
            gaps[f'{name}_gap_front_vec'] += [vecs_front[front_idx]]
            gaps[f'{name}_gap_rear_vec'] += [vecs_rear[rear_idx]]
        gaps['id'] += [raft.id]
    if return_type == 'dict':
        return gaps
    elif return_type == 'table':
        return Table(gaps)
    else:
        assert False, f'unrecognized return_type = {return_type}'

statfuncs = {'min': min, 'max': max, 'median': np.median, 'mean': np.mean, 'rms': lambda a: np.sqrt(np.sum(np.power(a, 2))/len(a))}
def print_stats(table, column_keys):
    for key in column_keys:
        s = f'For "{key}" column:'
        for name, func in statfuncs.items():
            s += f'\n  {name:>6} = {func(table[key]):.3f}'
        logger.info(s)

def calc_and_print_gaps(rafts, return_type='table'):
    '''verbose combination of gap calculation and printing stats'''
    gap_timer = time.perf_counter()
    gaps = calc_gaps(rafts, return_type=return_type)
    logger.info(f'Calculated gaps for {len(rafts)} rafts in {time.perf_counter() - gap_timer:.2f} sec.')
    print_stats(gaps, gap_mag_keys)
    return gaps

# global table of gaps between rafts
global_gaps = calc_and_print_gaps(rafts, return_type='table')
gap_minima = {k: min(global_gaps[k]) for k in gap_mag_keys}
if not skip_interference_checks:
    assert not(any(np.array(list(gap_minima.values())) <= 0)), 'Initial pattern already has interference between rafts. Check focal surface input geometry and/or consider increasing raft_gap value.'

# collate stats into table
global_gaps.sort('id')
t.sort('id')
for key in gap_mag_keys:
    t[key] = global_gaps[key]
for raft in rafts:
    row_idx = int(np.flatnonzero(t['id'] == raft.id))
    row = t[row_idx]
    row['x'] = raft.x
    row['y'] = raft.y
    row['radius'] = raft.r
    row['z'] = raft.z
    row['precession'] = raft.precession
    row['nutation'] = raft.nutation
    row['spin'] = raft.spin
    row['id'] = raft.id
    row['max_front_vertex_radius'] = raft.max_front_vertex_radius
    row['min_front_vertex_radius'] = raft.min_front_vertex_radius
neighbor_ids = []
for raft in rafts:
    neighbor_ids += ['; '.join(str(n.id) for n in raft.neighbors)]
t['neighbor_ids'] = neighbor_ids

# output tables and plots, varying radius of vignette circle within which to restrict array
overall_max_front_vertex_radius = t["max_front_vertex_radius"].max()
limit_radii = [vigR, vigR + h2, vigR + h3, vigR + RB, overall_max_front_vertex_radius]
limit_radii = sorted(limit_radii, reverse=True)  # start with largest and work inward
n_cases_to_output = min([len(limit_radii), userargs.max_vignette_cases])
print(n_cases_to_output)
limit_radii = limit_radii[:n_cases_to_output]
for limit_radius in limit_radii:
    logger.info(f'Exporting data and plots for layout with limit radius = {limit_radius:.3f}.')
    subselection = t['max_front_vertex_radius'] <= limit_radius
    t2 = t[subselection]
    rafts2 = [raft for raft in rafts if raft.id in t2['id']]
    n_rafts = len(rafts2)
    n_robots = n_rafts*72
    logger.info(f'Selected {n_rafts} rafts (containing {n_robots} robots) with all front vertices within limit radius.')
    t2_str = '\n' + '\n'.join(t2.pformat_all())
    logger.info(t2_str)

    # instrumented area calcs
    avg_spacing_x = RB + np.mean([t2['min_gap_front'].mean(), t2['max_gap_front'].mean()]) * math.sqrt(3)
    avg_consumed_area_per_raft = avg_spacing_x**2 * 3**.5 / 4
    logger.info(f'Avg area consumed on focal surface per raft = {avg_consumed_area_per_raft:.3f} mm^2')
    instr_area_efficiency = instr_area_per_raft / avg_consumed_area_per_raft
    logger.info(f'Instrumented area efficiency (local per raft) = {instr_area_efficiency * 100:.1f}%')
    total_instr_area = instr_area_per_raft * n_rafts
    logger.info(f'Total instrumented area (including outside vignette radius) = {total_instr_area:.1f} mm^2')
    surface_area_within_vigR = math.pi * R2S(vigR)**2 * userargs.wedge / 360
    logger.info(f'Surface area within vignette radius = {surface_area_within_vigR:.1f} mm^2')
    total_instr_area_ratio = total_instr_area / surface_area_within_vigR
    logger.info(f'Instrumented area ratio = (instrumented area) / (area within vignette) = {total_instr_area_ratio:.3f}')

    # file names and plot titles
    basename = f'{timestamp}_{focsurf_name}_raftlen{RL:.1f}_nomgap{userargs.raft_gap:.1f}_limitR{limit_radius:.1f}_nrafts{n_rafts}_nrobots{n_robots}'
    if limit_radii.index(limit_radius) == 0:
        basename0 = basename
    typtitle = f'Run: {timestamp}, FocalSurf: "{focsurf_name}", LimitRadius: {limit_radius:.1f} mm, RaftLength: {RL:.1f} mm' \
               f'\nNumRafts: {n_rafts}, NumRobots: {n_robots}' \
               f', MinGapFront: {t2["min_gap_front"].min():.2f} mm, MinGapRear: {t2["min_gap_rear"].min():.2f} mm' \
               f'\nPerRaftAreaEffic: {instr_area_efficiency*100:.1f}%, TotalInstrArea: {total_instr_area / 1e6:.3f} m^2' \
               f', InstrArea/UnvignArea: {total_instr_area_ratio:.3f}'
    filename = basename + '.csv'

    # save table
    t2.write(filename, overwrite=True)
    logger.info(f'Saved table to {os.path.abspath(filename)}')
    
    # print out more statistics
    print_stats(t2, gap_mag_keys + ['radius'])
    logger.info(f'Maximum radius of any front vertex (i.e. at the focal surface) in any raft polygon is'
                f' {t2["max_front_vertex_radius"].max():.3f} mm on raft {t2[t2["max_front_vertex_radius"].argmax()]["id"]}.')
    poly_exceeds_vigR = t2['id', 'max_front_vertex_radius'][t2['max_front_vertex_radius'] > vigR]
    poly_exceeds_vigR_tbl_str = '\n'.join(poly_exceeds_vigR.pformat_all())
    poly_exceeds_vigR_str = f'With limit radius {limit_radius:.3f} mm, '
    poly_exceeds_vigR_str += f'{len(poly_exceeds_vigR)} of {n_rafts} rafts have some' if poly_exceeds_vigR else 'no rafts have any'
    poly_exceeds_vigR_str += f' vertex at the focal surface outside the nominal vignette radius of {vigR:.3f} mm'
    poly_exceeds_vigR_str += f':\n{poly_exceeds_vigR_tbl_str}' if poly_exceeds_vigR else '.'
    logger.info(poly_exceeds_vigR_str)

    # plot rafts
    max_rafts_to_plot = math.inf  # limit plot complexity, sometimes useful in debugging
    fig = plt.figure(figsize=plt.figaspect(1)*2, dpi=200, tight_layout=True)
    ax = fig.add_subplot(projection='3d', proj_type='ortho')
    outlines = []
    for i, raft in enumerate(rafts2):
        if i >= max_rafts_to_plot:
            break
        f = np.transpose(raft.poly3d)
        ax.plot(f[0], f[1], f[2], '-', linewidth=0.7)

    # plot envelope
    ax.plot(envelope_x, envelope_y, envelope_z, 'k--', linewidth=1.0)

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

    plt.title(typtitle)
    views = [(-114, 23), (-90, 90), (0, 0), (-90, 0), (-80, 52), (-61, 14)]
    for i, view in enumerate(views):
        ax.azim = view[0]
        ax.elev = view[1]
        filename = f'{basename}_view{i}.png'
        filepath = os.path.join(logdir, filename)
        plt.savefig(filepath)
        logger.info(f'Saved 3D plot to {filepath}')

    # 2d raft plots
    plt.figure(figsize=(16, 8), dpi=200, tight_layout=True)
    for p, name in enumerate(['front', 'rear']):
        plt.subplot(1, 2, p + 1)
        for i, raft in enumerate(rafts2):
            if i >= max_rafts_to_plot:
                break
            f = np.transpose(eval(f'raft.{name}_poly'))
            f0 = np.append(f[0], f[0][0])
            f1 = np.append(f[1], f[1][0])
            plt.plot(f0, f1, '-', linewidth=0.7)
            plt.text(np.mean(f[0]), np.mean(f[1]), f'{raft.id:03}', family='monospace', fontsize=6,
                    verticalalignment='center', horizontalalignment='center')
        plt.plot(envelope_x, envelope_y, 'k--', linewidth=1.0,
                 label=f'vignette @ R{vigR:.1f}')
        plt.legend(loc='lower right')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.axis('equal')
        plt.title(f'raft {name} faces')
    plt.suptitle(typtitle)
    filename = f'{basename}_2D.png'
    filepath = os.path.join(logdir, filename)
    plt.savefig(filepath)
    logger.info(f'Saved 2D plot to {filepath}')

plt.close('all')
logger.info(f'Completed in {time.perf_counter() - start_time:.1f} sec')
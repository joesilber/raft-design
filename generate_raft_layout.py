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
from astropy.table import Table, vstack
import scipy.interpolate as interpolate
from scipy import optimize
import matplotlib.pyplot as plt
import os
import argparse
import simple_logger
from raft import Raft, RaftProfile

timestamp_fmt = '%Y%m%dT%H%M%S'
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
parser.add_argument('-r', '--limit_radius', type=float, default=0, help='maximum radius beyond which no part of raft instrumented area shall protrude, argue 0 to use the default for the given focal surface')
parser.add_argument('-b', '--raft_tri_base', type=float, default=80.0, help='mm, length of base edge of a raft triangle')
parser.add_argument('-l', '--raft_length', type=float, default=657.0, help='mm, length of raft from origin (at center fiber tip) to rear')
parser.add_argument('-g', '--raft_gap', type=float, default=3.0, help='mm, minimum gap between rafts')
parser.add_argument('-c', '--raft_chamfer', type=float, default=2.5, help='mm, chamfer at triangle tips')
parser.add_argument('-ic', '--instr_chamfer', type=float, default=8.5, help='mm, chamfer to instrumented area of raft')
parser.add_argument('-iw', '--instr_wall', type=float, default=0.3, help='mm, shield wall thickness to instrumented area of raft (argue 0 to have no wall)')
parser.add_argument('-w', '--wedge', type=float, default=60.0, help='deg, angle of wedge envelope, argue 360 for full circle')
parser.add_argument('-o', '--offset', type=str, default='hex', help='argue "hex" to do a 6-raft ring at the middle of the focal plate, or "tri" to center one raft triangle there')
parser.add_argument('-rp', '--robot_pitch', type=float, default=6.2, help='mm, center-to-center distance between robot centers within the raft')
parser.add_argument('-rr', '--robot_reach', type=float, default=3.6, help='mm, local to a robot, max patrol radius of fiber at full extension')
parser.add_argument('-re', '--robot_max_extent', type=float, default=4.4, help='mm, local to a robot, max radius of any mechanical part at full extension')
parser.add_argument('-igr', '--ignore_chief_ray_dev', action='store_true', help='ignore chief ray deviation in patterning')
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

# other input validations
assert userargs.instr_wall >= 0, f'negative thickness ({userargs.instr_wall} shield wall is an undefined case'

# set up geometry functions
focsurf_name = focsurfs_index[userargs.focal_surface_number]
focsurf = focal_surfaces[focsurf_name]
logger.info(f'Focal surface name: {focsurf_name}')
logger.info(f'Focal surface parameters: {focsurf}')
CRD2R_undefined = False
force_CRD_to_zero = userargs.ignore_chief_ray_dev
if all(label in focsurf for label in {'Z', 'CRD'}):
    R2Z = focsurf['Z']  # should be a function accepting numpy array argument for radius, returning z
    R2CRD = focsurf['CRD']  # should be a function accepting numpy array argument for radius, returning chief ray deviation
elif 'file' in focsurf:
    t = Table.read(focsurf['file'], comment='#')
    R2Z = interp1d(t['R'], t['Z'])
    if 'CRD' in t:
        R2CRD = interp1d(t['R'], t['CRD'])
    else:
        force_CRD_to_zero = True
else:
    assert False, 'unrecognized geometry input data'
if force_CRD_to_zero:
    R2CRD = Polynomial([0])  # in the absence of chief ray deviation information
    CRD2R_undefined = True
    logger.warning('no chief ray deviation defined, letting CRD(R)=0')
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
sphR_sign = -1 if is_convex else +1
logger.info(f'Best-fit sphere radius = {sphR:.3f} mm, is_convex = {is_convex}')

# single raft profile --> outer geometry
outer_profile = RaftProfile(tri_base=userargs.raft_tri_base,
                            length=userargs.raft_length,
                            chamfer=userargs.raft_chamfer,
                            )
for key, val in {'length (RL)': outer_profile.RL, 'triangle base (RB)': outer_profile.RB, 'triangle height (h1)': outer_profile.h1,
                'triangle base to center (h2)': outer_profile.h2, 'triangle center to tip (h3)': outer_profile.h3,
                'corner chamfer height (RC)': outer_profile.RC, 'corner chamfer base (CB)': outer_profile.CB}.items():
    logger.info(f'Raft geometry {key.upper()} = {val:.3f}')
logger.info(f'Raft outer profile polygon: {outer_profile.polygon2D}')

# special function used for projecting from a coordinate like "S", but at rear
# of raft, to the corresponding point at the focal surface (in convex case)
r2 = r - outer_profile.RL * np.sin(np.radians(R2NUT(r)))
z2 = R2Z(r) - outer_profile.RL * np.cos(np.radians(R2NUT(r)))
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

# generate grid of raft center points
# (based on two sets of staggered equilateral triangles)
spacing_x = outer_profile.RB + userargs.raft_gap * math.sqrt(3)
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

# instrumented area calca
has_shield_wall = userargs.instr_wall != 0
if has_shield_wall:
    instr_profile = RaftProfile(tri_base=outer_profile.RB - 2 * userargs.instr_wall * 3**0.5,
                                length=userargs.raft_length,
                                chamfer=userargs.instr_chamfer - 2 * userargs.instr_wall,
                                )
    instr_chamfer_area = instr_profile.CB**2 * 3**.5 / 4
    instr_triangle_area = instr_profile.RB**2 * 3**.5 / 4
    instr_area_per_raft = instr_triangle_area - 3 * instr_chamfer_area
    logger.info(f'Raft has shield wall of thickness = {userargs.instr_wall} mm')
    logger.info(f'Instrumented area for a single raft = {instr_area_per_raft:.3f} mm^2')
    logger.info(f'Raft\'s instrumented profile polygon: {instr_profile.polygon2D}')
else:
    # This is a rather special (and important) case. Without a mechanical shield
    # around the perimeter, the fibers can patrol outside the raft outline. Area
    # coverage calculation is more complicated than for an isolated single raft.
    # The area calculation is performed later in this script, *after* the robot
    # positions have been tabulated. Here we just set some nominal "instr_profile"
    # for congruity wiht the other case.
    logger.info(f'Raft has no shield wall.')
    instr_profile = outer_profile

# table structure for raft positions and orientations
t = Table(grid)
t['radius'] = np.hypot(t['x'], t['y'])
t.sort('radius')  # not important, just a trick to give the raft ids some sort of readability, when they are auto-generated below during raft instantiation
other_cols = {'z': float, 'precession': float, 'nutation': float, 'spin': float, 'id': int,
              'max_front_vertex_radius': float, 'min_front_vertex_radius': float,
              'max_instr_vertex_radius': float, 'min_instr_vertex_radius': float, }
for col, typecast in other_cols.items():
    t[col] = [typecast(0)]*len(t)

# generate raft instances
rafts = []
for row in t:
    raft = Raft(x=row['x'],
                y=row['y'],
                spin0=row['spin0'],
                outer_profile=outer_profile,
                instr_profile=instr_profile,
                r2nut=R2NUT,
                r2z=R2Z,
                sphR=sphR*sphR_sign,
                robot_pitch=userargs.robot_pitch,
                robot_max_extent=userargs.robot_max_extent,
                )
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
    row['max_front_vertex_radius'] = raft.max_front_vertex_radius(instr=False)
    row['min_front_vertex_radius'] = raft.min_front_vertex_radius(instr=False)
    row['max_instr_vertex_radius'] = raft.max_front_vertex_radius(instr=True)
    row['min_instr_vertex_radius'] = raft.min_front_vertex_radius(instr=True)
neighbor_ids = []
for raft in rafts:
    neighbor_ids += ['; '.join(str(n.id) for n in raft.neighbors)]
t['neighbor_ids'] = neighbor_ids

# output tables and plots
limit_radius = vigR if userargs.limit_radius <= 0 else userargs.limit_radius
logger.info(f'Exporting data and plots for layout with limit radius = {limit_radius:.3f}.')
subselection = t['max_instr_vertex_radius'] <= limit_radius
t2 = t[subselection]
rafts2 = [raft for raft in rafts if raft.id in t2['id']]
n_rafts = len(rafts2)
n_robots = n_rafts*75
logger.info(f'Selected {n_rafts} rafts (containing {n_robots} robots) with all front vertices within limit radius.')
t2_str = '\n' + '\n'.join(t2.pformat_all())
logger.info(t2_str)

# table of individual robot center positions
robot_table_headers = ['global robot idx', 'raft idx', 'local robot idx', 'r', 'x', 'y', 'z', 'precession', 'nutation', 'spin', 'intersects perimeter']
raft_robot_tables = []
for raft in rafts2:
    these_robots = raft.generate_robots_table(global_coords=True)
    these_robots.rename_column('idx', 'local robot idx')
    these_robots['raft idx'] = raft.id
    raft_robot_tables += [these_robots]
robots = vstack(raft_robot_tables)
robots['global robot idx'] = np.arange(len(robots))
robots = robots[robot_table_headers]
logger.info(f'Generated table of {len(robots)} individual robot positions.')
ideal_asphere_z = R2Z(robots['r'])
ideal_asphere_nut = R2NUT(robots['r'])
robots['chief ray error'] = robots['nutation'] - ideal_asphere_nut
robots['z error'] = robots['z'] - ideal_asphere_z
for key, unit in {'z error': 'mm', 'chief ray error': 'deg'}.items():
    argmax = robots[key].argmax()
    argmin = robots[key].argmin()
    prefix = 'robot centers -->'
    logger.info(f'{prefix} max {key} = {robots[key][argmax]:.3f} {unit}, occurring at robot {robots["global robot idx"][argmax]}, at radius {robots["r"][argmax]:.3f} mm')
    logger.info(f'{prefix} min {key} = {robots[key][argmin]:.3f} {unit}, occurring at robot {robots["global robot idx"][argmin]}, at radius {robots["r"][argmin]:.3f} mm')
    logger.info(f'{prefix} mean {key} = {robots[key].mean():.3f} {unit}')
    logger.info(f'{prefix} median {key} = {np.median(robots[key]):.3f} {unit}')
    logger.info(f'{prefix} rms {key} = {(np.sum(robots[key]**2)/len(robots))**0.5:.3f} {unit}')

# instrumented area calcs for the NO SHIELD case
# (shield case was already calculated above, prior to raft objects instantiation)
if not has_shield_wall:
    # calculation below is based on script I had made in Feb 2020 for DESI-5513-v1

    # collect positioner centers in flatxy coordinates
    bots_s = R2S(robots['r'])
    bots_q = np.arctan2(robots['y'], robots['x'])
    bots_flatx = bots_s * np.cos(bots_q)
    bots_flaty = bots_s * np.sin(bots_q)
    centers = np.transpose([bots_flatx, bots_flaty]).tolist()

    # set up area evaluation grid
    areagrid_spacing = 0.3 # mm, edge length of grid unit
    areagrid_limit = limit_radius
    areagrid_max_mm = areagrid_limit
    areagrid_min_mm = -areagrid_limit
    areagrid_mm = np.linspace(areagrid_min_mm, areagrid_max_mm, math.ceil((areagrid_max_mm - areagrid_min_mm)/areagrid_spacing) + 1).tolist()
    areagrid = [[0]*len(areagrid_mm) for x in range(len(areagrid_mm))]
    def get_grid_idx(continuous_value_mm):
        translated = continuous_value_mm - areagrid_min_mm
        scaled = translated / areagrid_spacing
        rounded = int(round(scaled))
        return rounded

    # fill the grid with coverage
    rmax = userargs.robot_reach
    ndim = range(2)
    for center in centers:
        center_idx = [get_grid_idx(center[i]) for i in ndim]
        box_min_idx = [get_grid_idx(center[i] - rmax) for i in ndim]
        box_max_idx = [get_grid_idx(center[i] + rmax) for i in ndim]
        box_idxs = [[j for j in range(box_min_idx[i], box_max_idx[i] + 1)] for i in ndim]
        for i in box_idxs[0]:
            for j in box_idxs[1]:
                xy_mm = [areagrid_mm[i], areagrid_mm[j]]
                test_r = sum((xy_mm[k] - center[k])**2 for k in ndim)**0.5
                if test_r <= rmax:
                    areagrid[i][j] += 1

    # calculate total covered area by at least one fiber
    total_grids_covered = np.sum(np.where(areagrid, 1, 0))
    total_area_covered_at_least_once_mm = total_grids_covered * areagrid_spacing**2
    total_area_covered_at_least_once_m = total_area_covered_at_least_once_mm * 0.001**2
    total_grids_patrolled = np.sum(areagrid) # i.e. including overlap
    total_grids_patrolled_mm = total_grids_patrolled * areagrid_spacing**2
    total_grids_patrolled_m = total_grids_patrolled_mm * 0.001**2

avg_spacing_x = outer_profile.RB + np.mean([t2['min_gap_front'].mean(), t2['max_gap_front'].mean()]) * math.sqrt(3)
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
overall_max_instr_vertex_radius = t2["max_instr_vertex_radius"].max()
basename = f'{timestamp}_{focsurf_name}_raftlen{outer_profile.RL:.1f}_nomgap{userargs.raft_gap:.1f}_maxR{overall_max_instr_vertex_radius:.1f}_nrafts{n_rafts}_nrobots{n_robots}'
typtitle = f'Run: {timestamp}, FocalSurf: "{focsurf_name}", RaftLength: {outer_profile.RL:.1f} mm' \
           f'\nNumRafts: {n_rafts}, NumRobots: {n_robots}' \
           f', MinGapFront: {t2["min_gap_front"].min():.2f} mm, MinGapRear: {t2["min_gap_rear"].min():.2f} mm' \
           f'\nMaxMechanicalVertexRadius: {t2["max_front_vertex_radius"].max():.2f} mm'\
           f', MaxInstrumentedVertexRadius: {overall_max_instr_vertex_radius:.2f} mm' \
           f'\nPerRaftAreaEffic: {instr_area_efficiency*100:.1f}%, TotalInstrArea: {total_instr_area / 1e6:.3f} m^2' \
           f', InstrArea/UnvignArea: {total_instr_area_ratio:.3f}'
rafts_filename = f'{basename}_raftdata.csv'
robots_filename = f'{basename}_robotdata.csv'

# save tables
t2.write(rafts_filename, overwrite=True)
logger.info(f'Saved rafts data table to {os.path.abspath(rafts_filename)}')
robots.write(robots_filename, overwrite=True)
logger.info(f'Saved robots data table to {os.path.abspath(robots_filename)}')

# print out more statistics
print_stats(t2, gap_mag_keys + ['radius'])
for key, desc in {'front': 'raft outline', 'instr': 'instrumented area'}.items():
    maxval = t2[f'max_{key}_vertex_radius'].max()
    raftid = t2[t2[f'max_{key}_vertex_radius'].argmax()]['id']
    logger.info(f'Maximum radius of any {key} vertex (i.e. at the focal surface) in any'
                f' {desc} polygon is {maxval:.3f} mm on raft {raftid}.')
poly_exceeds_vigR = t2['id', 'max_instr_vertex_radius'][t2['max_instr_vertex_radius'] > vigR]
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
        poly_cmd = f'raft.{name}_poly'
        cmd_suffix = '(instr=False)' if name == 'front' else ''
        f = np.transpose(eval(poly_cmd + cmd_suffix))
        f0 = np.append(f[0], f[0][0])
        f1 = np.append(f[1], f[1][0])
        if name == 'front':
            cmd_suffix = '(instr=True)'
            fi = np.transpose(eval(poly_cmd + cmd_suffix))
            f0 = np.append(f0, fi[0])
            f1 = np.append(f1, fi[1])
            f0 = np.append(f0, fi[0][0])
            f1 = np.append(f1, fi[1][0])
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
logger.info(f'Saved 2D rafts plot to {filepath}')

# robots plot
plt.figure(figsize=(16, 8), dpi=200, tight_layout=True)
interiors = robots[np.logical_not(robots['intersects perimeter'])]
perimeters = robots[robots['intersects perimeter']]
plt.subplot(2, 2, (1, 3))
plt.plot(interiors['x'], interiors['y'], 'b+', label='interior', markersize=9, fillstyle='none')
plt.plot(perimeters['x'], perimeters['y'], 'k+', label='perimeter', markersize=9, fillstyle='none')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.legend(loc='upper left')
plt.axis('equal')
for subplot, abscissa in {2: 'x', 4: 'y'}.items():
    plt.subplot(2, 2, subplot)
    plt.plot(interiors[abscissa], interiors['z'], 'b+', markersize=4)
    plt.plot(perimeters[abscissa], perimeters['z'], 'k+', markersize=4)
    plt.xlabel(f'{abscissa} (mm)')
    plt.ylabel('z (mm)')
title = typtitle + f'\nNumRaftPerimeterRobots: {len(perimeters)}'
plt.suptitle(title)
filename = f'{basename}_robots.png'
filepath = os.path.join(logdir, filename)
plt.savefig(filepath)
logger.info(f'Saved robots plot to {filepath}')

# area coverage plot (no shield case)
if not has_shield_wall:
    plt.figure(figsize=(16, 8), dpi=200, tight_layout=True)
    ax = plt.contourf(areagrid, levels=np.max(areagrid))
    plt.axis('square')
    plt.xlabel('')
    plt.axis('off')
    plt.title(f'Single and double coverage of focal plane, for the NO SHIELD WALL case.' +
              f'\nTotal area covered by at least one fiber = {total_area_covered_at_least_once_m:.3f} m^2' +
              f'\nTotal patrolled area (i.e. counting overlap) = {total_grids_patrolled_m:.3f} m^2')
    filename = f'{basename}_noshieldcoverage.png'
    filepath = os.path.join(logdir, filename)
    plt.savefig(filepath)
    logger.info(f'Saved no-shield area coverage plot to {filepath}')

plt.close('all')
logger.info(f'Completed in {time.perf_counter() - start_time:.1f} sec')
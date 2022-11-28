#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Produces a table of individual fiber robot positions from a table of raft positions.
Contact: Joe Silber, jhsilber@lbl.gov
'''

import time
start_time = time.perf_counter()
from datetime import datetime
import os
import argparse
import math
import simple_logger
timestamp_fmt = '%Y%m%dT%H%M%S'
timestamp = datetime.now().astimezone().strftime(timestamp_fmt)
logdir = os.getcwd()
logname = f'{timestamp}_make_robots_table.log'
logpath = os.path.join(logdir, logname)
logger, _, _ = simple_logger.start_logger(logpath)

# command line argument parsing
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-t', '--table_path', type=str, required=True,
    help='Path to input table (csv) of raft positions. Table should be like that produced ' \
         'by generate_raft_layout.py, in particular it must include columns "x", "y", "z", ' \
         '"precession", "nutation", and "spin". All units mm and degrees.',
    )
parser.add_argument('-r', '--spherical_radius', type=float, default=math.inf,
    help='Best-fit spherical radius of focal surface. Locally within the raft, robot centers ' \
         '(i.e. fiber-tip position when a robot is centered) will be placed on a sphere of this radius. '\
         'The sphere is assumed to have its center on the line (x, y, z) = (0, 0, z), ' \
         'and to include the point (x, y, z) = (0, 0, 0) on its surface. ' \
         'Fiber axes are kept parallel to raft z-axis. Argument should be a magnitude. '\
         'Skip this argument for case of flat focal plane.',
    )
parser.add_argument('--is_convex', action='store_true',
    help='Focal surface is assumed concave, unless this argument.',
    )
userargs = parser.parse_args()
logger.info(f'User inputs: {userargs}')

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

# read rafts table
basename = os.path.basename(userargs.table_path)
rafts = Table.read(userargs.table_path)
if 'id' not in rafts.columns:
    rafts['id'] = [i for i in range(len(rafts))]
required_columns = ['x', 'y', 'z', 'precession', 'nutation', 'spin']
missing_columns = set(required_columns) - set(rafts.columns)
simple_logger.assert2(not(any(missing_columns)), f'rafts input table is missing columns: {missing_columns}')
logger.info(f'Raft positions input table: {basename}\n' + '\n'.join(rafts[['id'] + required_columns].pformat_all()))

## REPLACE STUFF HERE WITH WHAT IS NOW IN RAFT CLASS
## AND CONSIDER DOING ALL THIS DIRECTLY IN GENERATE RAFT LAYOUT MODULE -- LESS DUPLICATIVE SETUP WORK!

# square-ish local robot pattern
n_robots_per_raft = 75
robot_pitch = 6.2
logger.info(f'Number of robots per raft = {n_robots_per_raft}')
logger.info(f'Center-to-center pitch between robots = {robot_pitch} mm')
overwidth_count = math.ceil(n_robots_per_raft**0.5)
pattern_row_x = robot_pitch * np.arange(-overwidth_count, overwidth_count)
pattern_row_y = np.zeros(np.shape(pattern_row_x))
offset_x = 0.  # overall offset of the pattern
offset_y = -robot_pitch / 3**0.5  # overall offset of the pattern
step_x = robot_pitch / 2  # row-to-row x pitch
step_y = robot_pitch * 3**0.5 / 2  # row-to-row y pitch
pattern = np.array([[],[]])
for i in range(-overwidth_count, overwidth_count):
    new_row_x = pattern_row_x + offset_x + step_x * (i % 2)
    new_row_y = pattern_row_y + offset_y + step_y * i
    pattern = np.append(pattern, [new_row_x, new_row_y], axis=1)

# crop to actual raft triangle
base_to_ctr = 22.594
corner_to_ctr = 37.588
logger.info(f'Distance from base limit of raft to center = {base_to_ctr} mm')
logger.info(f'Distance from corner limit of raft to center = {corner_to_ctr} mm')
pattern_limits = {
     90: base_to_ctr, # angle of rotation, x-limit at that angle
    210: base_to_ctr,
    330: base_to_ctr,
     30: corner_to_ctr,
    150: corner_to_ctr,
    270: corner_to_ctr,
    }
exclude = set()
for angle, x_limit in pattern_limits.items():
    cos = np.cos(np.radians(angle))
    sin = np.sin(np.radians(angle))
    rotation = np.array([[cos, -sin], [sin, cos]])
    rotated = np.matmul(rotation, pattern)
    new_exclusions = np.argwhere(rotated[0] > x_limit).transpose()[0]
    exclude |= set(new_exclusions)
pattern = np.delete(pattern, list(exclude), axis=1)
n_robots_in_pattern = len(pattern[0]) 
simple_logger.assert2(n_robots_in_pattern == n_robots_per_raft,
    f'pattern has {n_robots_in_pattern} robots, unequal to presumed number ({n_robots_per_raft})')

# decide best placement of focus sphere
# assume here that the raft position is defined such that

# set focus of robots on sphere


plt.plot(pattern[0], pattern[1], 'bo')
plt.axis('equal')
plt.show()

pass

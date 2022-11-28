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
                    help='path to input table (csv) of raft positions. Table should be like that produced ' \
                         'by generate_raft_layout.py, in particular it must include columns "x", "y", "z", ' \
                         '"precession", "nutation", and "spin". All units mm and degrees.',
                         )
parser.add_argument('-r', '--spherical_radius', type=float, default=math.inf,
                    help='best-fit spherical radius of focal surface. Locally within the raft, robot centers ' \
                         '(i.e. fiber-tip position when a robot is centered) will be placed on a sphere of this ' \
                         'radius. The sphere is positioned midway at the average focus position for the raft. ' \
                         'Fiber axes are kept parallel to raft z-axis. Argument should be a magnitude. Skip for ' \
                         'case of flat focal plane.',
                         )
parser.add_argument('--is_convex', action='store_true',
                    help='focal surface is assumed concave, unless this argument'
                         )
userargs = parser.parse_args()
logger.info(f'User inputs: {userargs}')

import numpy
from astropy.table import Table

# read rafts table
basename = os.path.basename(userargs.table_path)
rafts = Table.read(userargs.table_path)
if 'id' not in rafts.columns:
    rafts['id'] = [i for i in range(len(rafts))]
required_columns = ['x', 'y', 'z', 'precession', 'nutation', 'spin']
missing_columns = set(required_columns) - set(rafts.columns)
simple_logger.assert2(not(any(missing_columns)), f'rafts input table is missing columns: {missing_columns}')
logger.info(f'Raft positions input table: {basename}\n' + '\n'.join(rafts[['id'] + required_columns].pformat_all()))

# robot pattern local to raft
n_robots_per_raft = 75
robot_pitch = 6.2  # mm, robot pitch
base_to_ctr = 22.594  # mm, distance from base limit of raft to center
corner_to_ctr = 37.588  # mm, distance from corner limit of raft to center
pattern_limits = { 90: base_to_ctr, # angle of rotation, x-limit at that angle
                  210: base_to_ctr,
                  330: base_to_ctr,
                   30: corner_to_ctr,
                  150: corner_to_ctr,
                  270: corner_to_ctr,
                 }
logger.info(f'Number of robots per raft = {n_robots_per_raft}')
logger.info(f'Center-to-center pitch between robots = {robot_pitch} mm')
logger.info(f'max robot center distance from raft center = {max_robot_dist:.3f} mm')


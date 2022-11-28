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
parser.add_argument('-c', '--is_convex', action='store_true',
                    help='focal surface is assumed concave, unless this argument'
                         )
userargs = parser.parse_args()
logger.info(f'User inputs: {userargs}')

import numpy

# robot pattern local to raft
n_robots_per_raft = 75
robot_pitch = 6.2  # mm
logger.info(f'Number of robots per raft = {n_robots_per_raft}')
logger.info(f'Center-to-center pitch between robots = {robot_pitch} mm')

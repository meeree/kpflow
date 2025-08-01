# This script runs a sweep of inital scale, g, values.

import argparse
import train 
import numpy as np
from itertools import product
from tqdm import tqdm

import sys
sys.path.append('../../')
from analysis_utils import ping_dir

# INHERTIS commandline arguments from train.py in addition to the ones below.
parser = argparse.ArgumentParser(description='Training Sweep')
parser.add_argument('--scale_min', default=1e-5, type = float)
parser.add_argument('--scale_max', default=5., type = float)
parser.add_argument('--nscales', default=10, type = int)
parser.add_argument('--nrepeats', default=1, type = int)
parser.add_argument('--linear_scale', action='store_true', help='whether to use a linear sweep scaling. Defaults to exponential scale if not specified.')

args = train.parse_arguments(parser) # Append command line arguments for main, so user can also specify those. 

if args.linear_scale:
    scales = np.linspace(args.scale_min, args.scale_max, args.nscales)
else:
    mn, mx = np.log10(args.scale_min), np.log10(args.scale_max)
    scales = 10.**np.linspace(mn, mx, args.nscales)

root_savename = str(args.save_dir)
if root_savename == '':
    raise 'You need to specify a directory to save the sweep in.'

ping_dir(root_savename)

with tqdm(total=args.nscales*args.nrepeats) as pbar:
    for repeat, scale in product(range(args.nrepeats), scales):
        pbar.set_description("*********************************               SWEEP PROGRESS")
        ping_dir(root_savename + f'/init_{scale}/')
        args.save_dir = root_savename + f'/init_{scale}/repeat_{repeat}/'
        args.init_level = scale
        train.main(args)
        pbar.update(1)

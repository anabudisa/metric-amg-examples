"""
\file based on solvers of xD-1D problems from HAZMATH library.
Call of HAZniCS solver for the 3d-1d coupled problem.
This instance solves EMI equations, i.e. systems assembled using file emi_3d1d.py

Usage:
    python3 run_solver_3d1d.py -infile INPUTFILE -indir INPUTDIR -outdir OUTPUTDIR

where INPUTFILE is the path to a ".dat" file that sets solver parameters, INPUTDIR is the path to directory
where assembled matrices and rhs are saved, and OUTPUTDIR is the path to directory where solution (.txt) will be saved.
"""

import argparse, os
import utils
import haznics

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-infile', action="store", type=str, default="./src/input_metric.dat", help="Solver input file")
parser.add_argument('-indir', action="store", type=str, default="./data/emi_3d1d/",
                    help="Directory with matrices (.npy)")
parser.add_argument('-outdir', action="store", type=str, default="./results/emi_3d1d/",
                    help="Directory to output solution")
args, _ = parser.parse_known_args()

utils.print_red("Path to solver input file: " + args.infile)
utils.print_red("Path to matrices directory: " + args.indir)
utils.print_red("Path to output directory: " + args.outdir)

assert os.path.exists(args.infile)
assert os.path.exists(args.indir)
not os.path.exists(args.outdir) and os.makedirs(args.outdir)

sfile = os.path.abspath(args.infile)
mdir = os.path.abspath(args.indir)+'/'
odir = os.path.abspath(args.outdir)+'/'

# Call solver
haznics.fenics_metric_solver_xd_1d(sfile, mdir, odir)

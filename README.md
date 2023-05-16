# metric-amg-examples

Code for the numerical examples presented in "Algebraic multigrid methods for metric-perturbed coupled problems" by A. Budisa, X. Hu, M. Kuchta, K.-A. Mardal and L. T. Zikatanov (2023).

### Problems in $2d$
These things are mostly done on `UnitSquareMesh` geometries to check correctness
- Bidomain model: `bidomain_2d.py`, `bidomain_2d_firedrake.py`
- EMI model: `emi_2d.py`

### Problems in $3d$
These things are mostly done on `UnitCubeMesh` geometries to check correctness
- Bidomain model: `bidomain_3d.py`
- EMI model: `emi_3d.py`

### Others
- Reduced $3d$ - $1d$ EMI model on a cube with an embedded [neuron](https://neuromorpho.org/neuron_info.jsp?neuron_name=PolyIC_3AS2_1): `emi_3d1d.py`

## Requirements
- Install **HAZniCS** software that includes [FEniCS](https://fenicsproject.org/download/archive/) (legacy), [FEniCS_ii](https://github.com/MiroK/fenics_ii), [cbc.block](https://bitbucket.org/fenics-apps/cbc.block/src/master/) and [HAZmath](https://github.com/HAZmathTeam/hazmath): 
  - Check out this [README](https://github.com/HAZmathTeam/hazmath/blob/main/examples/haznics/README.md) file at HAZmath repo for installation instructions
- Additional Python packages (recommended versions): `numpy (v1.21.5)`, `scipy (v1.8.0)`, `sympy (v1.9)`, `tabulate (v0.8.10)`, `networkx (v2.8.4)`, [gmshnics](https://github.com/MiroK/gmshnics)
- For the geometric MG, install [Firedrake](https://www.firedrakeproject.org/download.html)
- For the 3D-1D problem, download required mesh files by executing `bash downloads.sh`

### How to run demo examples
e.g. AMG for the 2D bidomain problem
- with default parameters
```
python3 ./src/bidomain_2d.py
```
- with specifying parameters (**nrefs** = number of refinements, **gamma** = coupling parameter, **precond** = type of preconditioner)
```
python3 ./src/bidomain_2d.py -nrefs 5 -gamma 1e6 -precond metric_mono
```
where `metric_mono` is the tag for the monolithic Metric AMG preconditioner implemented in HAZniCS.

The results in the paper can be obtained by executing bash scripts `run_FILENAME.sh` where `FILENAME` is the name of the Python script you want to run, e.g.
```
bash run_bidomain_2d.sh
```
The results will be saved in folder `./results/FILENAME/` in form of a `.txt` file. 
## Cite
If you are using our numerical methods or their implementation in your work, please cite our [arXiv preprint](https://arxiv.org/abs/2305.06073).

## Note
This repository has been forked from the [original repo](https://github.com/MiroK/metric-amg-examples), where you can find other examples of problems that can be solved by the multilevel method (namely, Metric AMG) described in our paper.


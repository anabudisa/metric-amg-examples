#
# Solve on one mesh
#
import xii.assembler.average_matrix
from dolfin import *
from xii import *
import sympy as sp
import ulfy


def tangent(f, n):
    return f - dot(f, n)*n


def setup_mms(params):
    '''
    We solve

    -div(kappa_i*grad u_i) = f_i in Omega_i 

    with coupling conditions and mixed boundary conditions
    '''
    mesh = UnitCubeMesh(1, 1, 1)  # Dummy

    kappa1, kappa2, gamma = Constant(1), Constant(2), Constant(3)

    x, y, z = SpatialCoordinate(mesh)
    u1 = x + y + 2*z
    u2 = x - y + z

    sigma1 = kappa1*grad(u1)
    sigma2 = kappa2*grad(u2)

    f1 = -div(sigma1) 
    f2 = -div(sigma2) 

    kappa10, kappa20, gamma0 = sp.symbols('kappa1, kappa2, gamma')
    subs = {kappa1: kappa10, kappa2: kappa20, gamma: gamma0}

    as_expression = lambda f: ulfy.Expression(f,
                                              subs=subs,
                                              degree=4,
                                              kappa1=params.kappa1,
                                              kappa2=params.kappa2,
                                              gamma=params.gamma)

    data = {
        'u1': as_expression(u1),
        'flux1': as_expression(sigma1),
        'f1': as_expression(f1),
        # --
        'u2': as_expression(u2),
        'flux2': as_expression(sigma2),
        'f2': as_expression(f2),
    }
    return data


def get_system(mesh3d, mesh1d, radius, data, pdegree, parameters):
    """Setup the linear system A*x = b in W where W has bcs"""
    kappa1, kappa2, gamma = (Constant(parameters.kappa1),
                             Constant(parameters.kappa2),
                             Constant(parameters.gamma))

    V1 = FunctionSpace(mesh3d, 'Lagrange', pdegree)
    V2 = FunctionSpace(mesh1d, 'Lagrange', pdegree)
    W = [V1, V2]

    u1, u2 = map(TrialFunction, W)
    v1, v2 = map(TestFunction, W)

    # ---

    cylinder = Circle(radius=radius, degree=20)
    Ru1, Rv1 = Average(u1, mesh1d, cylinder), Average(v1, mesh1d, cylinder)
    
    dx_ = Measure('dx', domain=mesh1d)
    tau = TangentCurve(mesh1d)

    a = block_form(W, 2)
    a[0][0] = inner(kappa1*grad(u1), grad(v1))*dx + gamma*inner(Ru1, Rv1)*dx_
    a[0][1] = -gamma*inner(u2, Rv1)*dx_
    a[1][0] = -gamma*inner(Ru1, v2)*dx_
    a[1][1] = inner(kappa2*dot(grad(u2), tau), dot(grad(v2), tau))*dx_ + gamma*inner(u2, v2)*dx_

    f1, f2 = data['f1'], data['f2']

    L = block_form(W, 1)
    L[0] = inner(Constant(0), v1)*dx
    L[1] = inner(Constant(0), v2)*dx_

    V1_bcs = []
    u2_data = data['u2']
    V2_bcs = [DirichletBC(V2, u2_data, 'on_boundary')]
    bcs = [V1_bcs, V2_bcs]

    A, b = map(ii_assemble, (a, L))
    A, b = apply_bc(A, b, bcs)

    return A, b, W, bcs

# --------------------------------------------------------------------

if __name__ == '__main__':
    from block.iterative import ConjGrad
    from collections import namedtuple
    import argparse, time, tabulate
    import numpy as np
    import os, utils

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Material properties
    parser.add_argument('-kappa1', type=float, default=2, help='Diffusion in 1')
    parser.add_argument('-kappa2', type=float, default=3, help='Diffusion in 2')
    parser.add_argument('-gamma', type=float, default=5, help='Coupling strength')            
    # Discretization
    parser.add_argument('-pdegree', type=int, default=1, help='Polynomial degree in Pk discretization')
    # Solver
    parser.add_argument('-precond', type=str, default='diag',
                        choices=('diag', 'hypre', 'hazmath', 'metric', 'metric_mono', 'metric_hazmath'))

    parser.add_argument('-save', type=int, default=0, help='Save graphics')
    parser.add_argument('-dump', type=int, default=0, help='Save matrices and vectors to .npy format')

    args, _ = parser.parse_known_args()
    
    result_dir = f'./results/emi3d1d_brain/'
    not os.path.exists(result_dir) and os.makedirs(result_dir)

    def get_path(what, ext):
        template_path = f'{what}_precond{args.precond}_kappa1{args.kappa1}_kappa2{args.kappa2}_gamma{args.gamma}_pdegree{args.pdegree}.{ext}'
        return os.path.join(result_dir, template_path)

    Params = namedtuple('Params', ('kappa1', 'kappa2', 'gamma'))
    params = Params(args.kappa1, args.kappa2, args.gamma)
    utils.print_red(str(params))
    
    # Setup MMS
    test_case = setup_mms(params)

    # Setup problem geometry and discretization
    pdegree = args.pdegree

    # What we want to collect for KSP
    headers_ksp = ['ndofs', 'niters', 'cond', 'timeKSP', 'r', 'h']
    table_ksp = []

    get_precond = {'diag': utils.get_block_diag_precond,
                   'hypre': utils.get_hypre_monolithic_precond,
                   'hazmath': utils.get_hazmath_amg_precond,  # solve w cbc CG + R.T*hazmathAMG*R preconditioner
                   'metric': utils.get_hazmath_metric_precond,  # solve w cbc CG + R.T*metricAMG*R preconditioner
                   'metric_mono': utils.get_hazmath_metric_precond_mono,  # solve w cbc CG + metricAMG on Amonolithic
                   'metric_hazmath': None}[args.precond]  # solve w hazmath CG + hazmath metricAMG
    # Meshes
    mesh3d = Mesh()
    with HDF5File(mesh3d.mpi_comm(), './data/brava/darcy_mesh.h5', 'r') as h5:
        h5.read(mesh3d, '/mesh', False)

    mesh1d = Mesh()
    with HDF5File(mesh1d.mpi_comm(), './data/brava/BG001_data.h5', 'r') as h5:
        h5.read(mesh1d, '/mesh', False)

    mesh1d_radii = MeshFunction('double', mesh1d, 1, 0)
    with HDF5File(mesh1d.mpi_comm(), './data/brava/BG001_data.h5', 'r') as h5:
        h5.read(mesh1d_radii, '/radii_cell_foo')

    # Get radius info
    P0 = FunctionSpace(mesh1d, 'DG', 0)
    radii = Function(P0)
    radii.vector().set_local(mesh1d_radii.array())
    
    AA, bb, W, bcs = get_system(mesh3d, mesh1d, radii,
                                data=test_case, pdegree=pdegree, parameters=params)

    # cbk = lambda k, x, r, b=bb, A=AA: print(f'\titer{k} -> {[(b[i]-xi).norm("l2") for i, xi in enumerate(A*x)]}')

    # Write system to .npy files
    if args.dump and W[0].dim() + W[1].dim() > 1e6:
        utils.dump_system(AA, bb, W)
        exit(0)

    # mono or block
    if args.precond in {"metric_mono", "metric_hazmath"}:
        AA_ = ii_convert(AA)
        bb_ = ii_convert(bb)
        cbk = lambda k, x, r, b=bb_, A=AA_: print(f'\titer{k} -> {[(b - A * x).norm("l2")]}')
    else:
        cbk = lambda k, x, r, b=bb, A=AA: print(f'\titer{k} -> {[(b - A * x).norm("l2")]}')

    then = time.time()
    # For simplicity only use block diagonal preconditioner
    then = time.time()
    if args.precond == "metric_hazmath":
        # this one solves everything in hazmath
        niters, wh, ksp_dt = utils.solve_haznics(AA_, bb_, W)
        r_norm = 0
        cond = -1
    elif args.precond == "metric_mono":
        # this one solves the monolithic system w cbc.block CG + metricAMG
        interface_dofs = np.arange(W[0].dim(), W[0].dim() + W[1].dim(), dtype=np.int32)
        BB = get_precond(AA_, W, bcs, interface_dofs)

        AAinv = ConjGrad(AA_, precond=BB, tolerance=1E-8, show=4, maxiter=500, callback=cbk)
        xx = AAinv * bb_
        ksp_dt = time.time() - then

        wh = ii_Function(W)
        wh[0].vector().set_local(xx[:W[0].dim()])
        wh[1].vector().set_local(xx[W[0].dim():])

        niters = len(AAinv.residuals)
        r_norm = AAinv.residuals[-1]
        eigenvalues = AAinv.eigenvalue_estimates()
        cond = max(eigenvalues) / min(eigenvalues)
    else:
        # these solve in block fashion
        if args.precond == "metric":
            interface_dofs = np.arange(W[0].dim(), W[0].dim() + W[1].dim(), dtype=np.int32)
            BB = get_precond(AA, W, bcs, interface_dofs)
        else:
            BB = get_precond(AA, W, bcs)
        AAinv = ConjGrad(AA, precond=BB, tolerance=1E-8, show=4, maxiter=500, callback=cbk)
        xx = AAinv * bb
        ksp_dt = time.time() - then

        wh = ii_Function(W)
        for i, xxi in enumerate(xx):
            wh[i].vector().axpy(1, xxi)

        niters = len(AAinv.residuals)
        r_norm = AAinv.residuals[-1]

        eigenvalues = AAinv.eigenvalue_estimates()
        cond = max(eigenvalues) / min(eigenvalues)

    h = W[0].mesh().hmin()
    ndofs = sum(Wi.dim() for Wi in W)

    # Base print
    with open(get_path('iters', 'txt'), 'w') as out:
        out.write('%s\n' % ' '.join(headers_ksp))

    # ---
    ksp_row = (ndofs, niters, cond, ksp_dt, r_norm, h) 
    table_ksp.append(ksp_row)
    utils.print_blue(tabulate.tabulate(table_ksp, headers=headers_ksp))

    with open(get_path('iters', 'txt'), 'a') as out:
        out.write('%s\n' % (' '.join(tuple(map(str, ksp_row)))))

    if args.save:
        File(get_path('uh0', 'pvd')) << wh[0]
        File(get_path('uh1', 'pvd')) << wh[1]        

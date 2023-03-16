from dolfin import *
from xii import *
import sympy as sp
import ulfy


def setup_mms(params):
    '''
    We solve

    -div(kappa1*grad u1) + gamma*(u1-u2) = f1
    -div(kappa2*grad u2) + gamma*(u2-u1) = f2

    with mixed boundary conditions
    '''
    mesh = UnitCubeMesh(1, 1, 1)  # Dummy

    kappa1, kappa2, gamma = Constant(1), Constant(2), Constant(3)

    x, y, z = SpatialCoordinate(mesh)
    u1 = cos(pi*(x + y + 2*z))
    u2 = sin(pi*(x - y + z))

    sigma1 = -kappa1*grad(u1)
    sigma2 = -kappa2*grad(u2)

    f1 = div(sigma1) + gamma*(u1-u2)
    f2 = div(sigma2) + gamma*(u2-u1)

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
        'u2': as_expression(u2),
        'flux2': as_expression(sigma2),
        'f2': as_expression(f2)
    }
    return data

# --------------------------------------------------------------------

if __name__ == '__main__':
    from block.iterative import ConjGrad
    from collections import namedtuple
    import argparse, time, tabulate
    import numpy as np
    import os, utils

    from bidomain_2d import get_system

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-nrefs', type=int, default=1, help='Number of mesh refinements')
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
    
    result_dir = f'./results/bidomain_3d/'
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
    # What we want to collect for error
    headers_error = ['ndofs', 'h', '|eu1|_1', 'r|eu1|_1', '|eu2|_1', 'r|eu2|_1']
    table_error = []

    get_precond = {'diag': utils.get_block_diag_precond,
                   'hypre': utils.get_hypre_monolithic_precond,
                   'hazmath': utils.get_hazmath_amg_precond,  # solve w cbc CG + R.T*hazmathAMG*R preconditioner
                   'metric': utils.get_hazmath_metric_precond,  # solve w cbc CG + R.T*metricAMG*R preconditioner
                   'metric_mono': utils.get_hazmath_metric_precond_mono,  # solve w cbc CG + metricAMG on Amonolithic
                   'metric_hazmath': None}[args.precond]  # solve w hazmath CG + hazmath metricAMG

    mesh_generator = utils.UnitCubeMeshes()
    next(mesh_generator)

    u1_true, u2_true = test_case['u1'], test_case['u2']
    # Let's do this thing
    errors0, h0, diameters = None, None, None
    for ncells in (2**i for i in range(3, 3+args.nrefs)):
        mesh, entity_fs = mesh_generator.send(ncells)
        next(mesh_generator)

        cell_f, facet_f = entity_fs[3], entity_fs[2]

        AA, bb, W, bcs = get_system(facet_f, data=test_case, pdegree=pdegree, parameters=params)

        # For simplicity only use block diagonal preconditioner
        # BB = get_precond(AA, W, bcs)
        
        # AAinv = ConjGrad(AA, precond=BB, tolerance=1E-10, show=4, maxiter=500, callback=cbk)
        # xx = AAinv * bb
        # ksp_dt = time.time() - then

        # Write system to .npy files
        if args.dump and W[0].dim() + W[1].dim() > 1e6:
            utils.dump_system(AA, bb, W)
            exit(0)

        # mono or block
        then = time.time()
        if args.precond in {"metric_mono", "metric_hazmath"}:
            AA_ = ii_convert(AA)
            bb_ = ii_convert(bb)
            cbk = lambda k, x, r, b=bb_, A=AA_: print(f'\titer{k} -> {[(b - A * x).norm("l2")]}')
        else:
            cbk = lambda k, x, r, b=bb, A=AA: print(f'\titer{k} -> {[(b - A * x).norm("l2")]}')

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

        # wh = ii_Function(W)
        # for i, xxi in enumerate(xx):
        #     wh[i].vector().axpy(1, xxi)
        # niters = len(AAinv.residuals)
        # r_norm = AAinv.residuals[-1]
        
        # eigenvalues = AAinv.eigenvalue_estimates()
        # cond = max(eigenvalues)/min(eigenvalues)

        h = W[0].mesh().hmin()
        ndofs = sum(Wi.dim() for Wi in W)

        eu1 = errornorm(u1_true, wh[0], 'H1', degree_rise=1)
        eu2 = errornorm(u2_true, wh[1], 'H1', degree_rise=1)        
        errors = np.array([eu1, eu2])

        if errors0 is None:
            rates = [np.nan]*len(errors)

            # Base print
            with open(get_path('iters', 'txt'), 'w') as out:
                out.write('# %s\n' % ' '.join(headers_ksp))                
            
            with open(get_path('error', 'txt'), 'w') as out:
                out.write('# %s\n' % ' '.join(headers_error))
        else:
            rates = np.log(errors/errors0)/np.log(h/h0)
        errors0, h0 = errors, h

        # ---
        ksp_row = (ndofs, niters, cond, ksp_dt, r_norm, h0) 
        table_ksp.append(ksp_row)
        utils.print_blue(tabulate.tabulate(table_ksp, headers=headers_ksp))

        with open(get_path('iters', 'txt'), 'a') as out:
            out.write('%s\n' % (' '.join(tuple(map(str, ksp_row)))))

        # ---
        error_row = (ndofs, h0) + sum(zip(errors, rates), ())
        table_error.append(error_row)
        utils.print_green(tabulate.tabulate(table_error, headers=headers_error))        
        
        with open(get_path('error', 'txt'), 'a') as out:
            out.write('%s\n' % (' '.join(tuple(map(str, error_row)))))
        
    if args.save:
        File(get_path('uh0', 'pvd')) << wh[0]
        File(get_path('uh1', 'pvd')) << wh[1]        

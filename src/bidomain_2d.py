from dolfin import *
import sympy as sp
import ulfy
from xii import *


def setup_mms(params):
    '''
    We solve

    -div(kappa1*grad u1) + gamma*(u1-u2) = f1
    -div(kappa2*grad u2) + gamma*(u2-u1) = f2

    with mixed boundary conditions
    '''
    mesh = UnitSquareMesh(1, 1)  # Dummy

    kappa1, kappa2, gamma = Constant(1), Constant(2), Constant(3)

    x, y = SpatialCoordinate(mesh)
    u1 = cos(pi*(x + y))
    u2 = sin(pi*(x - y))

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


def get_system(boundaries, data, pdegree, parameters):
    """Setup the linear system A*x = b in W where W has bcs"""
    kappa1, kappa2, gamma = (Constant(parameters.kappa1),
                             Constant(parameters.kappa2),
                             Constant(parameters.gamma))

    mesh = boundaries.mesh()
    V = FunctionSpace(mesh, 'Lagrange', pdegree)
    W = [V, V]

    u1, u2 = map(TrialFunction, W)
    v1, v2 = map(TestFunction, W)

    a = block_form(W, 2)
    a[0][0] = inner(kappa1*grad(u1), grad(v1))*dx + gamma*inner(u1, v1)*dx
    a[0][1] = -gamma*inner(u2, v1)*dx
    a[1][0] = -gamma*inner(u1, v2)*dx
    a[1][1] = inner(kappa2*grad(u2), grad(v2))*dx + gamma*inner(u2, v2)*dx
               
    #  2
    # 3 4
    #  1
    dirichlet_tags = (1, 2)  

    all_tags = {1, 2, 3, 4}
    neumann_tags = tuple(all_tags - set(dirichlet_tags))  # Neumann is full stress
    
    f1, f2 = data['f1'], data['f2']
    sigma1, sigma2 = data['flux1'], data['flux2']
    u1, u2 = data['u1'], data['u2']

    L = block_form(W, 1)
    L[0] = inner(f1, v1)*dx
    L[1] = inner(f2, v2)*dx

    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    # Neumann
    n = FacetNormal(mesh)
    # Add full stress
    L[0] += -sum(inner(dot(sigma1, n), v1)*ds(tag) for tag in neumann_tags)
    L[1] += -sum(inner(dot(sigma2, n), v2)*ds(tag) for tag in neumann_tags)    
        
    bcs = [[DirichletBC(V, u1, boundaries, tag) for tag in dirichlet_tags],
           [DirichletBC(V, u2, boundaries, tag) for tag in dirichlet_tags]]

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
    parser.add_argument('-nrefs', type=int, default=1, help='Number of mesh refinements')
    # Material properties
    parser.add_argument('-kappa1', type=float, default=2, help='Diffusion in 1')
    parser.add_argument('-kappa2', type=float, default=3, help='Diffusion in 2')
    parser.add_argument('-gamma', type=float, default=5, help='Coupling strength')            
    # Discretization
    parser.add_argument('-pdegree', type=int, default=1, help='Polynomial degree in Pk discretization')
    # Solver
    parser.add_argument('-precond', type=str, default='diag', choices=('diag', 'hypre', 'hazmath', 'metric', 'metric_mono', 'metric_hazmath'))
    
    parser.add_argument('-save', type=int, default=0, help='Save graphics')
    parser.add_argument('-dump', type=int, default=0, help='Save matrices and vectors to .npy format')

    args, _ = parser.parse_known_args()
    
    result_dir = f'./results/bidomain_2d/'
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

    mesh_generator = utils.UnitSquareMeshes()
    next(mesh_generator)

    u1_true, u2_true = test_case['u1'], test_case['u2']
    # Let's do this thing
    errors0, h0, diameters = None, None, None
    for ncells in (2**i for i in range(5, 5+args.nrefs)):
        mesh, entity_fs = mesh_generator.send(ncells)
        next(mesh_generator)

        cell_f, facet_f = entity_fs[2], entity_fs[1]

        # assemble
        AA, bb, W, bcs = get_system(facet_f, data=test_case, pdegree=pdegree, parameters=params)

        R = ReductionOperator([2], W)
        error = 0.
        # Check action of system
        for _ in range(20):
            xx = AA.create_vec()
            # this is only cos of petsc error on my comp
            xx[0] = Vector(MPI.comm_self, xx[0].local_size())
            xx[1] = Vector(MPI.comm_self, xx[1].local_size())

            xx.randomize()

            yy0 = AA*xx

            A_ = ii_convert(AA)
            yy = R.T*A_*R*xx
            error = max(error, max(bj.norm('linf') for bj in (yy0 - yy)))
            print("Error system matrix:", error)
        assert error < 1E-14

        # Check action of system
        for _ in range(20):
            xx = AA.create_vec()
            # this is only cos of petsc error on my comp
            xx[0] = Vector(MPI.comm_self, xx[0].local_size())
            xx[1] = Vector(MPI.comm_self, xx[1].local_size())

            xx.randomize()

            yy0 = xx

            yy = R.T * R * xx
            error = max(error, max(bj.norm('linf') for bj in (yy0 - yy)))
            print("Error reduction:", error)
        assert error < 1E-14

        # Check action of preconditioners
        interface_dofs = np.arange(W[0].dim(), W[0].dim() + W[1].dim(), dtype=np.int32)
        AA_ = ii_convert(AA)
        BB_mono = utils.get_hazmath_metric_precond_mono(AA_, W, bcs, interface_dofs)

        BB = utils.get_hazmath_metric_precond(AA, W, bcs, interface_dofs)

        error = 0.
        # Check action of system
        for _ in range(20):
            xx = BB.create_vec()
            xx.randomize()

            yy0 = BB*xx

            yy = R.T*BB_mono*R*xx
            error = max(error, max(bj.norm('linf') for bj in (yy0 - yy)))
            for erri in yy0 - yy:
                indices = np.argwhere(erri[:] > 1e-14)
                print("Error in precond vectors > 1e-14 at ", len(indices), " out of ", erri.local_size(), " dofs")
            print("Max error is ", error)
            print("------------")
        try:
            assert error < 1E-14, error
        except AssertionError:
            utils.print_red('Vectors differ')

        # Is it the reduction?
        error = 0.
        # Check action of system
        for _ in range(20):
            xx = BB.create_vec()
            xx.randomize()

            yy0 = ii_convert(BB*xx)

            yy = BB_mono*ii_convert(xx)
            error = max(error, (yy0 - yy).norm('linf'))
            print("Error without reduction:", error)
        assert error < 1E-14, error
        

        cbk = lambda k, x, r, b=bb, A=AA: print(f'\titer{k} -> {[(b-A*x).norm("l2")]}')

        # Write system to .npy files
        if args.dump and W[0].dim()+W[1].dim() > 1e6:
            utils.dump_system(AA, bb, W)
            exit(0)

        # mono or block
        if args.precond in {"metric_mono", "metric_hazmath"}:
            AA_ = ii_convert(AA)
            bb_ = ii_convert(bb)
            cbk = lambda k, x, r, b=bb_, A=AA_: print(f'\titer{k} -> {[(b - A * x).norm("l2")]}')
        else:
            cbk = lambda k, x, r, b=bb, A=AA: print(f'\titer{k} -> {[(b-A*x).norm("l2")]}')

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

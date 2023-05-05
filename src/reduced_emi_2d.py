# Primal formulation w/out the multiplier
import xii.assembler.trace_matrix
from dolfin import *
from xii import *
import sympy as sp
import ulfy


def setup_mms(params):
    '''
    We solve

    -div(kappa_i*grad u_i) = f_i in Omega_i

    with coupling conditions and mixed boundary conditions
    '''
    mesh = UnitSquareMesh(1, 1)  # Dummy

    kappa1, kappa2, gamma = Constant(1), Constant(2), Constant(3)

    x, y = SpatialCoordinate(mesh)
    u1 = x + 2*y
    u2 = x - y

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


def get_system(cell_f, facet_f, data, pdegree, parameters, strict=True):
    """Setup the linear system A*x = b in W where W has bcs"""
    kappa1, kappa2, gamma = (Constant(parameters.kappa1),
                             Constant(parameters.kappa2),
                             Constant(parameters.gamma))

    # mesh2d = cell_f.mesh()
    # mesh1d = EmbeddedMesh(facet_f, 1)
    mesh2d = cell_f
    mesh1d = facet_f
    V1 = FunctionSpace(mesh2d, 'Lagrange', pdegree)
    V2 = FunctionSpace(mesh1d, 'Lagrange', pdegree)
    
    W = [V1, V2]
    print(f'{V1.dim()} x {V2.dim()}')

    u1, u2 = map(TrialFunction, W)
    v1, v2 = map(TestFunction, W)

    # ds = Measure('ds', domain=mesh2d, subdomain_data=facet_f)
    n = FacetNormal(mesh2d)

    # ---

    Tu1, Tv1 = Trace(u1, mesh1d), Trace(v1, mesh1d)
    dx_ = Measure('dx', domain=mesh1d)
    dx = Measure('dx', domain=mesh2d)

    tau = TangentCurve(mesh1d)

    a = block_form(W, 2)
    a[0][0] = inner(kappa1*grad(u1), grad(v1)) * dx + gamma*inner(Tu1, Tv1)*dx_
    a[0][1] = -gamma*inner(u2, Tv1)*dx_
    a[1][0] = -gamma*inner(Tu1, v2)*dx_
    a[1][1] = inner(kappa2 * dot(grad(u2), tau), dot(grad(v2), tau)) * dx_ + gamma * inner(u2, v2) * dx_
               
    f1, f2 = data['f1'], data['f2']
    sigma1, sigma2 = data['flux1'], data['flux2']
    u1_data, u2_data = data['u1'], data['u2']

    assert not strict or sqrt(abs(assemble(inner(u1_data-u2_data, u1_data-u2_data)*dx_))) < 1E-10

    L = block_form(W, 1)
    L[0] = inner(f1, v1) * dx
    L[1] = inner(f2, v2) * dx_

    V1_bcs = []
    u2_data = data['u2']
    V2_bcs = [DirichletBC(V2, u2_data, 'on_boundary')]
    bcs = [V1_bcs, V2_bcs]

    A, b = map(ii_assemble, (a, L))
    A, b = apply_bc(A, b, bcs)

    return A, b, W, bcs

# --------------------------------------------------------------------

if __name__ == '__main__':
    from block.iterative import ConjGrad, LGMRES
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
    parser.add_argument('-precond', type=str, default='hazmath', choices=('diag', 'hypre', 'hazmath'))
    
    parser.add_argument('-save', type=int, default=0, choices=(0, 1), help='Save graphics')    

    args, _ = parser.parse_known_args()
    
    result_dir = f'./results/reduced_emi_2d/'
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
                   'hazmath': utils.get_hazmath_metric_precond}[args.precond]

    mesh_generator = utils.EMISplitUnitSquareMeshes()
    next(mesh_generator)

    u1_true, u2_true = test_case['u1'], test_case['u2']

    # Let's do this thing
    errors0, h0, diameters = None, None, None
    # for ncells in (2**i for i in range(4, 4+args.nrefs)):
    # meshes = mesh_generator.send(ncells)
    # next(mesh_generator)
    # Meshes
    mesh2d = Mesh()
    with XDMFFile(mesh2d.mpi_comm(), './data/2d1d/2d_grid1.xdmf') as f:
        f.read(mesh2d)

    mesh1d = Mesh()
    with XDMFFile(mesh1d.mpi_comm(), './data/2d1d/1d_grid1.xdmf') as f:
        f.read(mesh1d)

    # cell_f, facet_f = meshes

    AA, bb, W, bcs = get_system(mesh2d, mesh1d,
                                data=test_case, pdegree=pdegree, parameters=params)

    cbk = lambda k, x, r, b=bb, A=AA: print(f'\titer{k} -> {[(b[i]-xi).norm("l2") for i, xi in enumerate(A*x)]}')

    then = time.time()

    BB = get_precond(AA, W, bcs)

    AAinv = ConjGrad(AA, precond=BB, tolerance=1E-10, show=4, maxiter=500, callback=cbk)
    xx = AAinv * bb
    ksp_dt = time.time() - then

    wh = ii_Function(W)
    for i, xxi in enumerate(xx):
        wh[i].vector().axpy(1, xxi)
    niters = len(AAinv.residuals)
    r_norm = AAinv.residuals[-1]

    try:
        eigenvalues = AAinv.eigenvalue_estimates()
        cond = max(eigenvalues)/min(eigenvalues)
    except:
        cond = -1

    h = W[0].mesh().hmin()
    ndofs = sum(Wi.dim() for Wi in W)

    eu1 = errornorm(u1_true, wh[0], 'H1', degree_rise=1)
    eu2 = errornorm(u2_true, wh[1], 'H1', degree_rise=1)
    errors = np.array([eu1, eu2])

    if errors0 is None:
        rates = [np.nan] * len(errors)

        # Base print
        with open(get_path('iters', 'txt'), 'w') as out:
            out.write('%s\n' % ' '.join(headers_ksp))

        with open(get_path('error', 'txt'), 'w') as out:
            out.write('%s\n' % ' '.join(headers_error))
    else:
        rates = np.log(errors / errors0) / np.log(h / h0)
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

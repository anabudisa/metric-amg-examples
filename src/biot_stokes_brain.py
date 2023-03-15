#
# Solve on one mesh
#

from dolfin import *
from xii import *
import sympy as sp
import ulfy


def tangent(f, n):
    return f - dot(f, n)*n


def setup_mms(params, symgrad):
    '''
    We solve

    -div(kappa_i*grad u_i) = f_i in Omega_i 

    with coupling conditions and mixed boundary conditions
    '''
    mesh = UnitCubeMesh(1, 1, 1)  # Dummy

    kappa1, kappa2, gamma = Constant(1), Constant(2), Constant(3)

    r = SpatialCoordinate(mesh)
    u1 = r/Constant(10)
    u2 = Constant(0)*r

    if symgrad:
        transf = sym
    else:
        transf = lambda x: x
    
    sigma1 = kappa1*transf(grad(u1))
    sigma2 = kappa2*transf(grad(u2))

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
        #               
        'g_n': as_expression(Constant((0, 0, 0))),
        # 
        'g_r': as_expression(Constant((0, 0, 0)))
    }
    return data


def get_system(boundaries1, boundaries2, interface, symgrad, data, pdegree, parameters):
    """Setup the linear system A*x = b in W where W has bcs"""
    kappa1, kappa2, gamma = (Constant(parameters.kappa1),
                             Constant(parameters.kappa2),
                             Constant(parameters.gamma))

    mesh1 = boundaries1.mesh()
    mesh2 = boundaries2.mesh()
    
    V1 = VectorFunctionSpace(mesh1, 'Lagrange', pdegree)
    V2 = VectorFunctionSpace(mesh2, 'Lagrange', pdegree)
    
    W = [V1, V2]

    u1, u2 = map(TrialFunction, W)
    v1, v2 = map(TestFunction, W)

    ds1 = Measure('ds', domain=mesh1, subdomain_data=boundaries1)
    n1 = FacetNormal(mesh1)

    ds2 = Measure('ds', domain=mesh2, subdomain_data=boundaries2)
    n2 = FacetNormal(mesh2)

    # ---

    Tu1, Tu2 = Trace(u1, interface), Trace(u2, interface)
    Tv1, Tv2 = Trace(v1, interface), Trace(v2, interface)
    n1_ = OuterNormal(interface, mesh1)
    n2_ = -n1_
    n_ = n1_
    dx_ = Measure('dx', domain=interface)

    if symgrad:
        transf = sym
    else:
        transf = lambda x: x
    
    a = block_form(W, 2)
    a[0][0] = (inner(kappa1*transf(grad(u1)), transf(grad(v1)))*dx +
               gamma*inner(tangent(Tu1, n_), tangent(Tv1, n_))*dx_)
    a[0][1] = -gamma*inner(tangent(Tu2, n_), tangent(Tv1, n_))*dx_
    a[1][0] = -gamma*inner(tangent(Tu1, n_), tangent(Tv2, n_))*dx_
    a[1][1] = (inner(kappa2*transf(grad(u2)), transf(grad(v2)))*dx + 
               gamma*inner(tangent(Tu2, n_), tangent(Tv2, n_))*dx_)
               
    f1, f2 = data['f1'], data['f2']
    sigma1, sigma2 = data['flux1'], data['flux2']
    u1_data, u2_data = data['u1'], data['u2']

    L = block_form(W, 1)
    L[0] = inner(f1, v1)*dx
    L[1] = inner(f2, v2)*dx

    dirichlet_tags1 = (1, )
    neumann_tags1 = ()
    # --
    dirichlet_tags2 = ()
    neumann_tags2 = ()
    
    # Neumann
    # Add full stress
    L[0] += sum(inner(dot(sigma1, n1), v1)*ds1(tag) for tag in neumann_tags1)
    L[1] += sum(inner(dot(sigma2, n2), v2)*ds2(tag) for tag in neumann_tags2)

    g_r, g_n = data['g_r'], data['g_n']    
    # BJS contribution to first ...
    L[0] += sum(inner(dot(dot(sigma1, n1), n1), dot(v1, n1))*ds1(tag) for tag in (1, ))
    L[0] += -sum(inner(g_r, tangent(v1, n1))*ds1(tag) for tag in (1, ))    
    # ... and second
    L[1] += sum(inner(dot(dot(sigma2, n2), n2), dot(v2, n2))*ds2(tag) for tag in (1, ))
    L[1] += -sum(inner(tangent(g_n, n2), tangent(v2, n2))*ds2(tag) for tag in (1, ))        
    L[1] += sum(inner(g_r, tangent(v2, n2))*ds2(tag) for tag in (1, ))
    
    bcs = [[DirichletBC(V1, u1_data, boundaries1, tag) for tag in dirichlet_tags1],
           [DirichletBC(V2, u2_data, boundaries2, tag) for tag in dirichlet_tags2]]

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
    # Problem
    parser.add_argument('-symgrad', type=int, default=1, choices=(0, 1), help='Use symgrad and not grad')        
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
    
    result_dir = f'./results/biot_stokes_brain/'
    not os.path.exists(result_dir) and os.makedirs(result_dir)

    def get_path(what, ext):
        template_path = f'{what}_symgrad{args.symgrad}_precond{args.precond}_kappa1{args.kappa1}_kappa2{args.kappa2}_gamma{args.gamma}_pdegree{args.pdegree}.{ext}'
        return os.path.join(result_dir, template_path)

    Params = namedtuple('Params', ('kappa1', 'kappa2', 'gamma'))
    params = Params(args.kappa1, args.kappa2, args.gamma)
    utils.print_red(str(params) + f' symGrad{args.symgrad}')
    
    # Setup MMS
    test_case = setup_mms(params, symgrad=bool(args.symgrad))

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
    # Load meshes
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), './data/16_enlarged/brain_mesh.h5', 'r') as f:
        f.read(mesh, '/mesh', False)
        
        tdim = mesh.topology().dim()
        subdomains = MeshFunction('size_t', mesh, tdim, 0)
        interfaces = MeshFunction('size_t', mesh, tdim-1, 0)
        
        f.read(subdomains, '/subdomains')
        f.read(interfaces, '/boundaries')

    lm_tags = (5, 6)
    # Stokes - SAS, ventricles, CP
    mesh1 = EmbeddedMesh(subdomains, (1, 3, 4))
    bdries1 = mesh1.translate_markers(interfaces, lm_tags)

    bdries1_ = MeshFunction('size_t', mesh1, mesh1.topology().dim()-1, 0)
    DomainBoundary().mark(bdries1_, 1)

    bdries1.array()[np.where(np.logical_and(bdries1_.array() == 1,
                                            ~np.logical_or(bdries1.array() == 5, bdries1.array() == 6)))] = 1
    
    mesh2 = EmbeddedMesh(subdomains, (2, ))
    bdries2 = mesh2.translate_markers(interfaces, lm_tags)

    interface_mesh = EmbeddedMesh(bdries1, lm_tags)
    interface_mesh.compute_embedding(bdries2, lm_tags)

    AA, bb, W, bcs = get_system(bdries1, bdries2, interface_mesh, symgrad=bool(args.symgrad),
                                data=test_case, pdegree=pdegree, parameters=params)
    # NOTE: For dedicated metric solver from Haznics
    interface_dofs = [utils.get_interface_dofs(Wi, interface_mesh) for Wi in W]
    assert all(len(d) for d in interface_dofs)
    
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
        # interface_dofs = np.arange(W[0].dim(), W[0].dim() + W[1].dim(), dtype=np.int32)
        BB = get_precond(AA_, W, bcs, interface_dofs[0])

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
            # interface_dofs = np.arange(W[0].dim(), W[0].dim() + W[1].dim(), dtype=np.int32)
            BB = get_precond(AA, W, bcs, interface_dofs[0])
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
        out.write('# %s\n' % ' '.join(headers_ksp))                

    # ---
    ksp_row = (ndofs, niters, cond, ksp_dt, r_norm, h) 
    table_ksp.append(ksp_row)
    utils.print_blue(tabulate.tabulate(table_ksp, headers=headers_ksp))

    with open(get_path('iters', 'txt'), 'a') as out:
        out.write('%s\n' % (' '.join(tuple(map(str, ksp_row)))))

    if args.save:
        File(get_path('uh0', 'pvd')) << wh[0]
        File(get_path('uh1', 'pvd')) << wh[1]        

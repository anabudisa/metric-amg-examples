#
# Solve on one mesh
#

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
    print(f'{V1.dim()} x {V2.dim()}')
    
    u1, u2 = map(TrialFunction, W)
    v1, v2 = map(TestFunction, W)

    # ---

    cylinder = Circle(radius=radius, degree=5)
    Ru1, Rv1 = Average(u1, mesh1d, cylinder), Average(v1, mesh1d, cylinder)
    
    dx_ = Measure('dx', domain=mesh1d)

    a = block_form(W, 2)
    a[0][0] = inner(kappa1*grad(u1), grad(v1))*dx + gamma*inner(Ru1, Rv1)*dx_ 
    a[0][1] = -gamma*inner(u2, Rv1)*dx_
    a[1][0] = -gamma*inner(Ru1, v2)*dx_
    a[1][1] = inner(kappa2*grad(u2), grad(v2))*dx_ + gamma*inner(u2, v2)*dx_ 
               
    f1, f2 = data['f1'], data['f2']

    L = block_form(W, 1)
    L[0] = inner(Constant(0), v1)*dx
    L[1] = inner(Constant(0), v2)*dx_

    V1_bcs = []
    u2_data = data['u2']
    V2_bcs = [DirichletBC(V2, u2_data, 'on_boundary && x[2] > 1000')]
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
    parser.add_argument('-precond', type=str, default='diag', choices=('diag', 'hypre'))
    
    parser.add_argument('-save', type=int, default=0, choices=(0, 1), help='Save graphics')    

    args, _ = parser.parse_known_args()
    
    result_dir = f'./results/emi3d1d_sylvie_haznics/'
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
                   'hypre': utils.get_hypre_monolithic_precond}[args.precond]

    # Meshes
    mesh3d = Mesh()
    with XDMFFile(mesh3d.mpi_comm(), './data/sylvie_haznics/3d_c.xdmf') as f:
        f.read(mesh3d)

    mesh1d = Mesh()
    with XDMFFile(mesh3d.mpi_comm(), './data/sylvie_haznics/1d_graph.xdmf') as f:
        f.read(mesh1d)

    radii = Constant(4)
    
    AA, bb, W, bcs = get_system(mesh3d, mesh1d, radii,
                                data=test_case, pdegree=pdegree, parameters=params)

    dump = False
    # ----
    
    if dump:
        print('Write begin')
        from petsc4py import PETSc
        import scipy.sparse as sparse

        [[A, Bt],
         [B, C]] = AA

        b0, b1 = bb

        V0perm = PETSc.IS().createGeneral(np.array(vertex_to_dof_map(W[0]), dtype='int32'))
        V1perm = PETSc.IS().createGeneral(np.array(vertex_to_dof_map(W[1]), dtype='int32'))     

        A_ = as_backend_type(ii_convert(A)).mat().permute(V0perm, V0perm)
        Bt_ = as_backend_type(ii_convert(Bt)).mat().permute(V0perm, V1perm)
        B_ = as_backend_type(ii_convert(B)).mat().permute(V1perm, V0perm)
        C_ = as_backend_type(ii_convert(C)).mat().permute(V1perm, V1perm)
        
        b0_ = as_backend_type(ii_convert(b0)).vec()
        b0_.permute(V0perm)
        b1_ = as_backend_type(ii_convert(b1)).vec()
        b1_.permute(V1perm)     

        def dump(thing, path):
            if isinstance(thing, PETSc.Vec):
                assert np.all(np.isfinite(thing.array))
                return np.save(path, thing.array)

            m = sparse.csr_matrix(thing.getValuesCSR()[::-1]).tocoo()
            assert np.all(np.isfinite(m.data))
            return np.save(path, np.c_[m.row, m.col, m.data])

        dump(A_, 'A.npy') 
        dump(Bt_, 'Bt.npy') 
        dump(B_, 'B.npy') 
        dump(C_, 'C.npy') 

        dump(b0_, 'b0.npy') 
        dump(b1_, 'b1.npy')
        print('Write done')

        exit()
    # ----

    cbk = lambda k, x, r, b=bb, A=AA: print(f'\titer{k} -> {[(b[i]-xi).norm("l2") for i, xi in enumerate(A*x)]}')

    then = time.time()
    # For simplicity only use block diagonal preconditioner
    BB = get_precond(AA, W, bcs)

    AAinv = ConjGrad(AA, precond=BB, tolerance=1E-10, show=4, maxiter=500, callback=cbk)
    xx = AAinv * bb
    ksp_dt = time.time() - then

    wh = ii_Function(W)
    for i, xxi in enumerate(xx):
        wh[i].vector().axpy(1, xxi)
    niters = len(AAinv.residuals)
    r_norm = AAinv.residuals[-1]

    eigenvalues = AAinv.eigenvalue_estimates()
    cond = max(eigenvalues)/min(eigenvalues)

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

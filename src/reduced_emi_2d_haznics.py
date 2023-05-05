# Primal formulation w/out the multiplier
from dolfin import *
from xii import *
import sympy as sp
import ulfy

from reduced_emi_2d import setup_mms, get_system

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
    parser.add_argument('-precond', type=str, default='diag', choices=('diag', 'hypre'))
    
    parser.add_argument('-save', type=int, default=0, choices=(0, 1), help='Save graphics')    

    args, _ = parser.parse_known_args()
    
    result_dir = f'./results/reduced_emi_2d_haznics/'
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
                   'hypre': utils.get_hypre_monolithic_precond}[args.precond]

    # NOTE: because of the geometry this example is not exacted to converge
    # to the manufactured solution (which is for straight crack). Crack splits
    # the unit square into two halves - it is specified by y coordinates of its
    # interior points where x are assumed to be regularly spaced in (0, 1).
    np.random.seed(46)

    crack = 0.8*np.random.rand(20)
    print(crack)
    mesh_generator = utils.ReducedZigZagSplit2d(crack=crack)
    normals = next(mesh_generator)

    u1_true, u2_true, u_true = test_case['u1'], test_case['u2'], test_case['u']
    # Let's do this thing
    errors0, h0, diameters = None, None, None
    for ncells in (1./2**i for i in range(1, 1+args.nrefs)):
        next(mesh_generator)        
        meshes = mesh_generator.send(ncells)

        cell_f, facet_f = meshes
        # Do the surgery here - 2, 3, 4 and 5, 6, 7 are boundaries. ZZ
        # marks interface with > 7 but we want to make it all the same as 1
        # for simplicity
        facet_f.array()[np.where(facet_f.array() > 7)] = 1
        # What is the interface length
        print('\t|iface|', assemble(Constant(1)*dS(domain=facet_f.mesh(), subdomain_data=facet_f)(1)))
        
        AA, bb, W, bcs = get_system(cell_f, facet_f,
                                    data=test_case, pdegree=pdegree, parameters=params,
                                    strict=False)

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

        try:
            eigenvalues = AAinv.eigenvalue_estimates()
            cond = max(eigenvalues)/min(eigenvalues)
        except:
            cond = -1

        h = W[0].mesh().hmin()
        ndofs = sum(Wi.dim() for Wi in W)

        # First is broken H1
        dX = Measure('dx', domain=cell_f.mesh(), subdomain_data=cell_f)
        DG = FunctionSpace(cell_f.mesh(), 'DG', W[0].ufl_element().degree()+3)
        e1a, e1b = interpolate(u1_true, DG), interpolate(u2_true, DG)
        e1a.vector().axpy(-1, interpolate(wh[0], DG).vector())
        e1b.vector().axpy(-1, interpolate(wh[0], DG).vector())
        eu = assemble(inner(grad(e1a), grad(e1a))*dX(1) + 
                      inner(grad(e1b), grad(e1b))*dX(2))
        eu1 = sqrt(eu)
        eu2 = errornorm(u_true, wh[1], 'H1', degree_rise=1)        
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
        print()
        
        with open(get_path('error', 'txt'), 'a') as out:
            out.write('%s\n' % (' '.join(tuple(map(str, error_row)))))
            
    # NOTE: Storing the system for Hazmath solve. This will make sense
    # once we assemble it on Ludmil's meshes but anyways
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

    dump(A_, 'data/neuron/radius1/gamma0/A.npy')
    dump(Bt_, 'Bt.npy') 
    dump(B_, 'B.npy') 
    dump(C_, 'C.npy') 

    dump(b0_, 'b0.npy') 
    dump(b1_, 'b1.npy')
    print('Write done')

    # Here we dump the 1d graph
    uh1d = wh[1]
    mesh1d = uh1d.function_space().mesh()

    num_nodes = mesh1d.num_vertices()
    num_edges = mesh1d.num_cells()
    
    nodes_x = mesh1d.coordinates().flatten().tolist()
    edges = mesh1d.cells().flatten().tolist()

    hazmath_graph = [num_nodes, num_edges, mesh1d.geometry().dim()] + nodes_x + edges
    np.save('reduced_emi2d_graph.npy', hazmath_graph)
        
    if args.save:
        File(get_path('uh0', 'pvd')) << wh[0]
        File(get_path('uh1', 'pvd')) << wh[1]        

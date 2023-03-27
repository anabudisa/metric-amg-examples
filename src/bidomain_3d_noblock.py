from petsc4py import PETSc
from dolfin import *
from xii import *
import sympy as sp
import ulfy

from bidomain_3d import setup_mms
from bidomain_2d_noblock import get_system

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
    parser.add_argument('-precond', type=str, default='hypre', choices=('pyamg', 'hypre'))
    
    parser.add_argument('-save', type=int, default=0, help='Save graphics')    

    args, _ = parser.parse_known_args()
    
    result_dir = f'./results/bidomain_3d_noblock/'
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

    mesh_generator = utils.UnitCubeMeshes()
    next(mesh_generator)

    u1_true, u2_true = test_case['u1'], test_case['u2']
    # Let's do this thing
    errors0, h0, diameters = None, None, None
    for ncells in (2**i for i in range(2, 2+args.nrefs)):
        mesh, entity_fs = mesh_generator.send(ncells)
        next(mesh_generator)

        cell_f, facet_f = entity_fs[3], entity_fs[2]

        AA, bb, W, bcs = get_system(facet_f, data=test_case, pdegree=pdegree, parameters=params)
        # NOTE: !!!
        as_backend_type(AA).mat().setBlockSize(1)
        
        then = time.time()
        
        ksp = PETScKrylovSolver('cg', 'hypre_amg')
        ksp.set_operators(AA, AA)

        ksp_ = ksp.ksp()
        
        OptDB = PETSc.Options()
        OptDB.setValue('ksp_rtol', 1E-12)                
        OptDB.setValue('ksp_view', None)
        OptDB.setValue('ksp_monitor_true_residual', None)                
        OptDB.setValue('ksp_converged_reason', None)
        # OptDB.setValue('options_view', None)        
        OptDB.setValue('ksp_monitor_singular_value', None)

        if args.precond == 'hypre':
            OptDB.setValue('pc_hypre_boomeramg_cycle_type', 'V')
            OptDB.setValue('pc_hypre_boomeramg_max_iter', 1)
            OptDB.setValue('pc_hypre_boomeramg_smooth_type', 'Schwarz-smoothers')
            OptDB.setValue('pc_hypre_boomeramg_interp_type', 'multipass')
            # OptDB.setValue('pc_hypre_boomeramg_nodal_coarsen', 0)  # Use a nodal based coarsening 1-6 (HYPRE_BoomerAMGSetNodal,
            # OptDB.setValue('pc_hypre_boomeramg_vec_interp_variant', 0)  # Variant of algorithm 1-3 (HYPRE_BoomerAMGSetInterpVecVariant,
            OptDB.setValue('pc_hypre_boomeramg_strong_threshold', 0.5)
            # OptDB.setValue('pc_hypre_boomeramg_nodal_coarsen', 2)
        else:
            import pyamg
            from scipy.sparse import csr_matrix
            from petsc_shell import petsc_py_wrapper

            A_mat = as_backend_type(AA).mat()
            A_ = csr_matrix(A_mat.getValuesCSR()[::-1])

            B_ = A_.tobsr(blocksize=(2, 2))
            ml = pyamg.smoothed_aggregation_solver(B_,
                                                   smooth='energy',
                                                   strength='symmetric')
            B_ = ml.aspreconditioner()

            pc = ksp_.getPC()
            pc.setType(PETSc.PC.Type.PYTHON)
            pc.setPythonContext(petsc_py_wrapper(B_))


        ksp_.setConvergenceHistory()
        ksp_.setFromOptions()
        
        wh = Function(W)
        ksp.solve(wh.vector(), bb)

        ksp_dt = time.time() - then
        
        niters = ksp_.getIterationNumber()
        r_norm = ksp_.getConvergenceHistory()[-1]
        
        eigenvalues = ksp_.computeEigenvalues()
        cond = max(eigenvalues)/min(eigenvalues)

        h = W.mesh().hmin()
        ndofs = W.dim()

        wh0, wh1 = wh.split(deepcopy=True)
        eu1 = errornorm(u1_true, wh0, 'H1', degree_rise=1)
        eu2 = errornorm(u2_true, wh1, 'H1', degree_rise=1)        
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
        File(get_path('uh0', 'pvd')) << wh0
        File(get_path('uh1', 'pvd')) << wh1        

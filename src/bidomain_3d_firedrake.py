from firedrake import *
from petsc4py import PETSc
from functools import partial
import numpy as np

print = PETSc.Sys.Print

from bidomain_2d_firedrake import print_red, print_green, print_blue, get_system


def setup_mms(mesh, params):
    '''
    We solve

    -div(kappa1*grad u1) + gamma*(u1-u2) = f1
    -div(kappa2*grad u2) + gamma*(u2-u1) = f2

    with mixed boundary conditions
    '''
    kappa1, kappa2, gamma = Constant(params.kappa1), Constant(params.kappa2), Constant(params.gamma)

    x, y, z = SpatialCoordinate(mesh)
    u1 = cos(pi*(x + y + 2*z))
    u2 = sin(pi*(x - y + z))

    sigma1 = -kappa1*grad(u1)
    sigma2 = -kappa2*grad(u2)

    f1 = div(sigma1) + gamma*(u1-u2)
    f2 = div(sigma2) + gamma*(u2-u1)

    data = {
        'u1': u1,
        'flux1': sigma1,
        'f1': f1,
        'u2': u2,
        'flux2': sigma2,
        'f2': f2
    }
    return data


def get_system(mesh, data, pdegree, parameters, precond):
    """Setup the linear system A*x = b in W where W has bcs"""
    kappa1, kappa2, gamma = (Constant(parameters.kappa1),
                             Constant(parameters.kappa2),
                             Constant(parameters.gamma))

    Velm = FiniteElement('Lagrange', mesh.ufl_cell(), pdegree)
    Welm = MixedElement([Velm, Velm])
    W = FunctionSpace(mesh, Welm)
    
    u1, u2 = TrialFunctions(W)
    v1, v2 = TestFunctions(W)

    a = inner(kappa1*grad(u1), grad(v1))*dx + gamma*inner(u1, v1)*dx
    a += -gamma*inner(u2, v1)*dx
    a += -gamma*inner(u1, v2)*dx
    a += inner(kappa2*grad(u2), grad(v2))*dx + gamma*inner(u2, v2)*dx

    #  2
    # 3 4
    #  1
    dirichlet_tags = (1, 2)  

    all_tags = set((1, 2, 3, 4))
    neumann_tags = tuple(all_tags - set(dirichlet_tags))  # Neumann is full stress
    
    f1, f2 = data['f1'], data['f2']
    sigma1, sigma2 = data['flux1'], data['flux2']
    u1, u2 = data['u1'], data['u2']

    L = inner(f1, v1)*dx
    L += inner(f2, v2)*dx

    ds = Measure('ds', domain=mesh)
    # Neumann
    n = FacetNormal(mesh)
    # Add full stress
    L += -sum(inner(dot(sigma1, n), v1)*ds(tag) for tag in neumann_tags)
    L += -sum(inner(dot(sigma2, n), v2)*ds(tag) for tag in neumann_tags)    
        
    bcs = [DirichletBC(W.sub(0), u1, dirichlet_tags)]
    bcs.append(DirichletBC(W.sub(1), u2, dirichlet_tags))

    solver_parameters = {
        'ksp_type': 'cg',
        'ksp_rtol': 1E-12,
        'ksp_monitor_true_residual': None,
        'ksp_view': None,
        'mg_log_view': None,
        'ksp_view_singularvalues': None,
    }

    mg_parameters = {
        "pc_type": "mg",  # NOTE: setting this one to LU is the exact preconditioner
        "pc_mg_type": 'full', #"full",
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_richardson_scale": 1/3,
        "mg_levels_ksp_max_it": 1,  # NOTE: this was 1
        "mg_levels_ksp_convergence_test": "skip",
        "mg_levels_pc_type": "python",
        "mg_levels_pc_python_type": "firedrake.PatchPC",
        "mg_levels_patch_pc_patch_save_operators": True,
        "mg_levels_patch_pc_patch_construct_type": "star",
        "mg_levels_patch_pc_patch_construct_dim": 0,
        "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
        "mg_levels_patch_sub_ksp_type": "preonly",
        "mg_levels_patch_sub_pc_type": "lu",
        "mg_coarse_pc_type": "python",
        "mg_coarse_pc_python_type": "firedrake.AssembledPC",
        "mg_coarse_assembled_pc_type": "lu",
        "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps"
    }

    amg_parameters = {
        "pc_type": "hypre",  # NOTE: setting this one to LU is the exact preconditioner
        'pc_hypre_boomeramg_cycle_type': 'V',
        'pc_hypre_boomeramg_smooth_type': 'Schwarz-smoothers',
        'pc_hypre_boomeramg_interp_type': 'multipass'
    }
    
    solver_parameters.update(mg_parameters if precond == 'mg' else
                             amg_parameters)

    wh = Function(W)
    problem = LinearVariationalProblem(a, L, wh, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)

    # transfer = TransferManager(use_averaging=True)
    # solver.set_transfer_manager(transfer)    
    
    ksp = solver.snes.ksp
    ksp.setComputeEigenvalues(1)
    ksp.setConvergenceHistory()
    
    solver.solve()

    return wh, ksp

# --------------------------------------------------------------------

if __name__ == '__main__':
    from collections import namedtuple
    import argparse, time, tabulate
    import numpy as np
    import os

    distribution_parameters = {"partition": True,
                               "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-nrefs', type=int, default=1, help='Number of mesh refinements')
    # Material properties
    parser.add_argument('-kappa1', type=float, default=2, help='Diffusion in 1')
    parser.add_argument('-kappa2', type=float, default=3, help='Diffusion in 2')
    parser.add_argument('-gamma', type=float, default=5, help='Coupling strength')            
    # Discretization
    parser.add_argument('-pdegree', type=int, default=1, help='Polynomial degree in Pk discretization')
    # Solver
    parser.add_argument('-precond', type=str, default='hypre', choices=('mg', 'hypre'))
    
    parser.add_argument('-save', type=int, default=0, help='Save graphics')    

    args, _ = parser.parse_known_args()
    
    result_dir = f'./results/bidomain_3d_firedrake/'
    not os.path.exists(result_dir) and os.makedirs(result_dir)

    def get_path(what, ext):
        template_path = f'{what}_precond{args.precond}_kappa1{args.kappa1}_kappa2{args.kappa2}_gamma{args.gamma}_pdegree{args.pdegree}.{ext}'
        return os.path.join(result_dir, template_path)

    Params = namedtuple('Params', ('kappa1', 'kappa2', 'gamma'))
    params = Params(args.kappa1, args.kappa2, args.gamma)
    print_red(str(params))
    
    # Setup problem geometry and discretization
    pdegree = args.pdegree

    # What we want to collect for KSP
    headers_ksp = ['ndofs', 'niters', 'cond', 'timeKSP', 'r', 'h', '|A|']
    table_ksp = []
    # What we want to collect for error
    headers_error = ['ndofs', 'h', '|eu1|_1', 'r|eu1|_1', '|eu2|_1', 'r|eu2|_1']
    table_error = []

    # Let's do this thing
    errors0, h0, diameters = None, None, None
    for ncells in (2**i for i in range(args.nrefs)):

        coarse_mesh = UnitCubeMesh(ncells, ncells, ncells, distribution_parameters=distribution_parameters)
        hierarchy = MeshHierarchy(coarse_mesh, 3)

        mesh = hierarchy[-1]
        mms = setup_mms(mesh, params=params)

        then = time.time()
        wh, ksp = get_system(mesh, data=mms, parameters=params, pdegree=1,
                             precond=args.precond)
        ksp_dt = time.time() - then
        
        niters = ksp.getIterationNumber()
        eigenvalues = ksp.computeEigenvalues()

        r_norm = ksp.getConvergenceHistory()[-1]    

        A_, _ = ksp.getOperators()
        Anorm = A_.norm(PETSc.NormType.NORM_INFINITY)
        
        h = mesh.cell_sizes.vector().dat.data_ro.min()
        ndofs = wh.function_space().dim()

        u1_true, u2_true = mms['u1'], mms['u2']
        
        wh0, wh1 = wh.split()
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

        cond = max(eigenvalues)/min(eigenvalues)
        
        # ---
        ksp_row = (ndofs, niters, cond, ksp_dt, r_norm, h0, Anorm) 
        table_ksp.append(ksp_row)
        print_blue(tabulate.tabulate(table_ksp, headers=headers_ksp))

        with open(get_path('iters', 'txt'), 'a') as out:
            out.write('%s\n' % (' '.join(tuple(map(str, ksp_row)))))

        # ---
        error_row = (ndofs, h0) + sum(zip(errors, rates), ())
        table_error.append(error_row)
        print_green(tabulate.tabulate(table_error, headers=headers_error))        
        
        with open(get_path('error', 'txt'), 'a') as out:
            out.write('%s\n' % (' '.join(tuple(map(str, error_row)))))

# WITH W cycle
from firedrake import *
from petsc4py import PETSc
import numpy as np

print = PETSc.Sys.Print


def bidomain_mms(mesh, params=None):
    '''Manufactured solution for [] case'''
    # Expect all
    if params is not None:
        kappa1, kappa2, gamma = (Constant(params[key])
                                 for key in ('kappa1', 'kappa2', 'gamma'))
    # All is one
    else:
        params = {'kappa1': 1.0, 'kappa2': 1.0, 'gamma': 1.0}
        return elasticity_mms(mesh, params)

    msg = f'kappa1 = {kappa1(0)} kappa2 = {kappa2(0)} gamma = {gamma(0)} | width = {width}'
    print(RED % msg)

    x, y = SpatialCoordinate(mesh)
    # div u = 0
    u1, u2 = sin(pi*(x**2 - y**2)), cos(pi*(x**2 + y**2))

    sigma1, sigma2 = kappa1*grad(u1), kappa2*grad(u2)
    f1, f2 = -div(sigma1) + gamma*(u1-u2), -div(sigma2) + gamma*(u2-u1)

    def get_errors(wh, w=(u1, u2), norm_ops=None):
        wh, = wh
        whs = wh.split()

        results = {}
        for i, wh in enumerate(whs):
            results[f'|eu_{i}|_1'] = errornorm(w[i], wh, 'H1', degree_rise=2)

        return results

    return {'parameters': params,
            'get_errors': get_errors,
            'forces': {'u1': f1, 'traction1': sigma1,
                       'u2': f2, 'traction2': sigma2},
            'solution': {'u1': u1, 'u2': u2}}


def bidomain_system(mesh, width, mms, mg_type='amg'):
    '''Auxiliary'''
    kappa1, kappa2, gamma = (Constant(mms['parameters'][key])
                             for key in ('kappa1', 'kappa2', 'gamma'))

    cell = mesh.ufl_cell()
    Velm = FiniteElement('Lagrange', cell, 1)

    Welm = MixedElement([Velm]*2)
    W = FunctionSpace(mesh, Welm)

    u, v = TrialFunction(W), TestFunction(W)
    us, vs = split(u), split(v)
    u1, u2 = us
    v1, v2 = vs

    x, y = SpatialCoordinate(mesh)
    localize = conditional(le(abs(x-Constant(0.5)), Constant(width)),
                           Constant(1),
                           Constant(0))

    a = (inner(kappa1*grad(u1), grad(v1))*dx + localize*gamma*inner(u1-u2, v1-v2)*dx
         + inner(kappa2*grad(u2), grad(v2))*dx)

    n = FacetNormal(mesh)

    dirichlet_tags = (1, 2)  

    all_tags = set((1, 2, 3, 4))
    neumann_tags = tuple(all_tags - set(dirichlet_tags))  # Neumann is full stress

    f1, f2 = mms['forces']['u1'], mms['forces']['u2']
    sigma1, sigma2 = mms['forces']['traction1'], mms['forces']['traction2']

    u1, u2 = mms['solution']['u1'], mms['solution']['u2']

    L = inner(f1, v1)*dx + inner(f2, v2)*dx
    # Neumann
    # Add full stress
    L += sum(inner(dot(sigma1, n), v1)*ds(tag) for tag in neumann_tags)
    L += sum(inner(dot(sigma2, n), v2)*ds(tag) for tag in neumann_tags)    

    bcs = [DirichletBC(W.sub(0), u1, dirichlet_tags),
           DirichletBC(W.sub(1), u2, dirichlet_tags)]

    solver_parameters = {
        'ksp_type': 'cg',
        'ksp_rtol': 1E-10,
        'ksp_monitor_true_residual': None,
        'ksp_view': None,
        'mg_log_view': None,
        'ksp_view_singularvalues': None,
    }

    if mg_type.lower() == 'amg':
        mg_parameters = {'pc_type': 'hypre'}
    else:
        assert mg_type.lower() == 'mg'

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
    solver_parameters.update(mg_parameters)

    wh = Function(W)
    problem = LinearVariationalProblem(a, L, wh, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    ksp = solver.snes.ksp

    ksp.setComputeEigenvalues(1)

    solver.solve()

    niters = ksp.getIterationNumber()
    eigws = ksp.computeEigenvalues()

    return (wh, niters, eigws)

# --------------------------------------------------------------------

if __name__ == '__main__':
    import os

    distribution_parameters = {"partition": True,
                               "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

    OptDB = PETSc.Options()
    # System size    
    n0 = OptDB.getInt('ncells', 2)
    nrefs = OptDB.getInt('nrefs', 6)

    kappa1 = OptDB.getScalar('kappa1', 3E0)
    kappa2 = OptDB.getScalar('kappa2', 5E0)
    gamma = OptDB.getScalar('gamma', 1E0)

    width = OptDB.getScalar('width', 2.0)

    mg_type = OptDB.getString('mg_type', 'amg')
    not os.path.exists('results') and os.mkdir('results')

    path = f'./results/bidomain_{mg_type}_width{width}_kappa1{kappa1}_kappa2{kappa2}_gamma{gamma}.txt'
    header_base = '# dim niters lmin lmax cond'    
    template_base = '%d %d %g %g %g'

    iters_history = []
    for k in range(n0, n0+nrefs):
        coarse_mesh = UnitSquareMesh(2**k, 2**k, 'left', distribution_parameters=distribution_parameters)
        hierarchy = MeshHierarchy(coarse_mesh, 3)

        mesh = hierarchy[-1]        
        mms = bidomain_mms(mesh, params={'kappa1': kappa1, 'kappa2': kappa2, 'gamma': gamma})

        uh, niters, eigws = bidomain_system(mesh, width=width, mms=mms, mg_type=mg_type)
        errors = mms['get_errors']([uh])
        msg = ' '.join([f'{key} = {val:.3E}' for key, val in errors.items()])
        msg = ' '.join([msg, f'niters = {niters}', f'dofs = {uh.function_space().dim()}'])
        print(GREEN % msg)

        header = ' '.join([header_base] + ['%s' % key for key in errors.keys()]) + '\n'
        template = ' '.join([template_base] + ['%.16f']*len(errors)) + '\n'

        eigws = np.abs(np.real(eigws))
        lmin, lmax = min(eigws), max(eigws)
        result = (uh.function_space().dim(), niters, lmin, lmax, lmax/lmin)
        result = result + tuple(errors[key] for key in errors)
        iters_history.append(result)

        with open(path, 'w') as out:
            out.write(header)
            for line in iters_history:
                out.write(template % line)

    print(iters_history)
# Check PyAMG for Poisson
from petsc4py import PETSc
from petsc_shell import petsc_py_wrapper

# --------------------------------------------------------------------

if __name__ == '__main__':
    from scipy.sparse import csr_matrix
    import dolfin as df
    import numpy as np
    import tabulate
    import pyamg

    seed = PETSc.Random().create(PETSc.COMM_WORLD)
    
    history = []
    for k in range(3, 10):
        ncells = 2**k
        mesh = df.UnitSquareMesh(ncells, ncells)
        x, y = df.SpatialCoordinate(mesh)
        
        V = df.VectorFunctionSpace(mesh, 'CG', 1)

        u, v = df.TrialFunction(V), df.TestFunction(V)
        a = df.inner(df.grad(u), df.grad(v))*df.dx 
        L = df.inner(df.as_vector((df.sin(df.pi*(x+y)), df.cos(df.pi*(x-y)))), v)*df.dx
        bcs = df.DirichletBC(V, df.Constant((0, 0)), 'near(x[0], 0)')

        A, b = df.assemble_system(a, L, bcs)
        A_mat, b_vec = df.as_backend_type(A).mat(), df.as_backend_type(b).vec()

        A_ = csr_matrix(A_mat.getValuesCSR()[::-1])
        b_ = b_vec.array_r

        # Setup the solver
        # NOTE: Ruge-Stuben doesn't use BCR ...
        # ml = pyamg.ruge_stuben_solver(A_.tobsr(blocksize=(2, 2)))
        # ... but block size matters for SAMG
        B_ = A_.tobsr(blocksize=(2, 2))
        ml = pyamg.smoothed_aggregation_solver(B_,
                                               smooth='energy',
                                               strength='symmetric')
        B_ = ml.aspreconditioner()

        # Now we want to solve the problem with PETSc

        ksp = PETSc.KSP().create()
        ksp.setConvergenceHistory()
        ksp.setNormType(PETSc.KSP.NormType.NORM_PRECONDITIONED)
        # Want to get eigenvalue estimates after KSP solve
        ksp.setComputeEigenvalues(1)

        opts = PETSc.Options()
        dopts = opts.getAll()
        # Configure the rest of the solver via options database
        opts.setValue('ksp_type', 'cg')     # Print solver info
        opts.setValue('ksp_view', None)
        opts.setValue('ksp_rtol', 1E-12)
        opts.setValue('ksp_atol', 1E-50)        
        opts.setValue('ksp_view_eigenvalues', None)    
        opts.setValue('ksp_converged_reason', None)
        opts.setValue('ksp_monitor_true_residual', None)
        # opts.setValue('options_view', None)
        opts.setValue('ksp_initial_guess_nonzero', 1)                    

        # opts.setValue('pc_type', 'hypre')
        ksp.setOperators(A_mat)
        # Attach preconditioner
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(petsc_py_wrapper(B_))

        ksp.setUp()
        ksp.setFromOptions()

        uh = df.Function(V)
        x = uh.vector()
        x_vec = df.as_backend_type(x).vec()
        x_vec.setRandom(seed)
        bcs.apply(x)

        ksp.solve(b_vec, x_vec)        

        reason = ksp.getConvergedReason()
        niters = ksp.getIterationNumber()
        residuals = ksp.getConvergenceHistory()
        eigs = ksp.computeEigenvalues()

        error = np.linalg.norm(A_@x_vec.array_r - b_vec.array_r)
        print('|Ax - b|', error)        
        cond = max(np.abs(eigs))/min(np.abs(eigs))
        print('cond', cond)

        history.append((V.dim(), niters, cond))

        print()
        print(tabulate.tabulate(history, headers=('ndofs', 'niters', 'cond')))
        print()

# How does it perform for vector valued problem?
# Any improvement with bcsr
# from IPython import embed
# embed()

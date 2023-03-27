# Check PyAMG for Poisson
from petsc4py import PETSc


def create_mat(A, sizes):
    mat = PETSc.Mat().createPython(sizes)
    mat.setPythonContext(petsc_py_wrapper(A))
    mat.setUp()
    return mat


def create_vec(x):
    return PETSc.Vec().createWithArray(x)


class petsc_py_wrapper:
    # Represent things that support action invoked @ on arrays for PETSc
    def __init__(self, A):
        self.A = A

    # NOTE: this is API for the system operator
    def mult(self, mat, x, y):
        new_y = create_vec(self.A@x.array_r)
        y.aypx(0, new_y)

    # NOTE: apply is the API for preconditioner
    def apply(self, pc, x, y):
        return self.mult(None, x, y)

    def setUp(self, pc):
        pass

# --------------------------------------------------------------------

if __name__ == '__main__':
    from scipy.sparse import csr_matrix
    import dolfin as df
    import numpy as np
    import tabulate
    import pyamg

    seed = PETSc.Random().create(PETSc.COMM_WORLD)
    
    history = []
    for k in range(5, 11):
        ncells = 2**k
        mesh = df.UnitSquareMesh(ncells, ncells)
        x, y = df.SpatialCoordinate(mesh)
        
        V = df.FunctionSpace(mesh, 'CG', 1)

        u, v = df.TrialFunction(V), df.TestFunction(V)
        a = df.inner(df.grad(u), df.grad(v))*df.dx 
        L = df.inner(df.sin(df.pi*(x+y)), v)*df.dx
        bcs = df.DirichletBC(V, df.Constant(0), 'near(x[0], 0)')

        A, b = df.assemble_system(a, L, bcs)
        A_mat, b_vec = df.as_backend_type(A).mat(), df.as_backend_type(b).vec()

        A_ = csr_matrix(A_mat.getValuesCSR()[::-1])
        b_ = b_vec.array_r

        # Setup the solver
        # ml = pyamg.ruge_stuben_solver(A_)
        ml = pyamg.smoothed_aggregation_solver(A_)        
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
        opts.setValue('options_view', None)
        opts.setValue('ksp_initial_guess_nonzero', 1)                    
        
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
        cond = max(np.abs(eigs))/min(np.abs(eigs))
        
        print('|Ax - b|', error)
        print('cond', cond)

        history.append((V.dim(), niters, cond))

        print()
        print(tabulate.tabulate(history, headers=('ndofs', 'niters', 'cond')))
        print()

# How does it perform for vector valued problem?
# Any improvement with bcsr

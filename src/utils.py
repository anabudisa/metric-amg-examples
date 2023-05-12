from functools import partial
from block.algebraic.hazmath import metricAMG
from block.algebraic.petsc import LU
import xii, gmsh
import dolfin as df
import numpy as np


def get_block_diag_precond(A, W, bcs):
    '''Exact blocks LU as preconditioner'''
    n, = set(A.blocks.shape)
    return xii.block_diag_mat([LU(A[i, i]) for i in range(n)])


def get_hazmath_amg_precond(A, W, bcs, parameters=None, interface_dofs=None):
    '''Invert block operator via hypre'''
    import haznics
    from block.algebraic.hazmath import AMG as AMGhaz

    parameters = parameters if parameters is not None else {
        "prectype": 2,  # which precond
        "AMG_type": haznics.UA_AMG,  # (UA, SA) + _AMG
        "cycle_type": haznics.W_CYCLE,  # (V, W, AMLI, NL_AMLI, ADD) + _CYCLE
        "max_levels": 20,
        "maxit": 1,
        "smoother": haznics.SMOOTHER_SGS,  # SMOOTHER_ + (JACOBI, GS, SGS, SSOR, ...) after schwarz method
        "relaxation": 1.2,
        "presmooth_iter": 1,
        "postsmooth_iter": 1,
        "coarse_dof": 100,
        "coarse_solver": 32,  # (32 = SOLVER_UMFPACK, 0 = ITERATIVE)
        "coarse_scaling": haznics.ON,  # (OFF, ON)
        "aggregation_type": haznics.VMB,  # (VMB, MIS, MWM, HEC, HEM)
        "strong_coupled": 0.1,  # threshold
        "max_aggregation": 100, # for HEM this can be any number; it is not used.
        "Schwarz_levels": 0,  # number for levels for Schwarz smoother
        "print_level": 10,
    }

    Minv = AMGhaz(A, parameters=parameters)

    return Minv


def get_hazmath_metric_precond(A, W, bcs, parameters=None, interface_dofs=None):
    '''Invert block operator via hazmath amg'''

    AA = xii.ii_convert(A)
    R = xii.ReductionOperator([len(W)], W)

    Minv = get_hazmath_metric_precond_mono(AA, W, bcs, parameters=parameters, interface_dofs=interface_dofs)

    return R.T * Minv * R


def get_hazmath_metric_precond_mono(A, W, bcs, parameters=None, interface_dofs=None):
    '''Invert block operator via hazmath amg'''
    import haznics

    parameters = parameters if parameters is not None else {
        "AMG_type": haznics.UA_AMG,  # (UA, SA) + _AMG
        "cycle_type": haznics.W_CYCLE,  # (V, W, AMLI, NL_AMLI, ADD) + _CYCLE
        "max_levels": 20,
        "maxit": 1,
        "smoother": haznics.SMOOTHER_SGS,  # SMOOTHER_ + (JACOBI, GS, SGS, SSOR, ...) on coarse levels w/o schwarz
        "relaxation": 1.2,
        "presmooth_iter": 1,
        "postsmooth_iter": 1,
        "coarse_dof": 100,
        "coarse_solver": 32,  # (32 = SOLVER_UMFPACK, 0 = ITERATIVE)
        "coarse_scaling": haznics.ON,  # (OFF, ON)
        "aggregation_type": haznics.HEM,  # (VMB, MIS, MWM, HEC)
        "strong_coupled": 0.1,  # threshold?
        "max_aggregation": 100,
        "amli_degree": 3,
        "Schwarz_levels": 1,  # number for levels where Schwarz smoother is used (1 starts with the finest level)
        "Schwarz_mmsize": 100,  # max block size in Schwarz method
        "Schwarz_maxlvl": 2,  # how many levels from Schwarz seed to take (how large each schwarz block will be)
        "Schwarz_type": haznics.SCHWARZ_SYMMETRIC,  # (SCHWARZ_FORWARD, SCHWARZ_BACKWARD, SCHWARZ_SYMMETRIC)
        "Schwarz_blksolver": 32,  # type of Schwarz block solver, 0 - iterative, 32 - UMFPACK
        "print_level": 10, # 0 - print none, 10 - print all
    }

    # NB: if interface_dofs \not= all dofs, then the interface_dofs has the Schwarz and the rest the GS smoother
    if interface_dofs is not None:
        Minv = metricAMG(A, W, idofs=interface_dofs, parameters=parameters)
    else:
        Minv = metricAMG(A, W, parameters=parameters)

    return Minv

# ---


def solve_haznics(A, b, W, interface_dofs=None):
    from block.algebraic.hazmath import PETSc_to_dCSRmat
    import haznics
    import time

    dimW = sum([VV.dim() for VV in W])
    start_time = time.time()
    # convert vectors
    b_np = b[:]
    bhaz = haznics.create_dvector(b_np)
    xhaz = haznics.dvec_create_p(dimW)

    # convert matrices
    Ahaz = PETSc_to_dCSRmat(A)
    # convert interface DOFs (optional)
    if interface_dofs.all():
        idofs = haznics.create_ivector(interface_dofs)
    else:
        idofs = None

    print("\n Data conversion time: ", time.time() - start_time, "\n")

    # call solver
    solve_start = time.time()
    niters = haznics.fenics_metric_amg_solver_dcsr(Ahaz, bhaz, xhaz, idofs)
    solve_end = time.time() - solve_start

    xx = xhaz.to_ndarray()
    wh = xii.ii_Function(W)
    wh[0].vector().set_local(xx[:W[0].dim()])
    wh[1].vector().set_local(xx[W[0].dim():])

    return niters, wh, solve_end


GREEN = '\033[1;37;32m%s\033[0m'
RED = '\033[1;37;31m%s\033[0m'
BLUE = '\033[1;37;34m%s\033[0m'


def print_color(color, string):
    '''Print with color'''
    print(color % string)
    # NOTE: this is here just to have something to test
    return color


print_red = partial(print_color, RED)
print_green = partial(print_color, GREEN)
print_blue = partial(print_color, BLUE)


# ---

def UnitSquareMeshes():
    '''Stream of meshes'''
    while True:
        ncells = yield

        mesh = df.UnitSquareMesh(ncells, ncells)

        cell_f = df.MeshFunction('size_t', mesh, 2, 1)

        facet_f = df.MeshFunction('size_t', mesh, 1, 0)
        df.CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
        df.CompiledSubDomain('near(x[0], 1)').mark(facet_f, 2)
        df.CompiledSubDomain('near(x[1], 0)').mark(facet_f, 3)
        df.CompiledSubDomain('near(x[1], 1)').mark(facet_f, 4)

        yield (mesh, {2: cell_f, 1: facet_f})


def UnitCubeMeshes():
    '''Stream of meshes'''
    while True:
        ncells = yield

        mesh = df.UnitCubeMesh(ncells, ncells, ncells)

        cell_f = df.MeshFunction('size_t', mesh, 3, 1)

        facet_f = df.MeshFunction('size_t', mesh, 2, 0)
        df.CompiledSubDomain('near(x[2], 0)').mark(facet_f, 1)
        df.CompiledSubDomain('near(x[2], 1)').mark(facet_f, 2)
        df.CompiledSubDomain('near(x[1], 0) || near(x[1], 1)').mark(facet_f, 3)
        df.CompiledSubDomain('near(x[0], 0) || near(x[0], 1)').mark(facet_f, 4)

        yield (mesh, {3: cell_f, 2: facet_f})


# --

def SplitUnitSquareMeshes():
    '''Stream of meshes'''
    while True:
        ncells = yield

        assert ncells >= 4
        mesh = df.UnitSquareMesh(ncells, ncells)

        cell_f = df.MeshFunction('size_t', mesh, 2, 1)
        # Top is 1 bottom i 2
        df.CompiledSubDomain('x[1] < 0.5 + DOLFIN_EPS').mark(cell_f, 2)

        facet_f = df.MeshFunction('size_t', mesh, 1, 0)
        #   3
        # 4  2
        #   1
        # 5  7
        #   6
        df.CompiledSubDomain('near(x[1], 0.5)').mark(facet_f, 1)
        df.CompiledSubDomain('near(x[0], 1) && x[1] > 0.5 - DOLFIN_EPS').mark(facet_f, 2)
        df.CompiledSubDomain('near(x[1], 1)').mark(facet_f, 3)
        df.CompiledSubDomain('near(x[0], 0) && x[1] > 0.5 - DOLFIN_EPS').mark(facet_f, 4)
        df.CompiledSubDomain('near(x[0], 0) && x[1] < 0.5 + DOLFIN_EPS').mark(facet_f, 5)
        df.CompiledSubDomain('near(x[1], 0)').mark(facet_f, 6)
        df.CompiledSubDomain('near(x[0], 1) && x[1] < 0.5 + DOLFIN_EPS').mark(facet_f, 7)

        mesh1 = xii.EmbeddedMesh(cell_f, 1)
        boundaries1 = mesh1.translate_markers(facet_f, (1, 2, 3, 4))

        mesh2 = xii.EmbeddedMesh(cell_f, 2)
        boundaries2 = mesh2.translate_markers(facet_f, (1, 5, 6, 7))

        interface_mesh = xii.EmbeddedMesh(boundaries1, (1,))
        interface_mesh.compute_embedding(boundaries2, (1,))

        yield (boundaries1, boundaries2, interface_mesh)


def SplitUnitCubeMeshes():
    '''Stream of meshes'''
    while True:
        ncells = yield

        assert ncells >= 4
        mesh = df.UnitCubeMesh(ncells, ncells, ncells)

        cell_f = df.MeshFunction('size_t', mesh, 3, 1)
        # Top is 1 bottom i 2
        df.CompiledSubDomain('x[2] < 0.5 + DOLFIN_EPS').mark(cell_f, 2)

        facet_f = df.MeshFunction('size_t', mesh, 2, 0)
        #   3
        # 4  2
        #   1
        # 5  7
        #   6
        df.CompiledSubDomain('near(x[2], 0.5)').mark(facet_f, 1)
        df.CompiledSubDomain('(near(x[0], 0) || near(x[0], 1)) && x[2] > 0.5 - DOLFIN_EPS').mark(facet_f, 2)
        df.CompiledSubDomain('near(x[2], 1)').mark(facet_f, 3)
        df.CompiledSubDomain('(near(x[1], 0) || near(x[1], 1)) && x[2] > 0.5 - DOLFIN_EPS').mark(facet_f, 4)
        df.CompiledSubDomain('(near(x[0], 0) || near(x[0], 1)) && x[2] < 0.5 + DOLFIN_EPS').mark(facet_f, 5)
        df.CompiledSubDomain('near(x[2], 0)').mark(facet_f, 6)
        df.CompiledSubDomain('(near(x[1], 0) || near(x[1], 1)) && x[2] < 0.5 + DOLFIN_EPS').mark(facet_f, 7)

        mesh1 = xii.EmbeddedMesh(cell_f, 1)
        boundaries1 = mesh1.translate_markers(facet_f, (1, 2, 3, 4))

        mesh2 = xii.EmbeddedMesh(cell_f, 2)
        boundaries2 = mesh2.translate_markers(facet_f, (1, 5, 6, 7))

        interface_mesh = xii.EmbeddedMesh(boundaries1, (1,))
        interface_mesh.compute_embedding(boundaries2, (1,))

        yield (boundaries1, boundaries2, interface_mesh)


# --


def get_interface_dofs(V, interface):
    '''Extract dofs of V=V(mesh) on interface'''
    mesh = V.mesh()

    mapping = interface.parent_entity_map
    assert mesh.id() in mapping

    tdim = interface.topology().dim()
    # For now interface should be manifold of co-dim 1
    assert tdim == mesh.topology().dim() - 1
    mapping = mapping[mesh.id()][tdim]

    facet_f = df.MeshFunction('size_t', mesh, tdim, 0)
    facet_f.array()[list(mapping.values())] = 1

    null = df.Constant(np.zeros(V.ufl_element().value_shape()))
    dofs = np.array(list(df.DirichletBC(V, null, facet_f, 1).get_boundary_values().keys()), dtype='int32')
    return dofs


def get_coupling_dofs(V, interface):
    '''Extract dofs of V=V(mesh) on interface'''
    mesh = V.mesh()

    mapping = interface.parent_entity_map
    assert mesh.id() in mapping

    tdim = interface.topology().dim()
    # For now interface should be a subdomain
    assert tdim == mesh.topology().dim()
    mapping = mapping[mesh.id()][tdim]

    dm = V.dofmap()
    dofs = np.concatenate([dm.cell_dofs(cell) for cell in mapping.values()])
    
    return np.unique(dofs)


def dump_system(AA, bb, W, folder=None):
    print('Write begin')
    from petsc4py import PETSc
    import scipy.sparse as sparse

    def dump(thing, path):
        if isinstance(thing, PETSc.Vec):
            assert np.all(np.isfinite(thing.array))
            return np.save(path, thing.array)
        m = sparse.csr_matrix(thing.getValuesCSR()[::-1]).tocoo()
        assert np.all(np.isfinite(m.data))
        return np.save(path, np.c_[m.row, m.col, m.data])

    A_ = df.as_backend_type(xii.ii_convert(AA)).mat()
    b_ = df.as_backend_type(xii.ii_convert(bb)).vec()

    dofs3d = np.arange(W[0].dim(), dtype=np.int32)
    interface_dofs = np.arange(W[0].dim(), W[0].dim() + W[1].dim(), dtype=np.int32)

    folder = './data/' if folder is None else folder
    print("Writing to "+folder+" ...")
    dump(A_, folder+'A.npy')
    dump(b_, folder+'b.npy')

    assert np.all(np.isfinite(interface_dofs.data))
    assert np.all(np.isfinite(dofs3d))
    np.save(folder+'idofs.npy', interface_dofs)
    np.save(folder+'idofs3d.npy', dofs3d)

    print('Write done')

# --------------------------------------------------------------------





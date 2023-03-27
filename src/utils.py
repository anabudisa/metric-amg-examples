from functools import partial
from block.algebraic.petsc import LU, AMG
from block.algebraic.hazmath import metricAMG
import xii
import dolfin as df
import numpy as np


def get_block_diag_precond(A, W, bcs):
    '''Exact blocks LU as preconditioner'''
    n, = set(A.blocks.shape)
    return xii.block_diag_mat([LU(A[i, i]) for i in range(n)])


def get_hypre_monolithic_precond(A, W, bcs):
    '''Invert block operator via hypre'''

    M = xii.ii_convert(A)
    R = xii.ReductionOperator([len(W)], W)

    # NOTE: this is just some settings at the moment    
    parameters = {
        'pc_hypre_boomeramg_cycle_type': 'V',  # (choose one of, V W (None,
        'pc_hypre_boomeramg_max_levels': 25,  # Number of levels (of grids, allowed (None,
        'pc_hypre_boomeramg_max_iter': 1,  # Maximum iterations used PER hypre call (None,
        'pc_hypre_boomeramg_tol': 0,
        # Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations, (None,
        'pc_hypre_boomeramg_truncfactor': 0,  # Truncation factor for interpolation (0=no truncation, (None,
        'pc_hypre_boomeramg_P_max': 0,  # Max elements per row for interpolation operator (0=unlimited, (None,
        'pc_hypre_boomeramg_agg_nl': 0,  # Number of levels of aggressive coarsening (None,
        'pc_hypre_boomeramg_agg_num_paths': 1,  # Number of paths for aggressive coarsening (None,
        'pc_hypre_boomeramg_strong_threshold': 0.25,  # Threshold for being strongly connected (None,
        'pc_hypre_boomeramg_max_row_sum': 0.9,  # Maximum row sum (None,
        'pc_hypre_boomeramg_grid_sweeps_all': 1,  # Number of sweeps for the up and down grid levels (None,
        'pc_hypre_boomeramg_nodal_coarsen': 0,  # Use a nodal based coarsening 1-6 (HYPRE_BoomerAMGSetNodal,
        'pc_hypre_boomeramg_vec_interp_variant': 0,  # Variant of algorithm 1-3 (HYPRE_BoomerAMGSetInterpVecVariant,
        'pc_hypre_boomeramg_grid_sweeps_down': 1,  # Number of sweeps for the down cycles (None,
        'pc_hypre_boomeramg_grid_sweeps_up': 1,  # Number of sweeps for the up cycles (None,
        'pc_hypre_boomeramg_grid_sweeps_coarse': 1,  # Number of sweeps for the coarse level (None,
        'pc_hypre_boomeramg_smooth_type': 'Schwarz-smoothers',
        # (choose one of, Schwarz-smoothers Pilut ParaSails Euclid (None,
        'pc_hypre_boomeramg_smooth_num_levels': 25,  # Number of levels on which more complex smoothers are used (None,

        'pc_hypre_boomeramg_relax_type_all': 'sequential-Gauss-Seidel',
        # (choose one of, Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None,
        'pc_hypre_boomeramg_no_CF': 1,  # Do not use CF-relaxation (None,

        'pc_hypre_boomeramg_measure_type': 'local',  # (choose one of, local global (None,
        'pc_hypre_boomeramg_coarsen_type': 'Falgout',
        # (choose one of, CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS (None,
        'pc_hypre_boomeramg_interp_type': 'multipass',
        # (choose one of, classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts   FF FF1 (None,

        # 'pc_hypre_boomeramg_print_statistics': None,
        # 'pc_hypre_boomeramg_print_debug': None,
        # 'pc_hypre_boomeramg_nodal_relaxation:': 1,  # Nodal relaxation via Schwarz (None,
    }

    Minv = AMG(M, parameters=parameters)

    return R.T * Minv * R


def get_hazmath_amg_precond(A, W, bcs):
    '''Invert block operator via hypre'''
    import haznics
    from block.algebraic.hazmath import AMG as AMGhaz

    M = xii.ii_convert(A)
    R = xii.ReductionOperator([len(W)], W)

    parameters = {
        "prectype": 2,  # which metric precond
        "AMG_type": haznics.UA_AMG,  # (UA, SA) + _AMG
        "cycle_type": haznics.W_CYCLE,  # (V, W, AMLI, NL_AMLI, ADD) + _CYCLE
        "max_levels": 2,
        "maxit": 1,
        "smoother": haznics.SMOOTHER_SGS,  # SMOOTHER_ + (JACOBI, GS, SGS, SSOR, ...) after schwarz method
        "relaxation": 1.0,
        "presmooth_iter": 1,
        "postsmooth_iter": 1,
        "coarse_dof": 100,
        "coarse_solver": 32,  # (32 = SOLVER_UMFPACK, 0 = ITERATIVE)
        "coarse_scaling": haznics.ON,  # (OFF, ON)
        "aggregation_type": haznics.HEC,  # (VMB, MIS, MWM, HEC)
        "strong_coupled": 0.0,  # threshold
        "max_aggregation": 2,
        "Schwarz_levels": 0,  # number for levels for Schwarz smoother
        # "Schwarz_mmsize": 200,  # max block size in Schwarz method
        # "Schwarz_maxlvl": 2,  # how many levels from dof to take
        # "Schwarz_type": haznics.SCHWARZ_SYMMETRIC,  # (SCHWARZ_FORWARD, SCHWARZ_BACKWARD, SCHWARZ_SYMMETRIC)
        # "Schwarz_blksolver": 32,  # type of Schwarz block solver, 0 - iterative, 32 - UMFPACK
        "print_level": 10,
    }

    Minv = AMGhaz(M, parameters=parameters)
    Ainv = R.T * Minv * R
    return Ainv


def get_hazmath_metric_precond(A, W, bcs, interface_dofs=None):
    '''Invert block operator via hazmath amg'''
    import haznics

    AA = xii.ii_convert(A)
    R = xii.ReductionOperator([len(W)], W)

    Minv = get_hazmath_metric_precond_mono(AA, W, bcs, interface_dofs=interface_dofs)

    return R.T * Minv * R


def get_hazmath_metric_precond_mono(A, W, bcs, interface_dofs=None):
    '''Invert block operator via hazmath amg'''
    import haznics

    parameters = {
        "AMG_type": haznics.UA_AMG,  # (UA, SA) + _AMG
        "cycle_type": haznics.W_CYCLE,  # (V, W, AMLI, NL_AMLI, ADD) + _CYCLE
        "max_levels": 10,
        "maxit": 1,
        "smoother": haznics.SMOOTHER_GS,  # SMOOTHER_ + (JACOBI, GS, SGS, SSOR, ...) on coarse levels w/o schwarz
        "relaxation": 1.2,
        "presmooth_iter": 1,
        "postsmooth_iter": 1,
        "coarse_dof": 100,
        "coarse_solver": 32,  # (32 = SOLVER_UMFPACK, 0 = ITERATIVE)
        "coarse_scaling": haznics.ON,  # (OFF, ON)
        "aggregation_type": haznics.HEC,  # (VMB, MIS, MWM, HEC)
        "strong_coupled": 0.1,  # threshold?
        "max_aggregation": 20,
        "amli_degree": 3,
        "Schwarz_levels": 1,  # number for levels where Schwarz smoother is used (1 starts with the finest level)
        "Schwarz_mmsize": 200,  # max block size in Schwarz method
        "Schwarz_maxlvl": 2,  # how many levels from Schwarz seed to take (how large each schwarz block will be)
        "Schwarz_type": haznics.SCHWARZ_SYMMETRIC,  # (SCHWARZ_FORWARD, SCHWARZ_BACKWARD, SCHWARZ_SYMMETRIC)
        "Schwarz_blksolver": 32,  # type of Schwarz block solver, 0 - iterative, 32 - UMFPACK
        "print_level": 5, # 0 - print none, 10 - print all
    }

    # NB: if interface_dofs \not= all dofs, then the interface_dofs has the Schwarz and the rest the GS smoother
    if interface_dofs is not None:
        Minv = metricAMG(A, W, idofs=interface_dofs, parameters=parameters)
    else:
        Minv = metricAMG(A, W, parameters=parameters)

    return Minv
    # return A
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

    if interface_dofs is None:
        interface_dofs = np.arange(W[0].dim(), W[0].dim() + W[1].dim(), dtype=np.int32)
    idofs = None #haznics.create_ivector(interface_dofs)
    print("\n------------------- Data conversion time: ", time.time() - start_time, "\n")

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

def EMISplitUnitSquareMeshes():
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

        yield (cell_f, facet_f)


# --------------------------------------------------------------------

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


def dump_system(AA, bb, W):
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

    [[A, Bt],
     [B, C]] = AA
    b0, b1 = bb
    V0perm = PETSc.IS().createGeneral(np.array(df.vertex_to_dof_map(W[0]), dtype='int32'))
    V1perm = PETSc.IS().createGeneral(np.array(df.vertex_to_dof_map(W[1]), dtype='int32'))
    A_ = df.as_backend_type(xii.ii_convert(A)).mat().permute(V0perm, V0perm)
    Bt_ = df.as_backend_type(xii.ii_convert(Bt)).mat().permute(V0perm, V1perm)
    B_ = df.as_backend_type(xii.ii_convert(B)).mat().permute(V1perm, V0perm)
    C_ = df.as_backend_type(xii.ii_convert(C)).mat().permute(V1perm, V1perm)

    b0_ = df.as_backend_type(xii.ii_convert(b0)).vec()
    b0_.permute(V0perm)
    b1_ = df.as_backend_type(xii.ii_convert(b1)).vec()
    b1_.permute(V1perm)

    csr = B_.getValuesCSR()
    interface_dofs = np.arange(W[0].dim(), W[0].dim() + W[1].dim(), dtype=np.int32)

    dump(A_, 'A.npy')
    dump(Bt_, 'Bt.npy')
    dump(B_, 'B.npy')
    dump(C_, 'C.npy')
    dump(b0_, 'b0.npy')
    dump(b1_, 'b1.npy')
    assert np.all(np.isfinite(interface_dofs.data))
    np.save('idofs.npy', interface_dofs)
    print('Write done')

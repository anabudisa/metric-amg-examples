from gmshnics import msh_gmsh_model, mesh_from_gmsh
from functools import partial
from block.algebraic.petsc import LU, AMG
import xii, gmsh
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
        'pc_hypre_boomeramg_max_levels': 25,  #  Number of levels (of grids, allowed (None,
        'pc_hypre_boomeramg_max_iter': 1,  #  Maximum iterations used PER hypre call (None,
        'pc_hypre_boomeramg_tol': 0,  #  Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations, (None,
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
        'pc_hypre_boomeramg_smooth_type': 'Schwarz-smoothers',   # (choose one of, Schwarz-smoothers Pilut ParaSails Euclid (None,
        'pc_hypre_boomeramg_smooth_num_levels': 25,  # Number of levels on which more complex smoothers are used (None,
        
        'pc_hypre_boomeramg_relax_type_all': 'sequential-Gauss-Seidel',  # (choose one of, Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None,
        'pc_hypre_boomeramg_no_CF': 1, # Do not use CF-relaxation (None,

        'pc_hypre_boomeramg_measure_type': 'local',  # (choose one of, local global (None,
        'pc_hypre_boomeramg_coarsen_type': 'Falgout',  # (choose one of, CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS (None,
        'pc_hypre_boomeramg_interp_type': 'classical',  # (choose one of, classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts   FF FF1 (None,
        
        # 'pc_hypre_boomeramg_print_statistics': None,
        # 'pc_hypre_boomeramg_print_debug': None,
        # 'pc_hypre_boomeramg_nodal_relaxation:': 1,  # Nodal relaxation via Schwarz (None,
    }
    
    Minv = AMG(M, parameters=parameters)
    
    return  R.T*Minv*R

# ---

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

        interface_mesh = xii.EmbeddedMesh(boundaries1, (1, ))
        interface_mesh.compute_embedding(boundaries2, (1, ))        

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

        interface_mesh = xii.EmbeddedMesh(boundaries1, (1, ))
        interface_mesh.compute_embedding(boundaries2, (1, ))

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


def ThinStripMeshes2d(width, view=False, **kwargs):
    '''[ [] ]'''
    assert 0 < width < 0.5

    gmsh.initialize(['', '-v', '0'] + sum([[f'-{key}', str(val)] for key, val in kwargs.items()], []))
    model = gmsh.model
    factory = model.occ

    points = [[0, 0],
              [0.5-width/2, 0],
              [0.5, 0],
              [0.5+width/2, 0],
              [1, 0],
              [1, 1],
              [0.5+width/2, 1],
              [0.5, 1],
              [0.5-width/2, 1],
              [0, 1]]
    points = np.array([factory.addPoint(xi, yi, z=0) for xi, yi in points])

    lines = [factory.addLine(points[p], points[q])
             for (p, q) in [(0, 1), (1, 2), (2, 3), (3, 4),
                            (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
                            (1, 8), (2, 7), (3, 6)]]

    factory.synchronize()

    loops = (factory.addCurveLoop([lines[0], lines[10], lines[8], lines[9]]),
             factory.addCurveLoop([lines[1], lines[2], lines[12], lines[6], lines[7], -lines[10]]),
             factory.addCurveLoop([lines[3], lines[4], lines[5], -lines[12]]))

    surfaces = [factory.addPlaneSurface([loop]) for loop in loops]

    factory.synchronize()
    
    [model.addPhysicalGroup(2, [surface], tag) for tag, surface in enumerate(surfaces, 1)]
    # Only pick the outer as
    #  4
    # 1  2
    #   3
    model.addPhysicalGroup(1, [lines[9]], 1)
    model.addPhysicalGroup(1, [lines[4]], 2)
    # Leave out the boundary pieces that are also strip boundaries
    model.addPhysicalGroup(1, [lines[0], lines[3]], 3)
    model.addPhysicalGroup(1, [lines[5], lines[8]], 4)

    factory.synchronize()

    if view:
        gmsh.fltk.initialize()
        gmsh.fltk.run()

    # These will stay the same for every resolution

    while True:
        mesh_size = yield

        if mesh_size < 0: break
        
        gmsh.option.setNumber('Mesh.MeshSizeFactor', float(mesh_size))

        nodes, topologies = msh_gmsh_model(model, 2)
        mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

        # 1 2 3
        cell_f, facet_f = entity_functions[2], entity_functions[1]
        #   4
        # 1   2
        #   3
        
        mesh1 = xii.EmbeddedMesh(cell_f, (1, 2))
        boundaries1 = mesh1.translate_markers(facet_f, (1, 3, 4))

        mesh2 = xii.EmbeddedMesh(cell_f, (2, 3))
        boundaries2 = mesh2.translate_markers(facet_f, (2, 3, 4))

        strip = xii.EmbeddedMesh(mesh1.marking_function, (2, ))
        strip.compute_embedding(mesh2.marking_function, (2, ))
        
        yield (boundaries1, boundaries2, strip)
        
        gmsh.model.mesh.clear()
        
    gmsh.finalize()        


def ThinStripMeshes3d(width, view=False, **kwargs):
    '''[ [] ]'''
    assert 0 < width < 0.5

    gmsh.initialize(['', '-v', '0'] + sum([[f'-{key}', str(val)] for key, val in kwargs.items()], []))
    model = gmsh.model
    factory = model.occ

    points = [[0, 0],
              [0.5-width/2, 0],
              [0.5, 0],
              [0.5+width/2, 0],
              [1, 0],
              [1, 1],
              [0.5+width/2, 1],
              [0.5, 1],
              [0.5-width/2, 1],
              [0, 1]]
    points = np.array([factory.addPoint(xi, yi, z=0) for xi, yi in points])

    lines = [factory.addLine(points[p], points[q])
             for (p, q) in [(0, 1), (1, 2), (2, 3), (3, 4),
                            (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
                            (1, 8), (2, 7), (3, 6)]]

    factory.synchronize()

    loops = (factory.addCurveLoop([lines[0], lines[10], lines[8], lines[9]]),
             factory.addCurveLoop([lines[1], lines[2], lines[12], lines[6], lines[7], -lines[10]]),
             factory.addCurveLoop([lines[3], lines[4], lines[5], -lines[12]]))

    surfaces = [factory.addPlaneSurface([loop]) for loop in loops]

    factory.extrude([(2, surf) for surf in surfaces], 0, 0, 1)
    factory.synchronize()

    factory.removeAllDuplicates()

    factory.synchronize()

    volumes = model.getEntities(3)
    volumes = sorted(volumes, key=lambda p: factory.getCenterOfMass(*p)[0])
    
    [model.addPhysicalGroup(3, [dimTag[1]], tag) for tag, dimTag in enumerate(volumes, 1)]

    left, center, right = volumes
    midboundary = set(model.getBoundary([center], oriented=False))

    left_boundaries = model.getBoundary([left], oriented=False)
    leftmost = min(left_boundaries, key=lambda p: factory.getCenterOfMass(*p)[0])
    lymin = min(left_boundaries, key=lambda p: factory.getCenterOfMass(*p)[1])
    lymax = max(left_boundaries, key=lambda p: factory.getCenterOfMass(*p)[1])    
    lzmin = min(left_boundaries, key=lambda p: factory.getCenterOfMass(*p)[2])
    lzmax = max(left_boundaries, key=lambda p: factory.getCenterOfMass(*p)[2])    

    right_boundaries = model.getBoundary([right], oriented=False)
    rightmost = max(right_boundaries, key=lambda p: factory.getCenterOfMass(*p)[0])
    rymin = min(right_boundaries, key=lambda p: factory.getCenterOfMass(*p)[1])
    rymax = max(right_boundaries, key=lambda p: factory.getCenterOfMass(*p)[1])    
    rzmin = min(right_boundaries, key=lambda p: factory.getCenterOfMass(*p)[2])
    rzmax = max(right_boundaries, key=lambda p: factory.getCenterOfMass(*p)[2])    
    
    model.addPhysicalGroup(2, [leftmost[1]], 1)
    model.addPhysicalGroup(2, [rightmost[1]], 2)
    # Leave out the boundary pieces that are also strip boundaries
    model.addPhysicalGroup(2, [lymin[1], rymin[1], lymax[1], rymax[1]], 3)
    model.addPhysicalGroup(2, [lzmin[1], rzmin[1], lzmax[1], rzmax[1]], 4)

    factory.synchronize()

    if view:
        gmsh.fltk.initialize()
        gmsh.fltk.run()

    # These will stay the same for every resolution

    while True:
        mesh_size = yield

        if mesh_size < 0: break
        
        gmsh.option.setNumber('Mesh.MeshSizeFactor', float(mesh_size))

        nodes, topologies = msh_gmsh_model(model, 3)
        mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

        # 1 2 3
        cell_f, facet_f = entity_functions[3], entity_functions[2]
        #   4
        # 1   2
        #   3
        
        mesh1 = xii.EmbeddedMesh(cell_f, (1, 2))
        boundaries1 = mesh1.translate_markers(facet_f, (1, 3, 4))

        mesh2 = xii.EmbeddedMesh(cell_f, (2, 3))
        boundaries2 = mesh2.translate_markers(facet_f, (2, 3, 4))

        strip = xii.EmbeddedMesh(mesh1.marking_function, (2, ))
        strip.compute_embedding(mesh2.marking_function, (2, ))
        
        yield (boundaries1, boundaries2, strip)
        
        gmsh.model.mesh.clear()
        
    gmsh.finalize()        

    
# ---

def get_interface_dofs(V, interface):
    '''Extract dofs of V=V(mesh) on interface'''
    mesh = V.mesh()

    mapping = interface.parent_entity_map
    assert mesh.id() in mapping

    tdim = interface.topology().dim()
    # For now interface should be manifold of co-dim 1
    assert tdim == mesh.topology().dim()-1
    mapping = mapping[mesh.id()][tdim]

    facet_f = df.MeshFunction('size_t', mesh, tdim, 0)
    facet_f.array()[list(mapping.values())] = 1

    null = df.Constant(np.zeros(V.ufl_element().value_shape()))
    dofs = list(df.DirichletBC(V, null, facet_f, 1).get_boundary_values().keys())

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

# --------------------------------------------------------------------

if __name__ == '__main__':

    meshes = ThinStripMeshes3d(width=0.2, view=True)
    next(meshes)

    for scale in (0.2, 0.1):
         bdry1, bdry2, strip = meshes.send(scale)
         next(meshes)

    df.File('bdr1.pvd') << bdry1
    df.File('bdr2.pvd') << bdry2    

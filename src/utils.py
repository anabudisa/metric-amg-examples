from gmshnics import msh_gmsh_model, mesh_from_gmsh
from functools import partial
from block.algebraic.hazmath import metricAMG
import xii, gmsh
import dolfin as df
import numpy as np


def get_hazmath_amg_precond(A, W, bcs, interface_dofs=None):
    '''Invert block operator via hypre'''
    import haznics
    from block.algebraic.hazmath import AMG as AMGhaz

    parameters = {
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


def get_hazmath_metric_precond(A, W, bcs, interface_dofs=None):
    '''Invert block operator via hazmath amg'''

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
        "Schwarz_maxlvl": 1,  # how many levels from Schwarz seed to take (how large each schwarz block will be)
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


def ZigZagSplit2d(crack, view=False, **kwargs):
    '''[__/\___] Unit square with a crack in the middle
       [       ]
    '''
    assert not crack or all(-0.5 < p < 0.5 for p in crack)

    gmsh.initialize(['', '-v', '0'] + sum([[f'-{key}', str(val)] for key, val in kwargs.items()], []))
    model = gmsh.model
    factory = model.occ

    outer_points = [[1, 0],          # 0
                    [1, 0.5],        
                    [0, 0.5],  
                    [0, 0],          # 3
                    [0, -0.5],
                    [1, -0.5]]       # 
    outer_points = np.array([factory.addPoint(xi, yi, z=0) for xi, yi in outer_points])
    # We can make the outer boundary
    npoints = len(outer_points)
    outer_lines = [factory.addLine(outer_points[i], outer_points[(i+1)%npoints])
                   for i in range(npoints)]

    # NOTE: crack[i] are y coordinates
    x_crack = np.linspace(0, 1, len(crack)+2)[1:-1]
    crack = np.c_[x_crack, crack]

    crack_points = [outer_points[3]]
    crack_points.extend(factory.addPoint(*p, z=0) for p in crack)
    crack_points.append(outer_points[0])

    factory.synchronize()

    tangents = []
    # Compute tangent vector
    for Atag, Btag in zip(crack_points[:-1], crack_points[1:]):
        A = model.getValue(0, Atag, [])[:2]
        B = model.getValue(0, Btag, [])[:2]
        t = np.array(B) - np.array(A)
        t = t / np.linalg.norm(t)
        tangents.append(t)
        
    crack_lines = [factory.addLine(crack_points[i], crack_points[i+1])
                   for i in range(len(crack_points)-1)]
    
    factory.synchronize()

    top_loop = [outer_lines[0], outer_lines[1], outer_lines[2]] + crack_lines
    top_loop = factory.addCurveLoop(top_loop)
    top_surface = factory.addPlaneSurface([top_loop])

    bottom_loop = [outer_lines[3], outer_lines[4], outer_lines[5]] + [-l for l in crack_lines]
    bottom_loop = factory.addCurveLoop(bottom_loop)
    bottom_surface = factory.addPlaneSurface([bottom_loop])
    
    factory.synchronize()
    
    model.addPhysicalGroup(2, [top_surface], 1)
    model.addPhysicalGroup(2, [bottom_surface], 2)

    outer_line_tags = [model.addPhysicalGroup(1, [l], tag) for tag, l in enumerate(outer_lines, 2)]

    Rot = np.array([[0, 1], [-1, 0]])
    normals = {}
    for tag, (l, tau) in enumerate(zip(crack_lines, tangents), 2+len(outer_lines)):
        model.addPhysicalGroup(1, [l], tag)
        normals[tag] = Rot@tau

    yield normals
        
    factory.synchronize()

    if view:
        gmsh.fltk.initialize()
        gmsh.fltk.run()

    nol = len(outer_line_tags)
    iface_tags = tuple(sorted(normals.keys()))
    # These will stay the same for every resolution
    while True:
        mesh_size = yield

        if mesh_size < 0: break
        
        gmsh.option.setNumber('Mesh.MeshSizeFactor', float(mesh_size))

        nodes, topologies = msh_gmsh_model(model, 2)
        mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

        cell_f, facet_f = entity_functions[2], entity_functions[1]

        mesh1 = xii.EmbeddedMesh(cell_f, 1)
        boundaries1 = mesh1.translate_markers(facet_f, tuple(outer_line_tags[:nol//2]) + iface_tags)

        mesh2 = xii.EmbeddedMesh(cell_f, 2)
        boundaries2 = mesh2.translate_markers(facet_f, tuple(outer_line_tags[nol//2:]) + iface_tags)

        interface_mesh = xii.EmbeddedMesh(boundaries1, iface_tags)
        interface_mesh.compute_embedding(boundaries2, iface_tags)

        yield (boundaries1, boundaries2, interface_mesh)
        
        gmsh.model.mesh.clear()
        
    gmsh.finalize()

    
def ReducedZigZagSplit2d(crack, view=False, **kwargs):
    '''[__/\___] Unit square with a crack in the middle
       [       ]
    '''
    assert not len(crack) or all(0 < p < 1 for p in crack)

    gmsh.initialize(['', '-v', '0'] + sum([[f'-{key}', str(val)] for key, val in kwargs.items()], []))
    model = gmsh.model
    factory = model.occ

    outer_points = [[1, 0.5],          # 0
                    [1, 1.0],        
                    [0, 1.0],  
                    [0, 0.5],          # 3
                    [0, 0.0],
                    [1, 0.0]]       # 
    outer_points = np.array([factory.addPoint(xi, yi, z=0) for xi, yi in outer_points])
    # We can make the outer boundary
    npoints = len(outer_points)
    outer_lines = [factory.addLine(outer_points[i], outer_points[(i+1)%npoints])
                   for i in range(npoints)]

    # NOTE: crack[i] are y coordinates
    x_crack = np.linspace(0, 1, len(crack)+2)[1:-1]
    crack = np.c_[x_crack, crack]

    crack_points = [outer_points[3]]
    crack_points.extend(factory.addPoint(*p, z=0) for p in crack)
    crack_points.append(outer_points[0])

    factory.synchronize()

    tangents = []
    # Compute tangent vector
    for Atag, Btag in zip(crack_points[:-1], crack_points[1:]):
        A = model.getValue(0, Atag, [])[:2]
        B = model.getValue(0, Btag, [])[:2]
        t = np.array(B) - np.array(A)
        t = t / np.linalg.norm(t)
        tangents.append(t)
        
    crack_lines = [factory.addLine(crack_points[i], crack_points[i+1])
                   for i in range(len(crack_points)-1)]
    
    factory.synchronize()

    top_loop = [outer_lines[0], outer_lines[1], outer_lines[2]] + crack_lines
    top_loop = factory.addCurveLoop(top_loop)
    top_surface = factory.addPlaneSurface([top_loop])

    bottom_loop = [outer_lines[3], outer_lines[4], outer_lines[5]] + [-l for l in crack_lines]
    bottom_loop = factory.addCurveLoop(bottom_loop)
    bottom_surface = factory.addPlaneSurface([bottom_loop])
    
    factory.synchronize()
    
    model.addPhysicalGroup(2, [top_surface], 1)
    model.addPhysicalGroup(2, [bottom_surface], 2)

    outer_line_tags = [model.addPhysicalGroup(1, [l], tag) for tag, l in enumerate(outer_lines, 2)]

    Rot = np.array([[0, 1], [-1, 0]])
    normals = {}
    for tag, (l, tau) in enumerate(zip(crack_lines, tangents), 2+len(outer_lines)):
        model.addPhysicalGroup(1, [l], tag)
        normals[tag] = Rot@tau

    yield normals
        
    factory.synchronize()

    if view:
        gmsh.fltk.initialize()
        gmsh.fltk.run()

    nol = len(outer_line_tags)
    iface_tags = tuple(sorted(normals.keys()))
    # These will stay the same for every resolution
    while True:
        mesh_size = yield

        if mesh_size < 0: break
        
        gmsh.option.setNumber('Mesh.MeshSizeFactor', float(mesh_size))

        nodes, topologies = msh_gmsh_model(model, 2)
        mesh, entity_functions = mesh_from_gmsh(nodes, topologies)

        cell_f, facet_f = entity_functions[2], entity_functions[1]

        yield (cell_f, facet_f)
        
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


if __name__ == '__main__':

    crack = [0.2, 0, -0.3, 0.2, 0.234, 0.4, 0.2, -0.1]
    
    meshes = ZigZagSplit2d(crack, view=True)
    normals = next(meshes)
    next(meshes)

    for scale in (0.2, 0.1):
         bdry1, bdry2, strip = meshes.send(scale)
         next(meshes)

    df.File('bdr1.pvd') << bdry1
    df.File('bdr2.pvd') << bdry2    




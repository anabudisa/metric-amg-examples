"""
\file taken from HAZniCS-examples/demo_3d1d.py

We solve on Omega (3d) and Gamma (1d) a reduced EMI problem of signal propagation in neurons,
that is the steady-state electrodiffusion in a porous tissue (3d) and the network of 1d curves (immersed in Omega)

    - div(sigma_3 grad(p3)) + (rho * C_m / deltat) * (p3 - Avg^T(p1)) * delta_Gamma = f3 on Omega
            - div(rho^2 sigma_1 grad(p1)) + (rho * C_m / deltat) * (p1 - Avg(p3)) = f1 on Gamma

with Avg(p3) the average of the function on Omega to Gamma over a cylinder-type surface
of rho-radius around Gamma. delta_Gamma is the delta-distribution on Gamma.
sigma_3 and sigma_1 are extra- and intracellular conductivities. C_m is the membrane capacitance parameter.
deltat is the time step size (this linear system results from a time-stepping scheme for the dynamic diffusion problem)

We enforce homogeneous Neumann conditions on the outer boundary of the 3d (and 1d) domain.
We solve the problem with Conjugate Gradient method preconditioned with "metric AMG" method that
uses block Schwarz smoothers.
"""
from scipy.sparse import csr_matrix
from xii.assembler.average_matrix import average_matrix as average_3d1d_matrix, trace_3d1d_matrix
from block.algebraic.hazmath import block_mat_to_block_dCSRmat
from dolfin import *
from xii import *
import haznics
import time


def get_mesh_neuron():
    '''Load it'''
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), './data/PolyIC_3AS2_1.CNG.c1.h5', 'r') as h5:
        h5.read(mesh, '/mesh', False)
        curves = MeshFunction('double', mesh, 1)
        h5.read(curves, '/curves')

    # As we do not care about mapping for radii we recolor the mesh
    values = curves.array()
    not_neuron = np.where(values == 0)
    values *= 0  # Zero
    values += 1  # Mark everyone
    values[not_neuron] = 0  # Only leave the neuron

    return curves


def get_system(edge_f, k3=1e0, k1=1e0, gamma=1e0, coupling_radius=0.):
    """A, b, W, bcs"""
    assert edge_f.dim() == 1

    # Meshes
    meshV = edge_f.mesh()  #
    meshQ = EmbeddedMesh(edge_f, 1)

    # Spaces
    V = FunctionSpace(meshV, 'CG', 1)
    Q = FunctionSpace(meshQ, 'CG', 1)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)

    # Average (coupling_radius > 0) or trace (coupling_radius = 0)
    if coupling_radius > 0:
        # Averaging surface
        cylinder = Circle(radius=coupling_radius, degree=10)
        Ru, Rv = Average(u, meshQ, cylinder), Average(v, meshQ, cylinder)
    else:
        Ru, Rv = Average(u, meshQ, None), Average(v, meshQ, None)

    # Line integral
    dx_ = Measure('dx', domain=meshQ)

    # Parameters
    k3, k1, gamma = map(Constant, (k3, k1, gamma))
    f3, f1 = Expression('x[0] + x[1]', degree=4), Constant(1)

    # We're building a 2x2 problem
    a = block_form(W, 2)
    a[0][0] = k3 * inner(grad(u), grad(v)) * dx + k3 * inner(u, v) * dx
    a[1][1] = k1 * inner(grad(p), grad(q)) * dx + k1 * inner(p, q) * dx

    m = block_form(W, 2)
    m[0][0] = gamma * inner(Ru, Rv) * dx_
    m[0][1] = -gamma * inner(p, Rv) * dx_
    m[1][0] = -gamma * inner(q, Ru) * dx_
    m[1][1] = gamma * inner(p, q) * dx_

    L = block_form(W, 1)
    L[0] = inner(f3, v) * dx
    L[1] = inner(f1, q) * dx

    A, b = map(ii_assemble, (a+m, L))

    return A, b, W

# --------------------------------------------------------------------


if __name__ == '__main__':
    import numpy as np
    import argparse, os
    import utils
    # Load mesh
    edge_f = get_mesh_neuron()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-gamma', type=float, default=1, help='Coupling strength')
    parser.add_argument('-dump', type=int, default=0, choices=(0, 1), help='Save matrices')
    parser.add_argument('-radius', type=float, default=1, help='Coupling radius')
    parser.add_argument('-outdir', type=str, default="./data/emi_3d1d/", help='Where to save matrices')
    parser.add_argument('-load_solution', type=str, default=None, help='Where to load solution from (.txt file)')
    args, _ = parser.parse_known_args()

    if args.dump:
        args.load_solution = None
    if args.load_solution is not None:
        assert os.path.exists(args.load_solution)
    not os.path.exists(args.outdir) and os.makedirs(args.outdir)

    def get_path(what, ext):
        template_path = f'{what}_gamma{args.gamma}.{ext}'
        return os.path.join(args.load_solution, template_path)

    # Parameters
    sigma3d, sigma1d = 3e0, 7e0  # conductivities in mS cm^-1 (from EMI book, Buccino paper)
    mc = 1  # membrane capacitance in microF cm^-2
    radius = args.radius  # radius (rho) of the averaging surface in micro m
    deltat_inv = args.gamma  # inverse of the time step, in s^-1 ( 1/dt )

    if radius > 0:
        gamma = deltat_inv * 2 * np.pi * radius * mc  # coupling parameter
        sigma1d = sigma1d * np.pi * radius**2  # scaled 1d conductivity
    else:
        gamma = deltat_inv * 2 * np.pi * mc  # assume radius = 1 micro meter
        sigma1d = sigma1d * np.pi

    # Get discrete system
    start_time = time.time()
    A, b, W = get_system(edge_f, k3=sigma3d, k1=sigma1d, gamma=gamma, coupling_radius=radius)
    # A = AD + gamma * M
    print("\n------------------ System setup and assembly time: ", time.time() - start_time, "\n")
    # load_solution = False
    if args.dump:
        utils.dump_system(A, b, W, folder=args.outdir)
    elif args.load_solution is not None:
        utils.print_red(f"Loading results from {args.load_solution}solution.txt...")
        Vunperm = (np.array(dof_to_vertex_map(W[0]), dtype='int32'), np.array(dof_to_vertex_map(W[1]), dtype='int32'))
        sol = np.loadtxt(args.load_solution+"solution.txt")
        utils.print_red("Loading done.")
        sol_size = int(sol[0])
        sol = sol[1:]
        xx = (sol[:W[0].dim()], sol[W[0].dim():sol_size])

        wh = ii_Function(W)
        for i, xxi in enumerate(xx):
            xxi = xxi[Vunperm[i]]
            wh[i].vector().set_local(xxi)
        utils.print_red(f"Saving results to directory {args.load_solution}...")
        File(get_path('uh0', 'pvd')) << wh[0]
        File(get_path('uh1', 'pvd')) << wh[1]
        utils.print_red("Saving done.")
    else:
        # alternative solver
        A_ = ii_convert(A)
        b_ = ii_convert(b)
        niters, wh, ksp_dt = utils.solve_haznics(A_, b_, W)



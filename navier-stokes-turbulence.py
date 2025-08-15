import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
from functools import partial
import os
import argparse
import time
import orbax.checkpoint as ocp
from jax.experimental import mesh_utils
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from typing import Callable
# import jax.profiler

jax.config.update("jax_enable_x64", True)


"""
Philip Mocz (2025), @pmocz

Simulate the 3D Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

    v_t + (v.nabla) v = nu * nabla^2 v - nabla P
    div(v) = 0

Example Usage:

python navier-stokes-turbulence.py --res 32

"""

# Setup parameters (user-controlled)
parser = argparse.ArgumentParser(description="3D Navier-Stokes Simulation")
parser.add_argument("--res", type=int, default=64, help="Grid size (default: 64)")
parser.add_argument(
    "--rk4",
    action="store_true",
    help="Enable 4th-order Runge-Kutta time integration (default: False, use backwards Euler by default)",
)
parser.add_argument(
    "--cpu-only",
    action="store_true",
    help="Use CPU only (default: False, use GPU)",
)
args = parser.parse_args()

# Setup distributed computing
if args.cpu_only:
    flags = os.environ.get("XLA_FLAGS", "")
    flags += " --xla_force_host_platform_device_count=1"  # change to, e.g., 8 for testing sharding virtually
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["XLA_FLAGS"] = flags
    print("Using CPU only mode")
else:
    jax.distributed.initialize()
    if jax.process_index() == 0:
        print("Using GPU distributed mode")

# Create mesh and sharding for distributed computation
n_devices = jax.device_count()
devices = mesh_utils.create_device_mesh((n_devices,))
mesh = Mesh(devices, axis_names=("gpus",))
sharding = NamedSharding(mesh, PartitionSpec(None, "gpus"))

if jax.process_index() == 0:
    for env_var in [
        "SLURM_JOB_ID",
        "SLURM_NTASKS",
        "SLURM_NODELIST",
        "SLURM_STEP_NODELIST",
        "SLURM_STEP_GPUS",
        "SLURM_GPUS",
    ]:
        print(f"{env_var}: {os.getenv(env_var, '')}")
    print("Total number of processes: ", jax.process_count())
    print("Total number of devices: ", jax.device_count())
    print("List of devices: ", jax.devices())
    print("Number of devices on this process: ", jax.local_device_count())


def fft_partitioner(
    fft_func: Callable[[jax.Array], jax.Array],
    partition_spec: PartitionSpec,
):
    @custom_partitioning
    def func(x):
        return fft_func(x)

    def supported_sharding(sharding, shape):
        return NamedSharding(sharding.mesh, partition_spec)

    def partition(mesh, arg_shapes, result_shape):
        # result_shardings = jax.tree.map(lambda x: x.sharding, result_shape)
        arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
        return (
            mesh,
            fft_func,
            supported_sharding(arg_shardings[0], arg_shapes[0]),
            (supported_sharding(arg_shardings[0], arg_shapes[0]),),
        )

    def infer_sharding_from_operands(mesh, arg_shapes, shape):
        arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
        return supported_sharding(arg_shardings[0], arg_shapes[0])

    func.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule="i j k -> i j k",
    )
    return func


def _fft_XY(x):
    return jfft.fftn(x, axes=[0, 1])


def _fft_Z(x):
    return jfft.fft(x, axis=2)


def _ifft_XY(x):
    return jfft.ifftn(x, axes=[0, 1])


def _ifft_Z(x):
    return jfft.ifft(x, axis=2)


# fft_XY/ifft_XY: operate on 2D slices (axes [0,1])
# fft_Z/ifft_Z: operate on 1D slices (axis 2)
fft_XY = fft_partitioner(_fft_XY, PartitionSpec(None, None, "gpus"))
fft_Z = fft_partitioner(_fft_Z, PartitionSpec(None, "gpus"))
ifft_XY = fft_partitioner(_ifft_XY, PartitionSpec(None, None, "gpus"))
ifft_Z = fft_partitioner(_ifft_Z, PartitionSpec(None, "gpus"))


def xfft3d(x):
    x = fft_Z(x)
    x = fft_XY(x)
    return x


def ixfft3d(x):
    x = ifft_XY(x)
    x = ifft_Z(x)
    return x


# set up xfft (distributed version of jfft)
with mesh:
    xfft3d_jit = jax.jit(
        xfft3d,
        in_shardings=sharding,
        out_shardings=sharding,
    )

with mesh:
    ixfft3d_jit = jax.jit(
        ixfft3d,
        in_shardings=sharding,
        out_shardings=sharding,
    )

# if n_devices > 1, should be using xfft instead of jfft:
# my_fftn = jfft.fftn
# my_ifftn = jfft.ifftn
my_fftn = xfft3d_jit
my_ifftn = ixfft3d_jit


# Make a distributed meshgrid function
def xmeshgrid(x_lin):
    xx, yy, zz = jnp.meshgrid(x_lin, x_lin, x_lin, indexing="ij")
    return xx, yy, zz


xmeshgrid_jit = jax.jit(xmeshgrid, in_shardings=None, out_shardings=sharding)


def poisson_solve(rho, kSq_inv):
    """solve the Poisson equation, given source field rho"""
    V_hat = -(my_fftn(rho)) * kSq_inv
    V = jnp.real(my_ifftn(V_hat))
    return V


def diffusion_solve(v, dt, nu, kSq):
    """solve the diffusion equation over a timestep dt, given viscosity nu"""
    v_hat = (my_fftn(v)) / (1.0 + dt * nu * kSq)
    v_new = jnp.real(my_ifftn(v_hat))
    return v_new


def grad(v, kx, ky, kz):
    """return gradient of v"""
    v_hat = my_fftn(v)
    dvx = jnp.real(my_ifftn(1j * kx * v_hat))
    dvy = jnp.real(my_ifftn(1j * ky * v_hat))
    dvz = jnp.real(my_ifftn(1j * kz * v_hat))
    return dvx, dvy, dvz


def gradi(v, ki):
    """return gradient of v"""
    v_hat = my_fftn(v)
    dvi = jnp.real(my_ifftn(1j * ki * v_hat))
    return dvi


def div(vx, vy, vz, kx, ky, kz):
    """return divergence of (vx,vy,vz)"""
    vx_hat = my_fftn(vx)
    vy_hat = my_fftn(vy)
    vz_hat = my_fftn(vz)
    dvx_x = jnp.real(my_ifftn(1j * kx * vx_hat))
    dvy_y = jnp.real(my_ifftn(1j * ky * vy_hat))
    dvz_z = jnp.real(my_ifftn(1j * kz * vz_hat))
    div = dvx_x + dvy_y + dvz_z
    return div


def curl(vx, vy, vz, kx, ky, kz):
    """return curl of (vx,vy,vz) as (wx, wy, wz)"""
    # wx = dvy/dz - dvz/dy
    # wy = dvz/dx - dvx/dz
    # wz = dvx/dy - dvy/dx
    vx_hat = my_fftn(vx)
    vy_hat = my_fftn(vy)
    vz_hat = my_fftn(vz)
    dvy_z = jnp.real(my_ifftn(1j * kz * vy_hat))
    dvz_y = jnp.real(my_ifftn(1j * ky * vz_hat))
    dvz_x = jnp.real(my_ifftn(1j * kx * vz_hat))
    dvx_z = jnp.real(my_ifftn(1j * kz * vx_hat))
    dvx_y = jnp.real(my_ifftn(1j * ky * vx_hat))
    dvy_x = jnp.real(my_ifftn(1j * kx * vy_hat))
    wx = dvy_z - dvz_y
    wy = dvz_x - dvx_z
    wz = dvx_y - dvy_x
    return wx, wy, wz


def curl_spectral(vx_hat, vy_hat, vz_hat, kx, ky, kz):
    """Compute curl in spectral space"""
    wx_hat = 1j * (ky * vz_hat - kz * vy_hat)
    wy_hat = 1j * (kz * vx_hat - kx * vz_hat)
    wz_hat = 1j * (kx * vy_hat - ky * vx_hat)
    return wx_hat, wy_hat, wz_hat


def cross_product_spectral(vx, vy, vz, wx_hat, wy_hat, wz_hat):
    """Compute cross product (v × curl) in spectral space"""
    # Transform curl back to real space
    wx = jnp.real(my_ifftn(wx_hat))
    wy = jnp.real(my_ifftn(wy_hat))
    wz = jnp.real(my_ifftn(wz_hat))

    # Compute cross product in real space
    cross_x = vy * wz - vz * wy
    cross_y = vz * wx - vx * wz
    cross_z = vx * wy - vy * wx

    # Transform back to spectral space
    cross_x_hat = my_fftn(cross_x)
    cross_y_hat = my_fftn(cross_y)
    cross_z_hat = my_fftn(cross_z)

    return cross_x_hat, cross_y_hat, cross_z_hat


def get_ke(vx, vy, vz, dV):
    """Calculate the kinetic energy in the system = 0.5 * integral |v|^2 dV"""
    v2 = vx**2 + vy**2 + vz**2
    ke = 0.5 * jnp.sum(v2) * dV
    return ke


def apply_dealias(f, dealias):
    """apply 2/3 rule dealias to field f"""
    f_hat = dealias * my_fftn(f)
    f_filtered = jnp.real(my_ifftn(f_hat))
    return f_filtered


def compute_rhs_rk4(
    vx, vy, vz, vx_hat, vy_hat, vz_hat, nu, kx, ky, kz, kSq, kSq_inv, dealias
):
    """Compute right-hand side of Navier-Stokes equations for the RK4 solver"""

    # Compute curl in spectral space
    wx_hat, wy_hat, wz_hat = curl_spectral(vx_hat, vy_hat, vz_hat, kx, ky, kz)

    # Compute cross product (v × curl)
    rhs_x_hat, rhs_y_hat, rhs_z_hat = cross_product_spectral(
        vx, vy, vz, wx_hat, wy_hat, wz_hat
    )

    # Apply dealiasing
    rhs_x_hat = dealias * rhs_x_hat
    rhs_y_hat = dealias * rhs_y_hat
    rhs_z_hat = dealias * rhs_z_hat

    # Project to enforce incompressibility (pressure correction)
    # P_hat = sum(rhs_hat * k / k^2, axis=0)
    P_hat = (
        kx * kSq_inv * rhs_x_hat + ky * kSq_inv * rhs_y_hat + kz * kSq_inv * rhs_z_hat
    )

    # Subtract pressure gradient
    rhs_x_hat = rhs_x_hat - kx * P_hat
    rhs_y_hat = rhs_y_hat - ky * P_hat
    rhs_z_hat = rhs_z_hat - kz * P_hat

    # Add viscous term
    rhs_x_hat = rhs_x_hat - nu * kSq * vx_hat
    rhs_y_hat = rhs_y_hat - nu * kSq * vy_hat
    rhs_z_hat = rhs_z_hat - nu * kSq * vz_hat

    return rhs_x_hat, rhs_y_hat, rhs_z_hat


@partial(jax.jit, static_argnames=["dt", "Nt", "nu"])
def run_simulation_simple(vx, vy, vz, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias):
    """Run the full Navier-Stokes simulation"""

    def update(_, state):
        (vx, vy, vz) = state

        # Advection: rhs = -(v.grad)v
        rhs_x = -(vx * gradi(vx, kx) + vy * gradi(vx, ky) + vz * gradi(vx, kz))
        rhs_y = -(vx * gradi(vy, kx) + vy * gradi(vy, ky) + vz * gradi(vy, kz))
        rhs_z = -(vx * gradi(vz, kx) + vy * gradi(vz, ky) + vz * gradi(vz, kz))

        rhs_x = apply_dealias(rhs_x, dealias)
        rhs_y = apply_dealias(rhs_y, dealias)
        rhs_z = apply_dealias(rhs_z, dealias)

        vx += dt * rhs_x
        vy += dt * rhs_y
        vz += dt * rhs_z

        # Poisson solve for pressure
        div_rhs = div(rhs_x, rhs_y, rhs_z, kx, ky, kz)
        P = poisson_solve(div_rhs, kSq_inv)

        # Correction (to eliminate divergence component of velocity)
        vx -= dt * gradi(P, kx)
        vy -= dt * gradi(P, ky)
        vz -= dt * gradi(P, kz)

        # Diffusion solve
        vx = diffusion_solve(vx, dt, nu, kSq)
        vy = diffusion_solve(vy, dt, nu, kSq)
        vz = diffusion_solve(vz, dt, nu, kSq)

        return (vx, vy, vz)

    (vx, vy, vz) = jax.lax.fori_loop(0, Nt, update, (vx, vy, vz))

    return vx, vy, vz


@partial(jax.jit, static_argnames=["dt", "Nt", "nu"])
def run_simulation_rk4(vx, vy, vz, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias):
    """Run the full Navier-Stokes simulation using 4th-order Runge-Kutta"""

    # RK4 coefficients
    a = jnp.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
    b = jnp.array([0.5, 0.5, 1.0])

    def update(_, state):
        (vx, vy, vz) = state

        # Transform to spectral space
        vx_hat = my_fftn(vx)
        vy_hat = my_fftn(vy)
        vz_hat = my_fftn(vz)

        # Store initial values
        vx_hat0 = vx_hat
        vy_hat0 = vy_hat
        vz_hat0 = vz_hat

        # Initialize accumulation
        vx_hat1 = jnp.zeros_like(vx_hat)
        vy_hat1 = jnp.zeros_like(vy_hat)
        vz_hat1 = jnp.zeros_like(vz_hat)

        # RK4 stages
        for rk in range(4):
            # Transform back to real space if needed (for cross product)
            if rk > 0:
                vx_temp = jnp.real(my_ifftn(vx_hat))
                vy_temp = jnp.real(my_ifftn(vy_hat))
                vz_temp = jnp.real(my_ifftn(vz_hat))
            else:
                vx_temp = vx
                vy_temp = vy
                vz_temp = vz

            # Compute RHS
            rhs_x_hat, rhs_y_hat, rhs_z_hat = compute_rhs_rk4(
                vx_temp,
                vy_temp,
                vz_temp,
                vx_hat,
                vy_hat,
                vz_hat,
                nu,
                kx,
                ky,
                kz,
                kSq,
                kSq_inv,
                dealias,
            )

            # Update for next stage
            if rk < 3:
                vx_hat = vx_hat0 + b[rk] * dt * rhs_x_hat
                vy_hat = vy_hat0 + b[rk] * dt * rhs_y_hat
                vz_hat = vz_hat0 + b[rk] * dt * rhs_z_hat

            # Accumulate for final update
            vx_hat1 = vx_hat1 + a[rk] * dt * rhs_x_hat
            vy_hat1 = vy_hat1 + a[rk] * dt * rhs_y_hat
            vz_hat1 = vz_hat1 + a[rk] * dt * rhs_z_hat

        # Final update
        vx_hat = vx_hat0 + vx_hat1
        vy_hat = vy_hat0 + vy_hat1
        vz_hat = vz_hat0 + vz_hat1

        # Transform back to real space
        vx = jnp.real(my_ifftn(vx_hat))
        vy = jnp.real(my_ifftn(vy_hat))
        vz = jnp.real(my_ifftn(vz_hat))

        return (vx, vy, vz)

    (vx, vy, vz) = jax.lax.fori_loop(0, Nt, update, (vx, vy, vz))

    return vx, vy, vz


def run_simulation_and_save_checkpoints(
    vx, vy, vz, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias, out_folder
):
    """Run the full Navier-Stokes simulation and save 100 checkpoints"""

    path = os.path.join(os.getcwd(), out_folder)
    if jax.process_index() == 0:
        path = ocp.test_utils.erase_and_create_empty(os.getcwd() + "/" + out_folder)
        print(f"Saving checkpoints to {path}")
    async_checkpoint_manager = ocp.CheckpointManager(path)

    num_checkpoints = 100
    snap_interval = max(1, Nt // num_checkpoints)
    checkpoint_id = 0
    if jax.process_index() == 0:
        print("saving initial condition")
    state = {}
    state["vx"] = vx
    state["vy"] = vy
    state["vz"] = vz
    vx.block_until_ready()
    vy.block_until_ready()
    vz.block_until_ready()
    async_checkpoint_manager.save(checkpoint_id, args=ocp.args.StandardSave(state))
    async_checkpoint_manager.wait_until_finished()
    checkpoint_id += 1
    if jax.process_index() == 0:
        print("about to start simulation")
    time_start = time.time()
    for i in range(0, Nt, snap_interval):
        if jax.process_index() == 0:
            print(f"step {i} of {Nt}")
        steps = min(snap_interval, Nt - i)
        if args.rk4:
            vx, vy, vz = run_simulation_rk4(
                vx, vy, vz, dt, steps, nu, kx, ky, kz, kSq, kSq_inv, dealias
            )
        else:
            vx, vy, vz = run_simulation_simple(
                vx, vy, vz, dt, steps, nu, kx, ky, kz, kSq, kSq_inv, dealias
            )
        if jax.process_index() == 0:
            print("about to create checkpoint")
        state["vx"] = vx
        state["vy"] = vy
        state["vz"] = vz
        vx.block_until_ready()
        vy.block_until_ready()
        vz.block_until_ready()
        async_checkpoint_manager.save(checkpoint_id, args=ocp.args.StandardSave(state))
        async_checkpoint_manager.wait_until_finished()
        checkpoint_id += 1
        if jax.process_index() == 0:
            print(
                "estimated time remaining: {:.2f} minutes".format(
                    (time.time() - time_start)
                    * (Nt - (i + snap_interval))
                    / (i + snap_interval)
                    / 60.0
                )
            )

    return vx, vy, vz


def main():
    """3D Navier-Stokes Simulation"""

    N = args.res
    Nt = 32000
    dt = 0.001
    nu = 1.0 / 1600.0

    if jax.process_index() == 0:
        print(
            f"Running 3D Navier-Stokes simulation with N={N}, Nt={Nt}, dt={dt}, nu={nu}"
        )
        print(
            f"using {'4th-order Runge-Kutta' if args.rk4 else 'backwards Euler'} method"
        )

    # assert Nt * dt > 10.0, "Run simulation long enough for turbulence to develop!"

    # Domain [0,2*pi]^3
    L = 2.0 * jnp.pi
    # dx = L / N
    x_lin = jnp.linspace(0, L, num=N + 1)
    x_lin = x_lin[0:N]
    # xx, yy, zz = jnp.meshgrid(x_lin, x_lin, x_lin, indexing="ij")
    xx, yy, zz = xmeshgrid_jit(x_lin)
    if jax.process_index() == 0:
        print("meshgrid set up")

    # Apply sharding to meshgrid arrays
    # xx = jax.lax.with_sharding_constraint(xx, sharding)
    # yy = jax.lax.with_sharding_constraint(yy, sharding)
    # zz = jax.lax.with_sharding_constraint(zz, sharding)

    # Fourier Space Variables
    k_lin = (2.0 * jnp.pi) / L * jnp.arange(-N / 2, N / 2)
    kmax = jnp.max(k_lin)
    # kx, ky, kz = jnp.meshgrid(k_lin, k_lin, k_lin, indexing="ij")
    kx, ky, kz = xmeshgrid_jit(k_lin)
    kx = jfft.ifftshift(kx)
    ky = jfft.ifftshift(ky)
    kz = jfft.ifftshift(kz)
    kSq = kx**2 + ky**2 + kz**2
    kSq_inv = 1.0 / (kSq + (kSq == 0)) * (kSq != 0)
    if jax.process_index() == 0:
        print("spectral vars set up")

    # kx = jax.lax.with_sharding_constraint(kx, sharding)
    # ky = jax.lax.with_sharding_constraint(ky, sharding)
    # kz = jax.lax.with_sharding_constraint(kz, sharding)
    # kSq = jax.lax.with_sharding_constraint(kSq, sharding)
    # kSq_inv = jax.lax.with_sharding_constraint(kSq_inv, sharding)

    # dealias with the 2/3 rule
    dealias = (
        (jnp.abs(kx) < (2.0 / 3.0) * kmax)
        & (jnp.abs(ky) < (2.0 / 3.0) * kmax)
        & (jnp.abs(kz) < (2.0 / 3.0) * kmax)
    )
    if jax.process_index() == 0:
        print("dealias vars set up")

    # Initial Condition (simple vortex, divergence free)
    # vx = -jnp.cos(2.0 * jnp.pi * yy) * jnp.cos(2.0 * jnp.pi * zz)
    # vy = jnp.cos(2.0 * jnp.pi * xx) * jnp.cos(2.0 * jnp.pi * zz)
    # vz = jnp.cos(2.0 * jnp.pi * xx) * jnp.cos(2.0 * jnp.pi * yy)
    # Ax = jnp.cos(2.0 * jnp.pi * xx) * jnp.sin(2.0 * jnp.pi * yy) / (2.0 * jnp.pi)
    # Ay = -jnp.cos(2.0 * jnp.pi * yy) * jnp.sin(2.0 * jnp.pi * zz) / (2.0 * jnp.pi)
    # Az = jnp.cos(2.0 * jnp.pi * zz) * jnp.sin(2.0 * jnp.pi * xx) / (2.0 * jnp.pi)
    # del xx, yy, zz  # clear meshgrid to save memory
    # vx, vy, vz = curl(Ax, Ay, Az, kx, ky, kz)
    # del Ax, Ay, Az  # clear initial condition variables to save memory

    # Taylor-Green vortex initial condition
    vx = jnp.sin(xx) * jnp.cos(yy) * jnp.cos(zz)
    vy = -jnp.cos(xx) * jnp.sin(yy) * jnp.cos(zz)
    vz = jnp.zeros_like(vx)

    if jax.process_index() == 0:
        print("vz:")
        print(f"  Shape: {vz.shape}")
        print(f"  Sharding: {vz.sharding}")

    del xx, yy, zz  # clear meshgrid to save memory

    # check the divergence of the initial condition
    div_v = div(vx, vy, vz, kx, ky, kz)
    div_error = jnp.max(jnp.abs(div_v))
    assert div_error < 1e-8, f"Initial divergence is too large: {div_error:.6e}"
    del div_v

    # Run the simulation
    out_folder = f"checkpoints{N}_rk4" if args.rk4 else f"checkpoints{N}"
    if jax.process_index() == 0:
        print(f"starting simulation with output folder: {out_folder}")
    start_time = time.time()
    state = run_simulation_and_save_checkpoints(
        vx,
        vy,
        vz,
        dt,
        Nt,
        nu,
        kx,
        ky,
        kz,
        kSq,
        kSq_inv,
        dealias,
        out_folder,
    )
    jax.block_until_ready(state)
    # jax.profiler.save_device_memory_profile("memory.prof") # for memory profiling
    end_time = time.time()
    if jax.process_index() == 0:
        print(f"Simulation N={N} completed in {end_time - start_time:.6f} seconds")
        with open(os.path.join(out_folder, "timing.txt"), "w") as f:
            f.write(f"{end_time - start_time:.6f} seconds\n")


if __name__ == "__main__":
    main()

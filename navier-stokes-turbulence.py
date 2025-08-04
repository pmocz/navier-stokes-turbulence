import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
from functools import partial
import os
import argparse
import time
import orbax.checkpoint as ocp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
# import jax.profiler

jax.config.update("jax_enable_x64", True)

"""
Philip Mocz (2025), @pmocz

Simulate the 3D Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v - nabla P
div(v) = 0

RK4 method uses:
- 4th-order Runge-Kutta time integration
- Spectral method in Fourier space
- Projection method for incompressibility
- 2/3 rule dealiasing

Example Usage:

python navier-stokes-turbulence.py --res 64

"""

# Setup parameters (user-controlled)
parser = argparse.ArgumentParser(description="3D Navier-Stokes Simulation")
parser.add_argument("--res", type=int, default=64, help="Grid size (default: 64)")
parser.add_argument(
    "--no-rk4",
    action="store_true",
    help="Disable 4th-order Runge-Kutta time integration (default: False, use RK4 by default)",
)
parser.add_argument(
    "--cpu-only",
    action="store_true",
    help="Use CPU only (default: False, use GPU if available)",
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
mesh = Mesh(mesh_utils.create_device_mesh((n_devices, 1, 1)), ("x", "y", "z"))
sharding = NamedSharding(mesh, PartitionSpec("x", "y", "z"))

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


def poisson_solve(rho, kSq_inv):
    """solve the Poisson equation, given source field rho"""
    V_hat = -(jfft.fftn(rho)) * kSq_inv
    V = jnp.real(jfft.ifftn(V_hat))
    return V


def diffusion_solve(v, dt, nu, kSq):
    """solve the diffusion equation over a timestep dt, given viscosity nu"""
    v_hat = (jfft.fftn(v)) / (1.0 + dt * nu * kSq)
    v = jnp.real(jfft.ifftn(v_hat))
    return v


def grad(v, kx, ky, kz):
    """return gradient of v"""
    v_hat = jfft.fftn(v)
    dvx = jnp.real(jfft.ifftn(1j * kx * v_hat))
    dvy = jnp.real(jfft.ifftn(1j * ky * v_hat))
    dvz = jnp.real(jfft.ifftn(1j * kz * v_hat))
    return dvx, dvy, dvz


def div(vx, vy, vz, kx, ky, kz):
    """return divergence of (vx,vy,vz)"""
    dvx_x = jnp.real(jfft.ifftn(1j * kx * jfft.fftn(vx)))
    dvy_y = jnp.real(jfft.ifftn(1j * ky * jfft.fftn(vy)))
    dvz_z = jnp.real(jfft.ifftn(1j * kz * jfft.fftn(vz)))
    return dvx_x + dvy_y + dvz_z


def curl(vx, vy, vz, kx, ky, kz):
    """return curl of (vx,vy,vz) as (wx, wy, wz)"""
    # wx = dvy/dz - dvz/dy
    # wy = dvz/dx - dvx/dz
    # wz = dvx/dy - dvy/dx
    dvy_z = jnp.real(jfft.ifftn(1j * kz * jfft.fftn(vy)))
    dvz_y = jnp.real(jfft.ifftn(1j * ky * jfft.fftn(vz)))
    dvz_x = jnp.real(jfft.ifftn(1j * kx * jfft.fftn(vz)))
    dvx_z = jnp.real(jfft.ifftn(1j * kz * jfft.fftn(vx)))
    dvx_y = jnp.real(jfft.ifftn(1j * ky * jfft.fftn(vx)))
    dvy_x = jnp.real(jfft.ifftn(1j * kx * jfft.fftn(vy)))
    wx = dvy_z - dvz_y
    wy = dvz_x - dvx_z
    wz = dvx_y - dvy_x
    return wx, wy, wz


def curl_spectral(vx_hat, vy_hat, vz_hat, kx, ky, kz):
    """Compute curl in spectral space"""
    # wx = dvy/dz - dvz/dy
    # wy = dvz/dx - dvx/dz
    # wz = dvx/dy - dvy/dx
    wx_hat = 1j * (ky * vz_hat - kz * vy_hat)
    wy_hat = 1j * (kz * vx_hat - kx * vz_hat)
    wz_hat = 1j * (kx * vy_hat - ky * vx_hat)
    return wx_hat, wy_hat, wz_hat


def cross_product_spectral(vx, vy, vz, wx_hat, wy_hat, wz_hat, kx, ky, kz):
    """Compute cross product (v × curl) in spectral space"""
    # Transform curl back to real space
    wx = jnp.real(jfft.ifftn(wx_hat))
    wy = jnp.real(jfft.ifftn(wy_hat))
    wz = jnp.real(jfft.ifftn(wz_hat))

    # Compute cross product in real space
    cross_x = vy * wz - vz * wy
    cross_y = vz * wx - vx * wz
    cross_z = vx * wy - vy * wx

    # Transform back to spectral space
    cross_x_hat = jfft.fftn(cross_x)
    cross_y_hat = jfft.fftn(cross_y)
    cross_z_hat = jfft.fftn(cross_z)

    return cross_x_hat, cross_y_hat, cross_z_hat


def get_ke(vx, vy, vz, dV):
    """Calculate the kinetic energy in the system = 0.5 * integral |v|^2 dV"""
    v2 = vx**2 + vy**2 + vz**2
    ke = 0.5 * jnp.sum(v2) * dV
    return ke


def apply_dealias(f, dealias):
    """apply 2/3 rule dealias to field f"""
    f_hat = dealias * jfft.fftn(f)
    return jnp.real(jfft.ifftn(f_hat))


def compute_rhs_rk4(
    vx, vy, vz, vx_hat, vy_hat, vz_hat, nu, kx, ky, kz, kSq, kSq_inv, dealias
):
    """Compute right-hand side of Navier-Stokes equations for the RK4 solver"""

    # Compute curl in spectral space
    wx_hat, wy_hat, wz_hat = curl_spectral(vx_hat, vy_hat, vz_hat, kx, ky, kz)

    # Compute cross product (v × curl)
    rhs_x_hat, rhs_y_hat, rhs_z_hat = cross_product_spectral(
        vx, vy, vz, wx_hat, wy_hat, wz_hat, kx, ky, kz
    )

    # Apply dealiasing
    rhs_x_hat = dealias * rhs_x_hat
    rhs_y_hat = dealias * rhs_y_hat
    rhs_z_hat = dealias * rhs_z_hat

    # Project to enforce incompressibility (pressure correction)
    # P_hat = sum(rhs_hat * k / k^2, axis=0)
    k_over_kSq = jnp.stack([kx * kSq_inv, ky * kSq_inv, kz * kSq_inv], axis=0)
    P_hat = jnp.sum(
        jnp.stack([rhs_x_hat, rhs_y_hat, rhs_z_hat], axis=0) * k_over_kSq, axis=0
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
        dvx_x, dvx_y, dvx_z = grad(vx, kx, ky, kz)
        dvy_x, dvy_y, dvy_z = grad(vy, kx, ky, kz)
        dvz_x, dvz_y, dvz_z = grad(vz, kx, ky, kz)

        rhs_x = -(vx * dvx_x + vy * dvx_y + vz * dvx_z)
        rhs_y = -(vx * dvy_x + vy * dvy_y + vz * dvy_z)
        rhs_z = -(vx * dvz_x + vy * dvz_y + vz * dvz_z)

        rhs_x = apply_dealias(rhs_x, dealias)
        rhs_y = apply_dealias(rhs_y, dealias)
        rhs_z = apply_dealias(rhs_z, dealias)

        vx += dt * rhs_x
        vy += dt * rhs_y
        vz += dt * rhs_z

        # Poisson solve for pressure
        div_rhs = div(rhs_x, rhs_y, rhs_z, kx, ky, kz)
        P = poisson_solve(div_rhs, kSq_inv)
        dPx, dPy, dPz = grad(P, kx, ky, kz)

        # Correction (to eliminate divergence component of velocity)
        vx -= dt * dPx
        vy -= dt * dPy
        vz -= dt * dPz

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
        vx_hat = jfft.fftn(vx)
        vy_hat = jfft.fftn(vy)
        vz_hat = jfft.fftn(vz)

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
                vx_temp = jnp.real(jfft.ifftn(vx_hat))
                vy_temp = jnp.real(jfft.ifftn(vy_hat))
                vz_temp = jnp.real(jfft.ifftn(vz_hat))
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
        vx = jnp.real(jfft.ifftn(vx_hat))
        vy = jnp.real(jfft.ifftn(vy_hat))
        vz = jnp.real(jfft.ifftn(vz_hat))

        return (vx, vy, vz)

    (vx, vy, vz) = jax.lax.fori_loop(0, Nt, update, (vx, vy, vz))

    return vx, vy, vz


def run_simulation_and_save_checkpoints(
    vx, vy, vz, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias, out_folder
):
    """Run the full Navier-Stokes simulation and save 100 checkpoints"""

    # path = ocp.test_utils.erase_and_create_empty(os.getcwd() + "/" + folder_name)
    path = os.path.join(os.getcwd(), out_folder)
    output_is_setup = False
    if jax.process_index() == 0:
        if os.path.exists(path):
            os.rmdir(path)
        os.makedirs(path)
        print(f"Saving checkpoints to {path}")
        output_is_setup = True
    jax.block_until_ready(output_is_setup)
    async_checkpoint_manager = ocp.CheckpointManager(path)

    num_checkpoints = 100
    snap_interval = max(1, Nt // num_checkpoints)
    checkpoint_id = 0
    time_start = time.time()
    for i in range(0, Nt, snap_interval):
        steps = min(snap_interval, Nt - i)
        if args.no_rk4:
            vx, vy, vz = run_simulation_simple(
                vx, vy, vz, dt, steps, nu, kx, ky, kz, kSq, kSq_inv, dealias
            )
        else:
            vx, vy, vz = run_simulation_rk4(
                vx, vy, vz, dt, steps, nu, kx, ky, kz, kSq, kSq_inv, dealias
            )
        state = {}
        state["vx"] = vx
        state["vy"] = vy
        state["vz"] = vz
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
            f"using {'4th-order Runge-Kutta' if not args.no_rk4 else 'backwards Euler'} method"
        )

    # assert Nt * dt > 10.0, "Run simulation long enough for turbulence to develop!"

    # Domain [0,1]^3
    L = 2.0 * jnp.pi
    # dx = L / N
    xlin = jnp.linspace(0, L, num=N + 1)
    xlin = xlin[0:N]
    xx, yy, zz = jnp.meshgrid(xlin, xlin, xlin, indexing="ij")

    # Apply sharding to meshgrid arrays
    xx = jax.lax.with_sharding_constraint(xx, sharding)
    yy = jax.lax.with_sharding_constraint(yy, sharding)
    zz = jax.lax.with_sharding_constraint(zz, sharding)

    # Fourier Space Variables
    klin = 2.0 * jnp.pi / L * jnp.arange(-N / 2, N / 2)
    kmax = jnp.max(klin)
    kx, ky, kz = jnp.meshgrid(klin, klin, klin, indexing="ij")
    kx = jfft.ifftshift(kx)
    ky = jfft.ifftshift(ky)
    kz = jfft.ifftshift(kz)
    kSq = kx**2 + ky**2 + kz**2
    kSq_inv = 1.0 / kSq
    kSq_inv = kSq_inv.at[kSq == 0].set(1.0)

    kx = jax.lax.with_sharding_constraint(kx, sharding)
    ky = jax.lax.with_sharding_constraint(ky, sharding)
    kz = jax.lax.with_sharding_constraint(kz, sharding)
    kSq = jax.lax.with_sharding_constraint(kSq, sharding)
    kSq_inv = jax.lax.with_sharding_constraint(kSq_inv, sharding)

    # dealias with the 2/3 rule
    dealias = (
        (jnp.abs(kx) < (2.0 / 3.0) * kmax)
        & (jnp.abs(ky) < (2.0 / 3.0) * kmax)
        & (jnp.abs(kz) < (2.0 / 3.0) * kmax)
    )

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
        print("vx:")
        print(f"  Shape: {vx.shape}")
        print(f"  Sharding: {vx.sharding}")

    del xx, yy, zz  # clear meshgrid to save memory

    # check the divergence of the initial condition
    div_v = div(vx, vy, vz, kx, ky, kz)
    div_error = jnp.max(jnp.abs(div_v))
    assert div_error < 1e-8, f"Initial divergence is too large: {div_error:.6e}"
    del div_v

    # Run the simulation
    out_folder = f"checkpoints{N}" if not args.no_rk4 else f"checkpoints{N}_simple"
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
        print(f"Simulation completed in {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    main()

import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
from functools import partial
import os
import argparse
import time
import orbax.checkpoint as ocp
# import jax.profiler

jax.config.update("jax_enable_x64", True)

"""
Philip Mocz (2025), @pmocz

Simulate the 3D Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v - nabla P
div(v) = 0

Example Usage:

python navier-stokes-turbulence.py --res 64

"""

# Setup parameters (user-controlled)
parser = argparse.ArgumentParser(description="3D Navier-Stokes Simulation")
parser.add_argument("--res", type=int, default=64, help="Grid size (default: 64)")
args = parser.parse_args()


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


def get_ke(vx, vy, vz, dV):
    """Calculate the kinetic energy in the system = 0.5 * integral |v|^2 dV"""
    v2 = vx**2 + vy**2 + vz**2
    ke = 0.5 * jnp.sum(v2) * dV
    return ke


def apply_dealias(f, dealias):
    """apply 2/3 rule dealias to field f"""
    f_hat = dealias * jfft.fftn(f)
    return jnp.real(jfft.ifftn(f_hat))


@partial(jax.jit, static_argnames=["dt", "Nt", "nu"])
def run_simulation(vx, vy, vz, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias):
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


def run_simulation_and_save_checkpoints(
    vx, vy, vz, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias, folder_name
):
    """Run the full Navier-Stokes simulation and save 100 checkpoints"""

    path = ocp.test_utils.erase_and_create_empty(os.getcwd() + "/" + folder_name)
    async_checkpoint_manager = ocp.CheckpointManager(path)

    num_checkpoints = 100
    snap_interval = max(1, Nt // num_checkpoints)
    checkpoint_id = 0
    time_start = time.time()
    for i in range(0, Nt, snap_interval):
        steps = min(snap_interval, Nt - i)
        vx, vy, vz = run_simulation(
            vx, vy, vz, dt, steps, nu, kx, ky, kz, kSq, kSq_inv, dealias
        )
        state = {}
        state["vx"] = vx
        state["vy"] = vy
        state["vz"] = vz
        async_checkpoint_manager.save(checkpoint_id, args=ocp.args.StandardSave(state))
        async_checkpoint_manager.wait_until_finished()
        checkpoint_id += 1
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

    print(jax.devices())
    N = args.res
    Nt = 32000
    dt = 0.001
    nu = 0.0005

    print(f"Running 3D Navier-Stokes simulation with N={N}, Nt={Nt}, dt={dt}, nu={nu}")

    assert Nt * dt > 10.0, "Run simulation long enough for turbulence to develop!"

    # Domain [0,1]^3
    L = 2.0 * jnp.pi
    # dx = L / N
    xlin = jnp.linspace(0, L, num=N + 1)
    xlin = xlin[0:N]
    xx, yy, zz = jnp.meshgrid(xlin, xlin, xlin, indexing="ij")

    # Fourier Space Variables
    klin = 2.0 * jnp.pi / L * jnp.arange(-N / 2, N / 2)
    kmax = jnp.max(klin)
    kx, ky, kz = jnp.meshgrid(klin, klin, klin, indexing="ij")
    kx = jnp.fft.ifftshift(kx)
    ky = jnp.fft.ifftshift(ky)
    kz = jnp.fft.ifftshift(kz)
    kSq = kx**2 + ky**2 + kz**2
    kSq_inv = 1.0 / kSq
    kSq_inv = kSq_inv.at[kSq == 0].set(1.0)

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
    del xx, yy, zz  # clear meshgrid to save memory

    # check the divergence of the initial condition
    div_v = div(vx, vy, vz, kx, ky, kz)
    div_error = jnp.max(jnp.abs(div_v))
    assert div_error < 1e-8, f"Initial divergence is too large: {div_error:.6e}"
    del div_v

    # Run the simulation
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
        f"checkpoints{N}",
    )
    jax.block_until_ready(state)
    # jax.profiler.save_device_memory_profile("memory.prof") # for memory profiling
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    main()

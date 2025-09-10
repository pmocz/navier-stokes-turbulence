import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import jaxdecomp as jd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import argparse
import time
import orbax.checkpoint as ocp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding

jax.config.update("jax_enable_x64", True)

"""
Philip Mocz (2025), @pmocz

Read and analyze the results of a 3D Navier-Stokes simulation.

Example Usage:

python analyze.py --res 32

"""

parser = argparse.ArgumentParser(description="Analyze Navier-Stokes Simulation")
parser.add_argument("--res", type=int, default=64, help="Grid size (default: 64)")
parser.add_argument("--show", action="store_true", help="Show plots interactively")
parser.add_argument(
    "--cpu", action="store_true", help="Use CPU only (default: False, use GPU)"
)
args = parser.parse_args()

# Set up distributed computing
if args.cpu:
    flags = os.environ.get("XLA_FLAGS", "")
    flags += " --xla_force_host_platform_device_count=8"  # change to, e.g., 8 for testing sharding virtually
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["XLA_FLAGS"] = flags
    print("Using CPU only mode")
else:
    jax.distributed.initialize()
    if jax.process_index() == 0:
        print("Using GPU distributed mode")

# Create mesh and sharding for distributed computation
n_devices = jax.device_count()
devices = mesh_utils.create_device_mesh((1, n_devices))
mesh = Mesh(devices, axis_names=("x", "y"))
sharding = NamedSharding(mesh, PartitionSpec("x", "y"))

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


# Make a distributed meshgrid function
def xmeshgrid(x_lin):
    xx, yy, zz = jnp.meshgrid(x_lin, x_lin, x_lin, indexing="ij")
    return xx, yy, zz


xmeshgrid_jit = jax.jit(xmeshgrid, in_shardings=None, out_shardings=sharding)


def xmeshgrid_transpose(x_lin):
    xx, yy, zz = jnp.meshgrid(x_lin, x_lin, x_lin, indexing="ij")
    xx = jnp.transpose(xx, (1, 2, 0))
    yy = jnp.transpose(yy, (1, 2, 0))
    zz = jnp.transpose(zz, (1, 2, 0))
    return xx, yy, zz


xmeshgrid_transpose_jit = jax.jit(
    xmeshgrid_transpose, in_shardings=None, out_shardings=sharding
)


# Make a distributed meshgrid function
def xzeros(N):
    return jnp.zeros((N, N, N))


xzeros_jit = jax.jit(
    xzeros, static_argnums=0, in_shardings=None, out_shardings=sharding
)


N = args.res
num_checkpoints = 100
skip = 10

path = os.path.join(os.path.dirname(__file__), f"checkpoints{N}")
async_checkpoint_manager = ocp.CheckpointManager(path)

# Fourier Space Variables
boxsize = 2.0 * jnp.pi  # Domain size
# k_lin = (2.0 * jnp.pi / boxsize) * jnp.arange(-N / 2, N / 2)
# since box size is 2*pi, make this integers to save memory
k_lin = jnp.arange(-N // 2, N // 2)
kmax = jnp.max(k_lin)
kx, ky, kz = xmeshgrid_transpose_jit(k_lin)
kx = jfft.ifftshift(kx)
ky = jfft.ifftshift(ky)
kz = jfft.ifftshift(kz)
# kSq = kx**2 + ky**2 + kz**2
# kSq_inv = 1.0 / (kSq + (kSq == 0)) * (kSq != 0)

colors = [(0, 0, 0), (1, 0, 0)]  # Black (0,0,0) to Red (1,0,0)

cmap_name = "BlackToRed"
custom_cmap = LinearSegmentedColormap.from_list(
    cmap_name, colors, N=num_checkpoints // skip + 1
)

# NOTE: jaxdecomp (jd) has pfft3d transpose the axis (X, Y, Z) --> (Y, Z, X), and pifft3d undo it
# so the fourier space variables (e.g. kx, ky, kz, kSq, dealias) all need to be transposed


def curl(vx, vy, vz, kx, ky, kz):
    """return curl of (vx,vy,vz) as (wx, wy, wz)"""
    vx_hat = jd.fft.pfft3d(vx)
    vy_hat = jd.fft.pfft3d(vy)
    vz_hat = jd.fft.pfft3d(vz)
    dvy_z = jnp.real(jd.fft.pifft3d(1j * kz * vy_hat))
    dvz_y = jnp.real(jd.fft.pifft3d(1j * ky * vz_hat))
    dvz_x = jnp.real(jd.fft.pifft3d(1j * kx * vz_hat))
    dvx_z = jnp.real(jd.fft.pifft3d(1j * kz * vx_hat))
    dvx_y = jnp.real(jd.fft.pifft3d(1j * ky * vx_hat))
    dvy_x = jnp.real(jd.fft.pifft3d(1j * kx * vy_hat))
    wx = dvy_z - dvz_y
    wy = dvz_x - dvx_z
    wz = dvx_y - dvy_x
    return wx, wy, wz


curl_jit = jax.jit(
    curl,
    in_shardings=(sharding, sharding, sharding, sharding, sharding, sharding),
    out_shardings=(sharding, sharding, sharding),
)


def radial_power_spectrum(data_cube, kx, ky, kz, boxsize):
    """
    Computes radially averaged power spectral density of data_cube (3D).
    data_cube: jnp.ndarray (3D, must be cubic)
    boxsize: float (physical size of box)
    Returns: Pf (radial power spectrum), k (wavenumbers), total_power
    """
    dim = data_cube.ndim
    N = data_cube.shape[0]
    dx = boxsize / N

    # Compute power spectrum
    data_cube_ft = jd.fft.pfft3d(data_cube)
    total_power = 0.5 * jnp.sum(jnp.abs(data_cube_ft) ** 2) / N**dim * dx**dim
    phi_k = 0.5 * jnp.abs(data_cube_ft) ** 2 / N**dim * dx**dim
    half_size = N // 2 + 1

    # Compute radially-averaged power spectrum
    # k_lin = jnp.arange(-N // 2, N // 2)
    # if dim == 2:
    #    kx, ky = jnp.meshgrid(k_lin, k_lin, indexing="ij")
    #    k_r = jnp.sqrt(kx**2 + ky**2)
    k_r = jnp.sqrt(kx**2 + ky**2 + kz**2)

    Pf, _ = jnp.histogram(
        k_r, range=(-0.5, half_size - 0.5), bins=half_size, weights=phi_k
    )
    norm, _ = jnp.histogram(k_r, range=(-0.5, half_size - 0.5), bins=half_size)
    Pf /= norm + (norm == 0)

    k = 2.0 * jnp.pi * jnp.arange(half_size) / boxsize
    dk = 2.0 * jnp.pi / boxsize

    Pf = Pf / dk**dim

    # Add geometrical factor
    # if dim == 2:
    #     Pf = Pf * 2.0 * jnp.pi * k
    Pf = Pf * 4.0 * jnp.pi * k**2

    return Pf, k, total_power


radial_power_spectrum_jit = jax.jit(
    radial_power_spectrum,
    in_shardings=(sharding, sharding, sharding, sharding, None),
    out_shardings=(None, None, None),
)


def get_slice(data_cube):
    """Get a 2D slice through the center of a 3D data cube."""
    slice = data_cube[:, :, data_cube.shape[2] // 2]
    return slice


get_slice_jit = jax.jit(get_slice, in_shardings=sharding, out_shardings=None)


def main():
    Pf_all = jnp.zeros((num_checkpoints // skip + 1, args.res // 2 + 1))
    target = {
        "vx": xzeros_jit(N),
        "vy": xzeros_jit(N),
        "vz": xzeros_jit(N),
    }

    start_time = time.time()
    for i in range(num_checkpoints // skip + 1):
        i_checkpoint = i * skip
        if jax.process_index() == 0:
            print(f"processing checkpoint {i_checkpoint}...")

        restored = async_checkpoint_manager.restore(
            i_checkpoint, args=ocp.args.StandardRestore(target)
        )

        vx = restored["vx"]
        vy = restored["vy"]
        vz = restored["vz"]
        if jax.process_index() == 0:
            print(f"calculating v...")
        v = jnp.sqrt(vx**2 + vy**2 + vz**2)
        v.block_until_ready()
        if jax.process_index() == 0:
            print(f"ready")

        if jax.process_index() == 0:
            print(f"calculating w...")
        wx, wy, wz = curl_jit(vx, vy, vz, kx, ky, kz)
        w = jnp.sqrt(wx**2 + wy**2 + wz**2)
        w.block_until_ready()
        if jax.process_index() == 0:
            print(f"ready")

        # Calculate the radial power spectrum
        Pf_vx, k, _ = radial_power_spectrum_jit(vx, kx, ky, kz, boxsize)
        if jax.process_index() == 0:
            print(f"calculated Pvx")
        Pf_vy, _, _ = radial_power_spectrum_jit(vy, kx, ky, kz, boxsize)
        Pf_vz, _, _ = radial_power_spectrum_jit(vz, kx, ky, kz, boxsize)
        if jax.process_index() == 0:
            print(f"calculated power spectrum")

        Pf = Pf_vx + Pf_vy + Pf_vz
        Pf_all = Pf_all.at[i].set(Pf)

        # get slices
        v_slice = jax.experimental.multihost_utils.process_allgather(
            v[:, :, v.shape[2] // 2]
        )
        w_slice = jax.experimental.multihost_utils.process_allgather(
            w[:, :, w.shape[2] // 2]
        )

        if jax.process_index() == 0:
            # Plot a slice of v as an image
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(v_slice, cmap="viridis")
            # plt.clim(-1, 1)
            plt.colorbar(label="velocity magnitude")
            plt.savefig(
                os.path.join(path, f"slice_v_{i_checkpoint}.png"),
                dpi=200,
                bbox_inches="tight",
            )
            if args.show:
                plt.show()
            plt.close(fig)

            # Plot a slice of w as an image
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(w_slice, cmap="viridis")
            plt.colorbar(label="velocity magnitude")
            plt.savefig(
                os.path.join(path, f"slice_w_{i_checkpoint}.png"),
                dpi=200,
                bbox_inches="tight",
            )
            if args.show:
                plt.show()
            plt.close(fig)

    if jax.process_index() == 0:
        # Save the results: Pf_all
        np.savez(
            os.path.join(path, "results_Pf.npz"),
            k=k,
            Pf_all=Pf_all,
        )

        # Plot the radial power spectrum
        fig = plt.figure(figsize=(8, 6))
        for i in range(num_checkpoints // skip + 1):
            plt.plot(
                k,
                Pf_all[i],
                color=custom_cmap(i),
                label=f"Checkpoint {i * skip}",
                alpha=0.5,
            )
        plt.xlabel("wavenumber (k)")
        plt.ylabel("velocity power spectrum")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim([1.0, 1024.0])
        plt.ylim([1.0e-4, 5.0e1])
        plt.savefig(
            os.path.join(path, "power_spectrum.png"), dpi=200, bbox_inches="tight"
        )
        if args.show:
            plt.show()
        plt.close(fig)

        end_time = time.time()
        print(f"Analysis completed in {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    main()

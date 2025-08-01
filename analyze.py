import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import argparse
import time
import orbax.checkpoint as ocp

jax.config.update("jax_enable_x64", True)

"""
Philip Mocz (2025), @pmocz

Read and analyze the results of a 3D Navier-Stokes simulation.

Example Usage:

python analyze.py --res 64

"""

parser = argparse.ArgumentParser(description="Analyze Navier-Stokes Simulation")
parser.add_argument("--res", type=int, default=64, help="Grid size (default: 64)")
parser.add_argument("--show", action="store_true", help="Show plots interactively")
args = parser.parse_args()

path = os.path.join(os.path.dirname(__file__), f"checkpoints{args.res}")
async_checkpoint_manager = ocp.CheckpointManager(path)

N = args.res
num_checkpoints = 100
skip = 20

# Fourier Space Variables
L = 2.0 * jnp.pi  # Domain size
klin = 2.0 * jnp.pi / L * jnp.arange(-N / 2, N / 2)
kmax = jnp.max(klin)
kx, ky, kz = jnp.meshgrid(klin, klin, klin, indexing="ij")
kx = jfft.ifftshift(kx)
ky = jfft.ifftshift(ky)
kz = jfft.ifftshift(kz)
kSq = kx**2 + ky**2 + kz**2
kSq_inv = 1.0 / kSq
kSq_inv = kSq_inv.at[kSq == 0].set(1.0)

colors = [(0, 0, 0), (1, 0, 0)]  # Black (0,0,0) to Red (1,0,0)

cmap_name = "BlackToRed"
custom_cmap = LinearSegmentedColormap.from_list(
    cmap_name, colors, N=num_checkpoints // skip
)


def curl(vx, vy, vz, kx, ky, kz):
    """return curl of (vx,vy,vz) as (wx, wy, wz)"""
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


def radial_power_spectrum(data_cube, Lbox):
    """
    Computes radially averaged power spectral density of data_cube (2D or 3D).
    data_cube: jnp.ndarray (2D or 3D, must be cubic)
    Lbox: float (physical size of box)
    Returns: Pf (radial power spectrum), k (wavenumbers), total_power
    """
    dim = data_cube.ndim
    N = data_cube.shape[0]
    dx = Lbox / N

    # Compute power spectrum
    data_cube_ft = jfft.fftshift(jfft.fftn(data_cube))
    total_power = 0.5 * jnp.sum(jnp.abs(data_cube_ft) ** 2) / N**dim * dx**dim
    phi_k = 0.5 * jnp.abs(data_cube_ft) ** 2 / N**dim * dx**dim
    half_size = N // 2 + 1

    # Compute radially-averaged power spectrum
    k_cartesian = jnp.arange(-N // 2, N // 2)
    if dim == 2:
        X, Y = jnp.meshgrid(k_cartesian, k_cartesian, indexing="ij")
        k_rho = jnp.sqrt(X**2 + Y**2)
    else:
        X, Y, Z = jnp.meshgrid(k_cartesian, k_cartesian, k_cartesian, indexing="ij")
        k_rho = jnp.sqrt(X**2 + Y**2 + Z**2)

    k_rho = jnp.round(k_rho).astype(jnp.int32)
    Pf = []
    for r in range(half_size):
        mask = k_rho == r
        vals = phi_k[mask]
        mean_val = jnp.nanmean(vals) if vals.size > 0 else 0.0
        Pf.append(mean_val)
    Pf = jnp.array(Pf)

    k = 2.0 * jnp.pi * jnp.arange(half_size) / Lbox
    dk = k[1] - k[0] if half_size > 1 else 1.0

    Pf = Pf / dk**dim

    # Add geometrical factor
    if dim == 2:
        Pf = Pf * 2.0 * jnp.pi * k
    else:
        Pf = Pf * 4.0 * jnp.pi * k**2

    return Pf, k, total_power


def main():
    # v_all = jnp.zeros((num_checkpoints, args.res, args.res, args.res))
    Pf_all = jnp.zeros((num_checkpoints // skip, args.res // 2 + 1))
    target = {
        "vx": jnp.zeros((N, N, N)),
        "vy": jnp.zeros((N, N, N)),
        "vz": jnp.zeros((N, N, N)),
    }

    start_time = time.time()
    for i in range(num_checkpoints // skip):
        print(f"processing checkpoint {i * skip}...")

        restored = async_checkpoint_manager.restore(
            i * skip, args=ocp.args.StandardRestore(target)
        )

        vx = restored["vx"]
        vy = restored["vy"]
        vz = restored["vz"]
        v = jnp.sqrt(vx**2 + vy**2 + vz**2)

        wx, wy, wz = curl(vx, vy, vz, kx, ky, kz)
        w = jnp.sqrt(wx**2 + wy**2 + wz**2)

        # v_all = v_all.at[i].set(v)

        # Calculate the radial power spectrum
        Pf_vx, k, total_power = radial_power_spectrum(vx, Lbox=L)
        Pf_vy, _, _ = radial_power_spectrum(vy, Lbox=L)
        Pf_vz, _, _ = radial_power_spectrum(vz, Lbox=L)

        Pf = Pf_vx + Pf_vy + Pf_vz
        Pf_all = Pf_all.at[i].set(Pf)

    # Save the results: v, k, Pf_all
    np.savez(
        os.path.join(path, "results.npz"),
        v=v,
        w=w,
        k=k,
        Pf_all=Pf_all,
        total_power=total_power,
    )

    # Plot a slice of v as an image
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(v[:, :, v.shape[2] // 2], cmap="viridis")
    # plt.imshow(vx[:, :, v.shape[2] // 2], cmap="viridis")
    # plt.clim(-1, 1)
    plt.colorbar(label="velocity magnitude")
    plt.savefig(os.path.join(path, "slice_v.png"), dpi=200, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)

    # Plot a slice of w as an image
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(w[:, :, w.shape[2] // 2], cmap="viridis")
    plt.colorbar(label="velocity magnitude")
    plt.savefig(os.path.join(path, "slice_w.png"), dpi=200, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)

    # Plot the radial power spectrum
    fig = plt.figure(figsize=(8, 6))
    for i in range(num_checkpoints // skip):
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
    plt.savefig(os.path.join(path, "power_spectrum.png"), dpi=200, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)

    end_time = time.time()
    print(f"Analysis completed in {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    main()

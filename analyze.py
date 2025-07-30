import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import matplotlib.pyplot as plt
import os
import argparse
import orbax.checkpoint as ocp

jax.config.update("jax_enable_x64", True)

"""
Philip Mocz (2025), @pmocz

Read and analyze the results of a 3D Navier-Stokes simulation.

Example Usage:

python analye.py --res 64

"""

parser = argparse.ArgumentParser(description="Analyze Navier-Stokes Simulation")
parser.add_argument("--res", type=int, default=32, help="Grid size (default: 32)")
args = parser.parse_args()

path = os.path.join(os.path.dirname(__file__), f"checkpoints{args.res}")
async_checkpoint_manager = ocp.CheckpointManager(path)

N = args.res

# Fourier Space Variables
L = 1.0  # Domain size
klin = 2.0 * jnp.pi / L * jnp.arange(-N / 2, N / 2)
kmax = jnp.max(klin)
kx, ky, kz = jnp.meshgrid(klin, klin, klin, indexing="ij")
kx = jnp.fft.ifftshift(kx)
ky = jnp.fft.ifftshift(ky)
kz = jnp.fft.ifftshift(kz)
kSq = kx**2 + ky**2 + kz**2
kSq_inv = 1.0 / kSq
kSq_inv = kSq_inv.at[kSq == 0].set(1.0)



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
    data_cube_ft = jnp.fft.fftshift(jnp.fft.fftn(data_cube))
    total_power = 0.5 * jnp.sum(jnp.abs(data_cube_ft)**2) / N**dim * dx**dim
    phi_k = 0.5 * jnp.abs(data_cube_ft)**2 / N**dim * dx**dim
    half_size = N // 2 + 1

    # Compute radially-averaged power spectrum
    k_cartesian = jnp.arange(-N//2, N//2)
    if dim == 2:
        X, Y = jnp.meshgrid(k_cartesian, k_cartesian, indexing='ij')
        k_rho = jnp.sqrt(X**2 + Y**2)
    else:
        X, Y, Z = jnp.meshgrid(k_cartesian, k_cartesian, k_cartesian, indexing='ij')
        k_rho = jnp.sqrt(X**2 + Y**2 + Z**2)

    k_rho = jnp.round(k_rho).astype(jnp.int32)
    Pf = []
    for r in range(half_size):
        mask = (k_rho == r)
        vals = phi_k[mask]
        mean_val = jnp.nanmean(vals) if vals.size > 0 else 0.0
        Pf.append(mean_val)
    Pf = jnp.array(Pf)

    k = 2 * jnp.pi * jnp.arange(half_size) / Lbox
    dk = k[1] - k[0] if half_size > 1 else 1.0

    Pf = Pf / dk**dim

    # Add geometrical factor
    if dim == 2:
        Pf = Pf * 2 * jnp.pi * k
    else:
        Pf = Pf * 4 * jnp.pi * k**2

    return Pf, k, total_power


def main():
    num_checkpoints = 100
    v_all = jnp.zeros((num_checkpoints, args.res, args.res, args.res))
    Pf_all = jnp.zeros((num_checkpoints, args.res // 2 + 1))

    for i in range(num_checkpoints):
        restored = async_checkpoint_manager.restore(i)

        vx = restored["vx"]
        vy = restored["vy"]
        vz = restored["vz"]
        v = jnp.sqrt(vx**2 + vy**2 + vz**2)
        
        v_all = v_all.at[i].set(v)

        # Calculate the radial power spectrum
        Pf_vx, k, total_power = radial_power_spectrum(vx, Lbox=L)
        Pf_vy, _, _ = radial_power_spectrum(vy, Lbox=L)
        Pf_vz, _, _ = radial_power_spectrum(vz, Lbox=L)

        Pf = (Pf_vx + Pf_vy + Pf_vz) / 3.0
        Pf_all = Pf_all.at[i].set(Pf)






    # Plot a slice of v as an image
    plt.imshow(v_all[-1][:, :, v_all.shape[2] // 2], cmap="viridis")
    plt.colorbar(label="Velocity Magnitude")
    plt.show()

    # Plot the radial power spectrum
    import matplotlib.cm as cm
    colors = cm.get_cmap('Reds', num_checkpoints)
    for i in range(num_checkpoints):
      plt.plot(k, Pf_all[i], color=colors(i), label=f"Checkpoint {i+1}", alpha=0.8)
    plt.xlabel("Wavenumber (k)")
    plt.ylabel("Power Spectrum")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    main()

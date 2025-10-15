import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import numpy as np
import matplotlib.pyplot as plt
import os
import orbax.checkpoint as ocp

jax.config.update("jax_enable_x64", True)

"""
Philip Mocz (2025), @pmocz

Read and analyze the results of a 3D Navier-Stokes simulation.

Example Usage:

python analyze_skewers.py

"""

resolutions = [32, 64, 128, 256, 512, 1024]
checkpoint_start = 20  # 20
checkpoint_end = 100
num_checkpoints = checkpoint_end - checkpoint_start
boxsize = 2.0 * jnp.pi
Nt = 32000
dt = 0.001


def main():
    # Load the data sets

    data_all = {}

    for N in resolutions:
        # if path exists
        path = os.path.join(os.path.dirname(__file__), f"checkpoints{N}")
        if not os.path.exists(path):
            continue
        print(f"Loading data from {path}")
        async_checkpoint_manager = ocp.CheckpointManager(path)

        data_vx = jnp.zeros((N, num_checkpoints))
        data_vy = jnp.zeros((N, num_checkpoints))
        data_vz = jnp.zeros((N, num_checkpoints))

        for i in range(checkpoint_start, checkpoint_end):
            target = {
                "vx": jnp.zeros((N, N, N)),
                "vy": jnp.zeros((N, N, N)),
                "vz": jnp.zeros((N, N, N)),
            }
            restored = async_checkpoint_manager.restore(
                i, args=ocp.args.StandardRestore(target)
            )

            vx = restored["vx"][N // 3, N // 3, :]
            vy = restored["vy"][N // 3, N // 3, :]
            vz = restored["vz"][N // 3, N // 3, :]

            data_vx = data_vx.at[:, i - checkpoint_start].set(vx)
            data_vy = data_vy.at[:, i - checkpoint_start].set(vy)
            data_vz = data_vz.at[:, i - checkpoint_start].set(vz)

        data_all[N] = {"vx": data_vx, "vy": data_vy, "vz": data_vz}

    # Make plots
    for N in resolutions:
        if N not in data_all:
            continue
        data_vx = data_all[N]["vx"]
        data_vy = data_all[N]["vy"]
        data_vz = data_all[N]["vz"]

        t = (checkpoint_start + jnp.arange(num_checkpoints)) * (
            Nt * dt / num_checkpoints
        )
        # x = jnp.arange(N) * (boxsize / N)

        # vmax is max of absolute value of data_vx, data_vy, data_vz
        vmax = jnp.max(jnp.abs(jnp.concatenate([data_vx, data_vy, data_vz])))
        vmin = -vmax

        plt.figure(figsize=(15, 5))
        plt.suptitle(f"1D skewers at resolution {N}^3")

        plt.subplot(1, 3, 1)
        plt.imshow(
            data_vx,
            extent=[t[0], t[-1], 0, boxsize],
            aspect="auto",
            origin="lower",
            cmap="RdBu",
            vmin=vmin,
            vmax=vmax,
        )
        plt.xlabel("time")
        plt.ylabel("space")

        plt.subplot(1, 3, 2)
        plt.imshow(
            data_vy,
            extent=[t[0], t[-1], 0, boxsize],
            aspect="auto",
            origin="lower",
            cmap="RdBu",
            vmin=vmin,
            vmax=vmax,
        )
        plt.xlabel("time")

        plt.subplot(1, 3, 3)
        plt.imshow(
            data_vz,
            extent=[t[0], t[-1], 0, boxsize],
            aspect="auto",
            origin="lower",
            cmap="RdBu",
            vmin=vmin,
            vmax=vmax,
        )
        plt.xlabel("time")
        plt.colorbar(label="velocity component")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"checkpoints{N}/skewers.png")
        plt.close()

    # Calculate and plot Spatial and Temporal Power Spectra
    for N in resolutions:
        if N not in data_all:
            continue
        data_vx = data_all[N]["vx"]
        data_vy = data_all[N]["vy"]
        data_vz = data_all[N]["vz"]

        # Spatial Power Spectrum (averaged over time)
        kx = jfft.fftfreq(N, d=boxsize / N) * 2 * jnp.pi
        kx = jnp.fft.fftshift(kx)

        ps_vx_space = jnp.abs(jfft.fftshift(jfft.fft(data_vx, axis=0), axes=0)) ** 2
        ps_vy_space = jnp.abs(jfft.fftshift(jfft.fft(data_vy, axis=0), axes=0)) ** 2
        ps_vz_space = jnp.abs(jfft.fftshift(jfft.fft(data_vz, axis=0), axes=0)) ** 2

        ps_space = (ps_vx_space + ps_vy_space + ps_vz_space) / 3.0
        ps_space_mean = jnp.mean(ps_space, axis=1)

        plt.figure()
        plt.loglog(kx[kx > 0], ps_space_mean[kx > 0], label=f"res={N}^3")
        # -5/3 reference line
        k_ref = kx[kx > 0]
        ref = ps_space_mean[kx > 0][0] * (k_ref / k_ref[0]) ** (-5 / 3)
        plt.loglog(k_ref, ref, "k--", label=r"$k^{-5/3}$")
        plt.legend()
        plt.ylim(1e-2, 1e1)
        plt.xlabel("k")
        plt.ylabel("power spectrum")
        plt.title(f"Spatial Power Spectrum")
        plt.savefig(f"skewer_pspec_spatial.png")
        plt.close()

        # Temporal Power Spectrum (averaged over space)
        omega = (
            jfft.fftfreq(num_checkpoints, d=(Nt * dt / num_checkpoints)) * 2 * jnp.pi
        )
        omega = jnp.fft.fftshift(omega)

        ps_vx_time = jnp.abs(jfft.fftshift(jfft.fft(data_vx, axis=1), axes=1)) ** 2
        ps_vy_time = jnp.abs(jfft.fftshift(jfft.fft(data_vy, axis=1), axes=1)) ** 2
        ps_vz_time = jnp.abs(jfft.fftshift(jfft.fft(data_vz, axis=1), axes=1)) ** 2

        ps_time = (ps_vx_time + ps_vy_time + ps_vz_time) / 3.0
        ps_time_mean = jnp.mean(ps_time, axis=0)

        plt.figure()
        plt.loglog(omega[omega > 0], ps_time_mean[omega > 0], label=f"res={N}^3")
        # -5/3 reference line
        omega_ref = omega[omega > 0]
        ref = ps_time_mean[omega > 0][0] * (omega_ref / omega_ref[0]) ** (-5 / 3)
        plt.loglog(omega_ref, ref, "k--", label=r"$\omega^{-5/3}$")
        plt.legend()
        plt.ylim(1e-1, 1e2)
        plt.xlabel("ω")
        plt.ylabel("power spectrum")
        plt.title(f"Temporal Power Spectrum")
        plt.savefig(f"skewer_pspec_temporal.png")
        plt.close()


if __name__ == "__main__":
    main()

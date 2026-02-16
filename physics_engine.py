# physics_engine.py


from typing import Tuple, Optional
import numpy as np

from math_core import (
    poisson_solve_fft,
    grad_x,
    grad_y,
    laplacian,
    lowpass_filter_fft,
)

# constants
OMEGA = 7.2921150e-5
R_EARTH = 6_371_000.0


def generate_centered_latlon(lat0_deg, lon0_deg, nx, ny, dx_m, dy_m):
    lat0_rad = np.deg2rad(lat0_deg)
    m_per_deg_lat = 111000.0
    m_per_deg_lon = 111000.0 * np.cos(lat0_rad)

    dx_deg = dx_m / m_per_deg_lon
    dy_deg = dy_m / m_per_deg_lat

    ix = np.arange(nx) - (nx - 1) / 2.0
    iy = np.arange(ny) - (ny - 1) / 2.0

    lon1d = lon0_deg + ix * dx_deg
    lat1d = lat0_deg + iy * dy_deg

    return np.meshgrid(lon1d, lat1d)


def _bilinear_interp_2d(arr, src_i, src_j):
    ny, nx = arr.shape
    src_i = np.clip(src_i, 0.0, ny - 1.0)
    src_j = np.clip(src_j, 0.0, nx - 1.0)

    i0 = np.floor(src_i).astype(int)
    j0 = np.floor(src_j).astype(int)
    i1 = np.minimum(i0 + 1, ny - 1)
    j1 = np.minimum(j0 + 1, nx - 1)

    wi = src_i - i0
    wj = src_j - j0

    return (
            (1 - wi) * (1 - wj) * arr[i0, j0]
            + (1 - wi) * wj * arr[i0, j1]
            + wi * (1 - wj) * arr[i1, j0]
            + wi * wj * arr[i1, j1]
    )


class PhysicsEngine:
    def __init__(
            self,
            lat0=25.0,
            lon0=-85.0,
            nx=128,
            ny=128,
            dx_m=4000.0,
            dy_m=4000.0,
            dt=60.0,
            nu=200.0,
    ):
        # domain
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = float(dx_m)
        self.dy = float(dy_m)
        self.dt = float(dt)
        self.nu = float(nu)

        # grid
        self.lon, self.lat = generate_centered_latlon(
            self.lat0, self.lon0, self.nx, self.ny, self.dx, self.dy
        )

        lat_rad = np.deg2rad(self.lat)
        self.X = (self.lon - self.lon0) * (111000.0 * np.cos(lat_rad))
        self.Y = (self.lat - self.lat0) * 111000.0

        # state
        self.zeta = np.zeros((self.ny, self.nx))
        self.psi = np.zeros_like(self.zeta)
        self.u = np.zeros_like(self.zeta)
        self.v = np.zeros_like(self.zeta)

        # steering
        self.u_steer = 0.0
        self.v_steer = 0.0
        self.u_bg = None
        self.v_bg = None

        # injection targets
        self._zeta_target = None
        self._rm_target = None
        self._r_grid = None

        # beta
        lat_center = float(np.mean(self.lat))
        self.beta = (2 * OMEGA * np.cos(np.deg2rad(lat_center))) / R_EARTH

        # tunable controls
        self.max_resolved_velocity = 400.0
        self.max_tendency = 20.0
        self.max_zeta = 5e-2

        self.beta_weight = 0.08
        self.env_weight = 1.0 - self.beta_weight

        self.diffusion_enabled = True
        self.enable_intensification = False
        self.intensification_alpha = 1e-4

        self.semi_lagrangian = True

        self.time = 0.0

    def set_background_wind(self, u_field=None, v_field=None):
        if u_field is None:
            self.u_bg = None
        else:
            self.u_bg = np.asarray(u_field, dtype=float)

        if v_field is None:
            self.v_bg = None
        else:
            self.v_bg = np.asarray(v_field, dtype=float)

    def clear_background_wind(self):
        self.u_bg = None
        self.v_bg = None



    def compute_environmental_steering(self, lat_deg, lon_deg=None):

        lat = float(lat_deg)
        lat_rad = np.deg2rad(lat)

        OMEGA = 7.2921150e-5
        R_EARTH = 6.371e6


        #ZONAL MEAN FLOW(m/s)


        abs_lat = abs(lat)

        if abs_lat < 20.0:
            # Tropical easterlies
            u_mean = -8.0 * np.cos(lat_rad)

        elif abs_lat < 35.0:
            # Subtropical ridge transition
            frac = (abs_lat - 20.0) / 15.0
            u_mean = -4.0 + 7.0 * frac

        else:
            # Midlatitude westerlies
            u_mean = 10.0 + 6.0 * np.sin(lat_rad) ** 2


        # Poleward drift from Hadley + baroclinic turning

        if abs_lat < 25.0:
            v_mean = 1.2 * np.sin(lat_rad)
        else:
            v_mean = 3.5 * np.sin(lat_rad)


        # BETA-PLANE DRIFT (m/s) Causes northwestward drift and recurvature in northern hemisphere

        beta = 2.0 * OMEGA * np.cos(lat_rad) / R_EARTH

        # Characteristic TC length scale (meters)
        L = 400_000.0

        u_beta = -0.15 * beta * L * L
        v_beta = 0.10 * beta * L * L * np.sign(lat)


        # (Rossby waves in theory), back up case if CNN doesnt work

        if lon_deg is not None:
            lon_rad = np.deg2rad(float(lon_deg))
            u_wave = 2.0 * np.sin(2.0 * lon_rad) * np.cos(lat_rad)
            v_wave = 1.5 * np.cos(2.0 * lon_rad) * np.sin(lat_rad)
        else:
            u_wave = 0.0
            v_wave = 0.0

        u = u_mean + u_beta + u_wave
        v = v_mean + v_beta + v_wave


        u = float(np.clip(u, -30.0, 30.0))
        v = float(np.clip(v, -25.0, 25.0))

        return u, v

    def get_speed_field(self):
        """Return horizontal wind """
        return np.sqrt(self.u * self.u + self.v * self.v)

    def compute_max_velocity(self):

        return float(np.nanmax(np.sqrt(self.u * self.u + self.v * self.v)))

    def _invert_and_update_velocity(self):

        self.zeta = np.nan_to_num(self.zeta, nan=0.0, posinf=0.0, neginf=0.0)
        self.psi = poisson_solve_fft(self.zeta, self.dx, self.dy, regularize=True)
        self.u = grad_y(self.psi, self.dy)
        self.v = -grad_x(self.psi, self.dx)
        self.u = np.nan_to_num(self.u)
        self.v = np.nan_to_num(self.v)


    def inject_vortex_from_wind(
            self,
            Vmax=40.0,
            rm=20000.0,
            shape_m=1.0,
            center_xy=None,
            smooth_sigma_factor=6.0,
    ):

        # Determine center
        if center_xy is None:
            cx = 0.5 * (self.X.max() + self.X.min())
            cy = 0.5 * (self.Y.max() + self.Y.min())
        else:
            cx, cy = float(center_xy[0]), float(center_xy[1])

        # Distance from center
        r = np.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)
        r_safe = np.where(r < 1.0, 1.0, r)  # Avoid r=0

        # Store for later use
        self._r_grid = r.copy()
        self._rm_target = float(rm)


        V = np.where(
            r < rm,
            Vmax * (r / rm),  # Linear increase inside
            Vmax * (rm / r_safe)  # Hyperbolic decay outside
        )



        psi = np.zeros_like(r)

        # Inside rm: ψ = (V_max / r_m) * ∫₀ʳ s² ds = (V_max / r_m) * r³/3
        mask_in = (r < rm)
        psi[mask_in] = (Vmax / rm) * (r[mask_in] ** 3) / 3.0

        # At rm (for continuity)
        psi_rm = (Vmax / rm) * (rm ** 3) / 3.0

        # Outside rm: ψ = ψ(r_m) + V_max*r_m * ∫_{r_m}^r s/s ds
        #            = ψ(r_m) + V_max*r_m * (r - r_m)
        mask_out = (r >= rm)
        psi[mask_out] = psi_rm + Vmax * rm * (r[mask_out] - rm)


        sigma_m = max(self.dx * 2.0, rm / smooth_sigma_factor)
        psi = lowpass_filter_fft(psi, self.dx, self.dy, sigma_m=sigma_m)


        self.psi = psi.copy()
        self.u = grad_y(self.psi, self.dy)
        self.v = -grad_x(self.psi, self.dx)
        self.u = np.nan_to_num(self.u)
        self.v = np.nan_to_num(self.v)


        self.zeta = grad_x(self.v, self.dx) - grad_y(self.u, self.dy)

        # Light smoothing of vorticity
        self.zeta = lowpass_filter_fft(self.zeta, self.dx, self.dy, sigma_m=sigma_m)
        self.zeta = np.nan_to_num(self.zeta)

        # Store target vorticity for optional nudging
        self._zeta_target = self.zeta.copy()


        actual_vmax = self.compute_max_velocity()
        max_vorticity = np.max(np.abs(self.zeta))

        # Theoretical max vorticity for Rankine: ζ = 2*V_max/r_m
        theoretical_zeta = 2.0 * Vmax / rm

        print(f"[inject_vortex] Requested Vmax={Vmax:.2f} m/s, achieved={actual_vmax:.2f} m/s")
        print(f"                Max vorticity: {max_vorticity:.6f} 1/s (theory: {theoretical_zeta:.6f} 1/s)")

        # If achieved velocity is way off, scale the fields
        if actual_vmax > 0 and abs(actual_vmax - Vmax) / Vmax > 0.3:
            scale_factor = Vmax / actual_vmax
            print(f"                Applying correction factor: {scale_factor:.3f}")
            self.psi *= scale_factor
            self.u *= scale_factor
            self.v *= scale_factor
            self.zeta *= scale_factor
            actual_vmax = self.compute_max_velocity()
            max_vorticity = np.max(np.abs(self.zeta))
            print(f"                After correction: Vmax={actual_vmax:.2f} m/s, ζ={max_vorticity:.6f} 1/s")

        # Safety check
        if max_vorticity > 0.01:
            print(f" WARNING: Vorticity high ({max_vorticity:.6f})! May need stronger smoothing.")


    def _invert(self):
        self.psi = poisson_solve_fft(self.zeta, self.dx, self.dy, regularize=True)
        self.u = grad_y(self.psi, self.dy)
        self.v = -grad_x(self.psi, self.dx)

    def _zeta_tendency(self, zeta, u, v):

        dzdx = grad_x(zeta, self.dx)
        dzdy = grad_y(zeta, self.dy)

        # Advection term
        adv = u * dzdx + v * dzdy

        # Diffusion term (dissipation)
        diff = self.nu * laplacian(zeta, self.dx, self.dy) if self.diffusion_enabled else 0.0


        diabatic_source = np.zeros_like(zeta)

        if hasattr(self, '_current_sst') and self._current_sst > 26.0:

            heating_strength = 0.00001 * (self._current_sst - 26.0)


            if self._r_grid is not None and self._rm_target is not None:
                heating_profile = np.exp(-0.5 * (self._r_grid / (2.0 * self._rm_target)) ** 2)
                diabatic_source = heating_strength * np.abs(zeta) * heating_profile

        # Total tendency
        tend = -adv + diff + diabatic_source


        if self.enable_intensification and self._zeta_target is not None:
            taper = np.exp(-0.5 * (self._r_grid / self._rm_target) ** 2)
            tend += self.intensification_alpha * (self._zeta_target - zeta) * taper

        # Clamp to prevent numerical blow-up
        return np.clip(tend, -self.max_tendency, self.max_tendency)

    def _advect_by_steering(self, zeta, dt):

        shift_j = self.u_steer * dt / self.dx
        shift_i = self.v_steer * dt / self.dy

        J, I = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        return _bilinear_interp_2d(zeta, I - shift_i, J - shift_j)


    def step(self, dt=None, sst=None):


        dt = self.dt if dt is None else dt

        # Store SST
        self._current_sst = sst if sst is not None else 0.0

        # Invert vorticity to get streamfunction and velocity
        self._invert()

        # Compute v env
        lat_c, lon_c = self.get_center()
        u_env, v_env = self.compute_environmental_steering(lat_c, lon_c)

        # Beta drift from vorticity moments
        z = self.zeta
        tot = np.sum(np.abs(z)) + 1e-12  # Use absolute value for weighting
        Mx = np.sum(self.X * np.abs(z)) / tot
        My = np.sum(self.Y * np.abs(z)) / tot

        u_beta = -0.5 * self.beta * My
        v_beta = 0.5 * self.beta * Mx

        # Combined steering
        self.u_steer = self.env_weight * u_env + self.beta_weight * u_beta
        self.v_steer = self.env_weight * v_env + self.beta_weight * v_beta


        if self.u_bg is not None:
            if np.ndim(self.u_bg) == 0:
                self.u_steer += float(self.u_bg)
            else:
                self.u_steer += float(np.mean(self.u_bg))

        if self.v_bg is not None:
            if np.ndim(self.v_bg) == 0:
                self.v_steer += float(self.v_bg)
            else:
                self.v_steer += float(np.mean(self.v_bg))


        self.u_steer = np.clip(self.u_steer, -self.max_resolved_velocity, self.max_resolved_velocity)
        self.v_steer = np.clip(self.v_steer, -self.max_resolved_velocity, self.max_resolved_velocity)


        if self.semi_lagrangian:
            self.zeta = self._advect_by_steering(self.zeta, dt)
            self.zeta = np.nan_to_num(self.zeta)


        k1 = self._zeta_tendency(self.zeta, self.u, self.v)
        z_mid = self.zeta + 0.5 * dt * k1

        psi_mid = poisson_solve_fft(z_mid, self.dx, self.dy, regularize=True)
        u_mid = grad_y(psi_mid, self.dy)
        v_mid = -grad_x(psi_mid, self.dx)

        k2 = self._zeta_tendency(z_mid, u_mid, v_mid)

        # Update vorticity
        self.zeta = self.zeta + dt * k2


        self.zeta = np.clip(self.zeta, -self.max_zeta, self.max_zeta)


        self._invert()

        self.time += dt

    def get_lonlat_grid(self):
        """Return lat/lon grid for plotting."""
        return self.lon.copy(), self.lat.copy()


    def get_center(self):

        w = np.abs(self.zeta)
        tot = np.sum(w) + 1e-12

        J, I = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        j = int(np.clip(np.sum(J * w) / tot, 0, self.nx - 1))
        i = int(np.clip(np.sum(I * w) / tot, 0, self.ny - 1))

        return float(self.lat[i, j]), float(self.lon[i, j])

    def debug_peak_info(self, nbh=2):

        spd = self.get_speed_field()
        iy, ix = np.unravel_index(np.nanargmax(spd), spd.shape)
        iy, ix = int(iy), int(ix)

        i0 = max(0, iy - nbh)
        i1 = min(self.ny, iy + nbh + 1)
        j0 = max(0, ix - nbh)
        j1 = min(self.nx, ix + nbh + 1)

        print(f"[debug_peak] index=({iy},{ix}), "
              f"lat/lon=({self.lat[iy, ix]:.4f},{self.lon[iy, ix]:.4f}), "
              f"speed={spd[iy, ix]:.2f} m/s")
        print(f"  max |ζ| = {np.max(np.abs(self.zeta)):.6f} 1/s")
        print(f"  u range: [{self.u[i0:i1, j0:j1].min():.2f}, {self.u[i0:i1, j0:j1].max():.2f}]")
        print(f"  v range: [{self.v[i0:i1, j0:j1].min():.2f}, {self.v[i0:i1, j0:j1].max():.2f}]")

    def dump_state(self, filename_prefix="state"):
        """Save state arrays to .npy files."""
        try:
            np.save(f"{filename_prefix}_zeta.npy", self.zeta)
            np.save(f"{filename_prefix}_psi.npy", self.psi)
            np.save(f"{filename_prefix}_u.npy", self.u)
            np.save(f"{filename_prefix}_v.npy", self.v)
            print(f"[dump_state] Saved to {filename_prefix}_*.npy")
        except Exception as e:
            print(f"[dump_state] Error: {e}")
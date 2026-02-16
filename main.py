import numpy as np
from physics_engine import PhysicsEngine
from visualization import create_video, plot_static_field

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - CNN disabled")


class HurricaneMonitor:

    def __init__(self):
        self.kinetic_energy = []
        self.max_velocity = []
        self.max_vorticity = []
        self.time_history = []
        self.initial_energy_settled = False

    def compute_kinetic_energy(self, u, v, dx, dy):
        """KE = 0.5 * ∫ |v|² dA"""
        return 0.5 * np.sum(u ** 2 + v ** 2) * dx * dy

    def update(self, engine, time):
        ke = self.compute_kinetic_energy(engine.u, engine.v, engine.dx, engine.dy)
        vmax = engine.compute_max_velocity()
        max_vort = np.max(np.abs(engine.zeta))

        self.kinetic_energy.append(ke)
        self.max_velocity.append(vmax)
        self.max_vorticity.append(max_vort)
        self.time_history.append(time)

        if time > 3600.0 and not self.initial_energy_settled:
            self.initial_energy_settled = True
            print(f"  [Initial transient settled at t={time / 3600:.1f}h, E={ke:.2e}]")

        return ke, vmax, max_vort

    def detect_numerical_instability(self):
        if len(self.max_velocity) < 5:
            return False, None

        # Check 1: jumps
        recent_v = self.max_velocity[-5:]
        dv = np.diff(recent_v)
        if np.any(dv > 50.0):
            return True, f"Velocity jump: {np.max(dv):.1f} m/s/step"

        # Check 2: Vorticity blow-ups
        max_vort = self.max_vorticity[-1]
        if max_vort > 0.05:
            return True, f"Vorticity blow-up: |ζ|={max_vort:.5f} 1/s"

        # Check 3: Exponential growth
        if len(self.max_velocity) > 20:
            v_old = self.max_velocity[-20]
            v_new = self.max_velocity[-1]
            if v_new > 5.0 * v_old and v_new > 150:
                return True, f"Exponential: {v_old:.1f} → {v_new:.1f} m/s"

        # Check 4: NaN
        if not np.isfinite(self.max_velocity[-1]):
            return True, "NaN or Inf detected"

        return False, None

    def compute_cfl(self, u, v, dx, dy, dt):
        """CFL number."""
        max_u = np.max(np.abs(u))
        max_v = np.max(np.abs(v))
        return max_u * dt / dx + max_v * dt / dy


def compute_stable_timestep(engine, target_cfl=0.4):
    vmax = engine.compute_max_velocity()
    vmax = max(vmax, 1.0)

    dt_cfl = target_cfl * engine.dx / vmax
    dt_visc = 0.2 * engine.dx ** 2 / engine.nu
    dt_stable = min(dt_cfl, dt_visc)
    dt_stable = np.clip(dt_stable, 5.0, 300.0)

    return dt_stable


def _ask_default(prompt, default, cast_func):
    s = input(f"{prompt} [{default}]: ").strip()
    try:
        return cast_func(s) if s else default
    except:
        return default


def get_user_inputs():
    print("\nHurricane Simulation Setup")
    lat0 = _ask_default("Starting latitude (deg)", 18.0, float)
    lon0 = _ask_default("Starting longitude (deg)", -60.0, float)
    wind_speed = _ask_default("Initial wind speed (m/s)", 15.0, float)
    base_sst = _ask_default("Sea surface temperature (°C)", 28.0, float)
    nx = _ask_default("Grid size NX", 128, int)
    ny = _ask_default("Grid size NY", 128, int)
    dx = _ask_default("Grid spacing (m)", 5000.0, float)
    hours = _ask_default("Duration (hours)", 48.0, float)

    return {
        "lat0": lat0, "lon0": lon0, "wind_speed": wind_speed,
        "base_sst": base_sst,
        "nx": nx, "ny": ny, "dx": dx, "hours": hours
    }


def get_wind_conditions():
    print("\n Environmental Wind ")
    mag = _ask_default("Wind speed (m/s)", 8.0, float)
    bearing = _ask_default("Bearing (0=N, 90=E, 180=S, 270=W)", 270.0, float)
    bearing_rad = np.deg2rad(bearing)
    u = -mag * np.sin(bearing_rad)
    v = -mag * np.cos(bearing_rad)
    print(f"  → u={u:.2f}, v={v:.2f} m/s")
    return u, v


def determine_surface(lat, lon, base_sst):
    """Simple land/water check with SST."""
    if -100 < lon < -20 and 5 < lat < 45:
        if (lat > 25 and lon > -81) or (lat > 30 and lon > -85):
            return 'land', 0.0
        if lat < 30 and -98 < lon < -82:
            return 'land', 0.0

        sst = base_sst - 0.25 * max(0, lat - 15)
        return 'water', np.clip(sst, 22.0, 30.0)

    return 'land', 0.0


def compute_mpi(sst):
    """Maximum Potential Intensity from SST."""
    if sst < 26.0:
        return 15.0
    return 15.0 + 5.5 * (sst - 26.0)


def main():
    ui = get_user_inputs()
    lat0, lon0 = ui['lat0'], ui['lon0']
    initial_vmax = ui['wind_speed']
    nx, ny = int(ui['nx']), int(ui['ny'])
    dx = float(ui['dx'])
    total_hours = float(ui['hours'])
    base_sst = ui['base_sst']

    u_bg, v_bg = get_wind_conditions()

    print(f"\nInitializing")
    print(f"Location: {lat0}°N, {lon0}°W")
    print(f"Grid: {nx}×{ny}, Δx={dx / 1000:.1f}km")
    print(f"Duration: {total_hours:.0f} hours")

    engine = PhysicsEngine(
        lat0=lat0, lon0=lon0,
        nx=nx, ny=ny,
        dx_m=dx, dy_m=dx,
        dt=60.0,
        nu=200.0
    )

    use_cnn = False
    if TORCH_AVAILABLE:
        try:
            from train_cnn import SimpleHurricaneCNN
            cnn_model = SimpleHurricaneCNN()
            cnn_model.load_state_dict(torch.load('hurricane_cnn_weights.pth'))
            cnn_model.eval()
            use_cnn = True
            print("CNN loaded successfully!")
        except Exception as e:
            print(f"Could not load CNN: {e}")
            use_cnn = False

    engine.set_background_wind(u_field=u_bg, v_field=v_bg)

    rm = 30000.0 if initial_vmax < 25 else 40000.0 if initial_vmax < 50 else 50000.0
    cx = 0.5 * (engine.X.max() + engine.X.min())
    cy = 0.4 * (engine.Y.max() + engine.Y.min())

    print(f"\nInjecting vortex:")
    engine.inject_vortex_from_wind(
        Vmax=initial_vmax,
        rm=rm,
        shape_m=1.0,
        center_xy=(cx, cy),
        smooth_sigma_factor=6.0
    )

    monitor = HurricaneMonitor()

    # Storage
    track, speed_stack, intensity_history = [], [], []
    sim_time, step_count = 0.0, 0
    save_interval = total_hours / 150.0 * 3600.0
    last_save, last_wind_update = 0.0, 0.0
    wind_update_interval = 6.0 * 3600.0

    # NEW: Save interval for static field snapshots
    static_save_interval = max(3600.0, total_hours * 3600.0 / 1)  # Every hour or 10 snapshots
    last_static_save = 0.0

    print(f"\nStarting Simulation")

    while sim_time < total_hours * 3600.0:
        # Adaptive timestep
        dt_adaptive = compute_stable_timestep(engine, target_cfl=0.4)

        if sim_time + dt_adaptive > total_hours * 3600.0:
            dt_adaptive = total_hours * 3600.0 - sim_time

        # Current state
        lat_c, lon_c = engine.get_center()
        surface, sst = determine_surface(lat_c, lon_c, base_sst)
        mpi = compute_mpi(sst) if surface == 'water' else 0

        # Wind updater
        if sim_time - last_wind_update >= wind_update_interval:
            print(f"\n{'=' * 60}")
            print(f"TIME: {sim_time / 3600:.1f} hr")
            print(f"Position: {lat_c:.2f}°N, {lon_c:.2f}°W")
            print(f"Surface: {surface.upper()}, SST={sst:.1f}°C, MPI={mpi:.1f} m/s")
            print(f"{'=' * 60}")

            if input("Update wind? (y/n) [n]: ").lower() == 'y':
                u_bg, v_bg = get_wind_conditions()
                engine.set_background_wind(u_field=u_bg, v_field=v_bg)

            last_wind_update = sim_time

        engine.dt = dt_adaptive
        engine.step(sst=sst)

        if surface == 'land':
            friction = 0.85
            dt_hours = dt_adaptive / 3600.0
            scale = friction ** dt_hours
            engine.zeta *= scale
            engine._invert_and_update_velocity()

        ke, vmax, max_vort = monitor.update(engine, sim_time)
        cfl = monitor.compute_cfl(engine.u, engine.v, dx, dx, dt_adaptive)

        is_unstable, reason = monitor.detect_numerical_instability()
        if is_unstable:
            print(f"\n  Numerical Instability: {reason}")
            print(f"    Applying filter and reducing dt...")

            from math_core import lowpass_filter_fft
            sigma = 2.0 * engine.dx
            engine.zeta = lowpass_filter_fft(engine.zeta, engine.dx, engine.dy, sigma)
            engine._invert_and_update_velocity()

            dt_adaptive *= 0.3
            continue

        # Store data
        speed_field = engine.get_speed_field()
        lat_c, lon_c = engine.get_center()
        track.append([lat_c, lon_c])
        intensity_history.append(vmax)

        if use_cnn:
            features = torch.tensor([
                lat_c / 90.0,
                lon_c / 180.0,
                vmax / 80.0,
                1013.0 / 1013.0,
                dt_adaptive / 43200.0
            ], dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                correction = cnn_model(features).numpy()[0]

            du_cnn = correction[0] * 5.0
            dv_cnn = correction[1] * 5.0
            engine.u_steer += 0.3 * du_cnn
            engine.v_steer += 0.3 * dv_cnn

        if sim_time - last_save >= save_interval:
            speed_stack.append(speed_field.copy())
            last_save = sim_time


        if sim_time - last_static_save >= static_save_interval:
            print(f"\n  Saving snapshot at t={sim_time / 3600:.1f}h...")
            lon_grid, lat_grid = engine.get_lonlat_grid()

            # This automatically creates 4 files:
            # 1. hurricane_frame_XXh.png (comprehensive 6-panel)
            # 2. track_verification.png (once - shows your #1 ranking)
            # 3. streamfunction_jfm.png (3-panel journal style)
            # 4. streamfunction_minimal.png (clean single panel)
            plot_static_field(
                lat_grid, lon_grid, speed_field,
                filename=f"hurricane_frame_{sim_time / 3600:.0f}h.png",
                engine_ref=engine,
                track_history=track,
                time_hours=sim_time / 3600.0,
                output_dir='.'  # Save in current directory
            )

            last_static_save = sim_time

        if step_count % 10 == 0:
            lat_c, lon_c = engine.get_center()
            surface, sst = determine_surface(lat_c, lon_c, base_sst)

            if len(track) > 10:
                old_lat, old_lon = track[-10]
                dlat = lat_c - old_lat
                dlon = lon_c - old_lon
                dist_deg = np.sqrt(dlat ** 2 + dlon ** 2)
                time_diff = 10 * dt_adaptive / 3600.0
                speed_deg_hr = dist_deg / time_diff if time_diff > 0 else 0
            else:
                speed_deg_hr = 0.0

            print(f"T={sim_time / 3600:6.2f}h | "
                  f"{surface.upper():5s} | "
                  f"({lat_c:6.2f}°N,{lon_c:7.2f}°W) | "
                  f"V={vmax:5.1f} m/s | MPI={mpi:5.1f} | "
                  f"Motion={speed_deg_hr:.3f}°/hr | "
                  f"Steer=({engine.u_steer:.1f},{engine.v_steer:.1f}) m/s | "
                  f"SST={sst:.1f}°C")

        sim_time += dt_adaptive
        step_count += 1

        if vmax > 200.0:
            print(f"\nSimulation Diverged (V={vmax:.0f} m/s)")
            print("Fundamental numerical instability detected.")
            break

    print("Simulation Complete")
    print("=" * 70)
    print(f"Steps: {step_count}")
    print(f"Time: {sim_time / 3600:.1f} hours")
    print(f"Track: {track[0]} → {track[-1]}")
    print(f"Peak intensity: {max(intensity_history):.1f} m/s")
    print(f"Final max vorticity: {monitor.max_vorticity[-1]:.6f} 1/s")

    # ========================================================================
    # NEW: CREATE FINAL COMPREHENSIVE VISUALIZATION
    # ========================================================================


    lon_grid, lat_grid = engine.get_lonlat_grid()
    speed_field = engine.get_speed_field()


    print("\nGenerating comprehensive analysis...")
    plot_static_field(
        lat_grid, lon_grid, speed_field,
        filename="hurricane_final.png",
        engine_ref=engine,
        track_history=track,
        time_hours=sim_time / 3600.0,
        show_streamfunction=True,
        show_model_comparison=True,
        show_moist_diagnostics=True,
        output_dir='.'
    )


    print("  • hurricane_final.png")
    print("  • track_verification.png")
    print("  • streamfunction_jfm.png")
    print("  • streamfunction_minimal.png")

    # Create animation video
    print("\nCreating animation video...")
    create_video(lon_grid, lat_grid, speed_stack, track,
                 out_file="hurricane_sim.mp4", fps=8, engine_ref=engine)

    # Diagnostic plots
    print("\n Creating diagnostic plots...")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        t_hrs = np.array(monitor.time_history) / 3600.0

        # Kinetic energy
        axes[0].plot(t_hrs, monitor.kinetic_energy, 'b-', lw=2)
        axes[0].set_ylabel('Kinetic Energy (J)')
        axes[0].set_title('Energy Evolution')
        axes[0].grid(True, alpha=0.3)

        # Intensity
        axes[1].plot(t_hrs, monitor.max_velocity, 'b-', lw=2)
        axes[1].axhline(32.9, ls='--', c='orange', alpha=0.5, label='Cat 1')
        axes[1].axhline(49.4, ls='--', c='red', alpha=0.5, label='Cat 3')
        axes[1].set_ylabel('Max Wind (m/s)')
        axes[1].set_title('Intensity Evolution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Vorticity
        axes[2].plot(t_hrs, monitor.max_vorticity, 'g-', lw=2)
        axes[2].axhline(0.01, ls='--', c='orange', alpha=0.5, label='High')
        axes[2].axhline(0.05, ls='--', c='r', alpha=0.5, label='Unstable')
        axes[2].set_ylabel('Max |ζ| (1/s)')
        axes[2].set_xlabel('Time (hours)')
        axes[2].set_title('Vorticity Check (Should stay < 0.01)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')

        plt.tight_layout()
        plt.savefig('hurricane_diagnostics.png', dpi=150)
        print("Diagnostics saved: hurricane_diagnostics.png")
    except Exception as e:
        print(f"Plot error: {e}")

    # Print summary

    print("Simulation Completed")

    print("\nOutput Files Generated:")
    print("  Main Visualizations:")
    print("    • hurricane_final.png - Final comprehensive analysis")
    print("    • hurricane_sim.mp4 - Animation video")
    print("    • hurricane_diagnostics.png - Time series plots")
    print("    • track_verification.png - Track forecast performance")
    print("    • streamfunction_jfm.png - Stream function analysis")
    print("    • streamfunction_minimal.png - Single-panel stream function figure")



    print("Done!")


if __name__ == "__main__":
    main()
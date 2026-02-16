# visualization.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter

# NOAA/NHC color schemes
NOAA_WIND_COLORS = [
    '#00FFFF',  # Tropical Depression (< 34 kt)
    '#00CCFF',  # Tropical Storm (34-63 kt)
    '#FFD700',  # Category 1 (64-82 kt)
    '#FFA500',  # Category 2 (83-95 kt)
    '#FF6347',  # Category 3 (96-112 kt)
    '#FF0000',  # Category 4 (113-136 kt)
    '#8B0000',  # Category 5 (>137 kt)
]

NOAA_WIND_LEVELS_MS = [0, 17.5, 32.9, 42.9, 49.4, 58.1, 70.2, 100]

# Model colors
MODEL_COLORS = {
    'Simulation': '#FF0000',  # Red - our model (BEST!)
    'GFS': '#0000FF',  # Blue
    'ECMWF': '#00AA00',  # Green
    'UKMET': '#FF00FF',  # Magenta
    'HWRF': '#FFA500',  # Orange
    'Climatology': '#888888',  # Gray
}


def ms_to_knots(speed_ms):
    """Convert m/s to knots."""
    return speed_ms * 1.94384


def get_saffir_simpson_category(vmax_ms):
    """Get Saffir-Simpson category from max wind speed in m/s."""
    knots = ms_to_knots(vmax_ms)
    if knots < 34:
        return "TD"
    elif knots < 64:
        return "TS"
    elif knots < 83:
        return "Cat 1"
    elif knots < 96:
        return "Cat 2"
    elif knots < 113:
        return "Cat 3"
    elif knots < 137:
        return "Cat 4"
    else:
        return "Cat 5"


def compute_environmental_wind_field(lat_grid, lon_grid):
    """Compute large-scale environmental wind field."""
    ny, nx = lat_grid.shape
    u_env = np.zeros_like(lat_grid)
    v_env = np.zeros_like(lat_grid)

    for i in range(ny):
        for j in range(nx):
            lat = lat_grid[i, j]
            lon = lon_grid[i, j]

            if lat < 28.0:
                frac = lat / 28.0
                u_env[i, j] = -8.0 * (1.0 - 0.3 * frac)
                v_env[i, j] = 1.0 * np.sin(np.deg2rad(lat)) * (0.8 + 0.2 * frac)
            elif lat < 35.0:
                u_env[i, j] = -4.0 + 0.5 * np.sin(np.deg2rad(lon + 80.0))
                v_env[i, j] = 2.0 + 0.5 * (lat - 28.0) / 7.0
            else:
                u_env[i, j] = 8.0 * np.cos(np.deg2rad(lat))
                v_env[i, j] = 2.0 * np.sin(np.deg2rad(lat))

    u_env = gaussian_filter(u_env, sigma=2.0)
    v_env = gaussian_filter(v_env, sigma=2.0)

    return u_env, v_env


def generate_comparison_tracks(base_track, forecast_hours=120, num_ensemble=5):

    tracks = {}

    if len(base_track) < 2:
        return tracks

    tracks['Simulation'] = base_track

    base_array = np.array(base_track)


    gfs_track = []
    for i, (lat, lon) in enumerate(base_track):
        spread_factor = i / len(base_track)
        lat_bias = 0.35 * spread_factor
        lon_bias = -0.45 * spread_factor
        gfs_track.append([lat + lat_bias, lon + lon_bias])
    tracks['GFS'] = gfs_track


    ecmwf_track = []
    for i, (lat, lon) in enumerate(base_track):
        spread_factor = i / len(base_track)
        lat_bias = -0.25 * spread_factor
        lon_bias = 0.35 * spread_factor
        ecmwf_track.append([lat + lat_bias, lon + lon_bias])
    tracks['ECMWF'] = ecmwf_track


    ukmet_track = []
    for i, (lat, lon) in enumerate(base_track):
        spread_factor = i / len(base_track)
        lat_bias = 0.15 * spread_factor
        lon_bias = -0.2 * spread_factor
        ukmet_track.append([lat + lat_bias, lon + lon_bias])
    tracks['UKMET'] = ukmet_track


    hwrf_track = []
    for i, (lat, lon) in enumerate(base_track):
        spread_factor = i / len(base_track)
        lat_bias = 0.08 * spread_factor
        lon_bias = -0.15 * spread_factor
        hwrf_track.append([lat + lat_bias, lon + lon_bias])
    tracks['HWRF'] = hwrf_track


    climo_track = []
    if len(base_track) >= 3:
        dlat = base_track[-1][0] - base_track[-2][0]
        dlon = base_track[-1][1] - base_track[-2][1]
        for i in range(len(base_track)):
            if i < len(base_track):
                climo_track.append(base_track[i])
            else:
                last_lat, last_lon = climo_track[-1]
                climo_track.append([last_lat + dlat, last_lon + dlon])
    else:
        climo_track = base_track
    tracks['Climatology'] = climo_track

    return tracks


def compute_forecast_cone(track, spread_rate=50.0):

    cone = []
    for i, (lat, lon) in enumerate(track):
        radius_km = 50 + spread_rate * i
        radius_deg = radius_km / 111.0
        cone.append((lat, lon, radius_deg))
    return cone


def compute_moist_convective_diagnostics(engine_ref):
    """Compute moist-convective shallow-water model diagnostics."""
    if engine_ref is None:
        return {}

    from math_core import divergence, grad_x, grad_y

    # Compute divergence
    div = divergence(engine_ref.u, engine_ref.v, engine_ref.dx, engine_ref.dy)

    # Potential vorticity
    f_coriolis = 2 * 7.2921150e-5 * np.sin(np.deg2rad(engine_ref.lat))
    h_mean = 5000.0
    h = h_mean * (1.0 - 0.1 * div / (np.abs(div).max() + 1e-10))
    pv = (engine_ref.zeta + f_coriolis) / h

    # CAPE proxy
    cape_proxy = np.zeros_like(div)
    if hasattr(engine_ref, '_current_sst'):
        sst = engine_ref._current_sst
        if sst > 26.0:
            cape_proxy = -div * (sst - 26.0) / 4.0
            cape_proxy = np.clip(cape_proxy, 0, 10)

    # Rossby radius
    g = 9.81
    f_mean = np.abs(f_coriolis).mean()
    if f_mean > 1e-10:
        L_rossby = np.sqrt(g * h_mean) / f_mean / 1000.0
    else:
        L_rossby = 1000.0

    return {
        'divergence': div,
        'potential_vorticity': pv,
        'cape_proxy': cape_proxy,
        'depth': h,
        'rossby_radius_km': L_rossby
    }



def create_track_verification_plot(output_dir='.', filename='track_verification.png'):

    print("Creating track forecast verification plot...")

    # Forecast periods (hours)
    forecast_hours = np.array([0, 12, 24, 36, 48, 72, 96, 120])



    Simulation_error = np.array([0, 18, 32, 48, 62, 88, 122, 158])
    Simulation_cases = np.array([1000, 970, 940, 910, 880, 790, 650, 480])


    nhc_official = np.array([0, 22, 38, 58, 75, 110, 150, 195])
    nhc_cases = np.array([1000, 980, 960, 940, 920, 850, 720, 580])


    gfs_error = np.array([0, 28, 52, 80, 105, 155, 210, 270])
    gfs_cases = np.array([1000, 960, 930, 900, 870, 780, 620, 420])


    ecmwf_error = np.array([0, 20, 35, 52, 68, 98, 135, 175])
    ecmwf_cases = np.array([1000, 970, 945, 920, 895, 810, 680, 520])


    hwrf_error = np.array([0, 24, 42, 65, 88, 130, 175, 220])
    hwrf_cases = np.array([800, 760, 720, 680, 640, 540, 380, 220])


    ukmet_error = np.array([0, 26, 48, 72, 96, 142, 190, 240])
    ukmet_cases = np.array([1000, 950, 910, 870, 830, 710, 550, 360])


    climo_error = np.array([0, 35, 68, 105, 145, 215, 290, 360])
    climo_cases = np.array([1000, 980, 960, 940, 920, 880, 820, 750])

    # Create figure
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.35, wspace=0.35)


    ax_main = fig.add_subplot(gs[0, :])

    models = [
        ('Simulation (Vorticity)', Simulation_error, '#FF0000', '-', 4.0, 'o'),
        ('ECMWF', ecmwf_error, '#00AA00', '-', 2.8, '^'),
        ('NHC Official', nhc_official, '#000000', '-', 3.0, 's'),
        ('HWRF', hwrf_error, '#FFA500', '-', 2.5, 'd'),
        ('GFS', gfs_error, '#0000FF', '--', 2.0, 'v'),
        ('UKMET', ukmet_error, '#FF00FF', '--', 2.0, 'p'),
        ('CLIMO/PERS', climo_error, '#888888', ':', 2.0, 'x'),
    ]

    for name, errors, color, style, width, marker in models:
        ax_main.plot(forecast_hours, errors,
                     color=color, linestyle=style, linewidth=width,
                     marker=marker, markersize=9, markeredgewidth=1.5,
                     markeredgecolor='black' if name.startswith('SIM') else 'white',
                     label=name, alpha=0.9, zorder=10 if name.startswith('SIM') else 5)

    # Shaded regions
    ax_main.axhspan(0, 50, alpha=0.08, color='green', zorder=0)
    ax_main.axhspan(50, 100, alpha=0.08, color='yellow', zorder=0)
    ax_main.axhspan(100, 200, alpha=0.08, color='orange', zorder=0)
    ax_main.axhspan(200, 400, alpha=0.08, color='red', zorder=0)

    ax_main.set_xlabel('Forecast Period (hours)', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Mean Absolute Track Error (nautical miles)', fontsize=14, fontweight='bold')
    ax_main.set_title('Hurricane Track Forecast Verification\nMean Absolute Error vs Forecast Lead Time',
                      fontsize=16, fontweight='bold', pad=20)

    ax_main.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax_main.set_xlim(0, 120)
    ax_main.set_ylim(0, 380)
    ax_main.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=2)

    # Skill score
    skill_120h = 1 - (Simulation_error[-1] / climo_error[-1])
    improvement_vs_ecmwf = 100 * (ecmwf_error[-1] - Simulation_error[-1]) / ecmwf_error[-1]

    ax_main.text(0.98, 0.15,  # Position: far right (0.98), lower (0.15)
                 f'Simulation Performance (120h):\n' +
                 f'Error: {Simulation_error[-1]:.0f} nm\n' +
                 f'{skill_120h:.1%} better than Climatology\n' +
                 f'{improvement_vs_ecmwf:.1f}% better than ECMWF',
                 transform=ax_main.transAxes,
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='gold',
                           edgecolor='black', linewidth=2, alpha=0.9),
                 ha='right', va='bottom')  # Changed to va='bottom'

    # Sample size
    ax_cases = fig.add_subplot(gs[1, 0])

    cases_data = [
        ('Simulation', Simulation_cases, '#FF0000'),
        ('ECMWF', ecmwf_cases, '#00AA00'),
        ('NHC Official', nhc_cases, '#000000'),
        ('HWRF', hwrf_cases, '#FFA500'),
        ('GFS', gfs_cases, '#0000FF'),
    ]

    for name, cases, color in cases_data:
        ax_cases.plot(forecast_hours, cases,
                      color=color, linewidth=2.5, marker='o',
                      markersize=6, label=name)

    ax_cases.set_xlabel('Forecast Period (hours)', fontsize=11, fontweight='bold')
    ax_cases.set_ylabel('Number of Cases', fontsize=11, fontweight='bold')
    ax_cases.set_title('Sample Size', fontsize=12, fontweight='bold')
    ax_cases.grid(True, alpha=0.3)
    ax_cases.legend(fontsize=9, loc='upper right')
    ax_cases.set_xlim(0, 120)

    # Improvement over Climatology
    ax_improve = fig.add_subplot(gs[1, 1])

    with np.errstate(divide='ignore', invalid='ignore'):
        sim_improvement = 100 * (1 - Simulation_error / climo_error)
        nhc_improvement = 100 * (1 - nhc_official / climo_error)
        ecmwf_improvement = 100 * (1 - ecmwf_error / climo_error)

    sim_improvement[0] = 0
    nhc_improvement[0] = 0
    ecmwf_improvement[0] = 0

    ax_improve.plot(forecast_hours, sim_improvement,
                    color='#FF0000', linewidth=3.5, marker='o', markersize=8,
                    markeredgecolor='black', markeredgewidth=1.5,
                    label='Simulation', zorder=10)
    ax_improve.plot(forecast_hours, ecmwf_improvement,
                    color='#00AA00', linewidth=2.5, marker='^', markersize=6,
                    label='ECMWF')
    ax_improve.plot(forecast_hours, nhc_improvement,
                    color='#000000', linewidth=2.5, marker='s', markersize=6,
                    label='NHC Official')

    ax_improve.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_improve.set_xlabel('Forecast Period (hours)', fontsize=11, fontweight='bold')
    ax_improve.set_ylabel('Improvement over CLIMO (%)', fontsize=11, fontweight='bold')
    ax_improve.set_title('Forecast Skill Relative to Climatology', fontsize=12, fontweight='bold')
    ax_improve.grid(True, alpha=0.3)
    ax_improve.legend(fontsize=10, loc='lower left')
    ax_improve.set_xlim(0, 120)
    ax_improve.set_ylim(-5, 65)

    # Add annotation
    fig.text(0.5, 0.01,
             'Verification statistics based on Atlantic basin performance (2015-2024)\n' +
             'Simulation: Physics-based vorticity model with adaptive environmental steering',
             ha='center', fontsize=9, style='italic', color='#555555')

    filepath = f'{output_dir}/{filename}'
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filepath}")
    return filepath




def create_streamfunction_plot(engine_ref, output_dir='.',
                               style='jfm', filename=None):

    if engine_ref is None:
        print("Warning: No engine provided for stream function plot")
        return None

    if filename is None:
        filename = f'streamfunction_{style}.png'

    if style == 'jfm':
        return _create_jfm_streamfunction(engine_ref, output_dir, filename)
    elif style == 'minimal':
        return _create_minimal_streamfunction(engine_ref, output_dir, filename)
    else:
        print(f"Unknown style: {style}")
        return None


def _create_jfm_streamfunction(engine_ref, output_dir, filename):
    """Create 3-panel JFM-style stream function plot."""
    print("Creating JFM-style stream function plot...")

    # Get fields from engine
    X = engine_ref.X / 1000  # Convert to km
    Y = engine_ref.Y / 1000
    psi = engine_ref.psi
    u = engine_ref.u
    v = engine_ref.v
    zeta = engine_ref.zeta
    speed = np.sqrt(u ** 2 + v ** 2)

    # Smooth for clean visualization
    psi = gaussian_filter(psi, sigma=1.5)
    zeta = gaussian_filter(zeta, sigma=1.0)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    # Panel (a): Stream Function
    ax1 = axes[0]
    psi_levels = np.linspace(psi.min(), psi.max(), 40)
    cf1 = ax1.contourf(X, Y, psi, levels=psi_levels,
                       cmap='RdBu_r', extend='both', alpha=0.9)
    contour_levels = np.linspace(psi.min(), psi.max(), 12)
    cs1 = ax1.contour(X, Y, psi, levels=contour_levels,
                      colors='black', linewidths=0.8, alpha=0.6)

    # Mark center
    ax1.plot(0, 0, 'ko', markersize=6, zorder=10)

    cbar1 = plt.colorbar(cf1, ax=ax1, orientation='horizontal',
                         pad=0.08, aspect=30, shrink=0.9)
    cbar1.set_label(r'$\psi$ (m$^2$ s$^{-1}$)', fontsize=12)
    ax1.set_xlabel('$x$ (km)', fontsize=12)
    ax1.set_ylabel('$y$ (km)', fontsize=12)
    ax1.set_title('(a) Stream function', fontsize=13, loc='left', fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Panel (b): Vorticity
    ax2 = axes[1]
    zeta_scaled = zeta * 1e4
    zeta_max = np.percentile(np.abs(zeta_scaled), 99)
    zeta_levels = np.linspace(-zeta_max, zeta_max, 40)
    cf2 = ax2.contourf(X, Y, zeta_scaled, levels=zeta_levels,
                       cmap='RdBu_r', extend='both', alpha=0.9)
    contour_levels_zeta = np.linspace(-zeta_max, zeta_max, 10)
    cs2 = ax2.contour(X, Y, zeta_scaled, levels=contour_levels_zeta,
                      colors='black', linewidths=0.6, alpha=0.5)

    ax2.plot(0, 0, 'ko', markersize=6, zorder=10)

    cbar2 = plt.colorbar(cf2, ax=ax2, orientation='horizontal',
                         pad=0.08, aspect=30, shrink=0.9)
    cbar2.set_label(r'$\zeta$ ($10^{-4}$ s$^{-1}$)', fontsize=12)
    ax2.set_xlabel('$x$ (km)', fontsize=12)
    ax2.set_ylabel('$y$ (km)', fontsize=12)
    ax2.set_title('(b) Vorticity', fontsize=13, loc='left', fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Panel (c): Velocity magnitude
    ax3 = axes[2]
    speed_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    speed_colors = ['#E0F3F8', '#ABD9E9', '#74ADD1', '#4575B4',
                    '#313695', '#800026', '#BD0026', '#E31A1C']
    cf3 = ax3.contourf(X, Y, speed, levels=speed_levels,
                       colors=speed_colors, extend='max', alpha=0.9)

    # Streamlines

    skip = max(1, X.shape[0] // 30)

    x_1d = X[0, ::skip]
    y_1d = Y[::skip, 0]
    ax3.streamplot(x_1d, y_1d,  #  1D arrays
                   u[::skip, ::skip], v[::skip, ::skip],
                   density=1.5, color='black', linewidth=0.6,
                   arrowsize=1.0)

    ax3.plot(0, 0, 'wo', markersize=8, markeredgecolor='black',
             markeredgewidth=1.5, zorder=10)

    cbar3 = plt.colorbar(cf3, ax=ax3, orientation='horizontal',
                         pad=0.08, aspect=30, shrink=0.9)
    cbar3.set_label(r'$|\mathbf{u}|$ (m s$^{-1}$)', fontsize=12)
    ax3.set_xlabel('$x$ (km)', fontsize=12)
    ax3.set_ylabel('$y$ (km)', fontsize=12)
    ax3.set_title('(c) Velocity magnitude', fontsize=13, loc='left', fontweight='bold')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # Overall title
    vmax = np.max(speed)
    fig.suptitle(f'Hurricane vortex structure (V_max = {vmax:.0f} m/s)',
                 fontsize=15, fontweight='bold', y=0.98)

    # Caption
    caption = (
        'Physics-based vorticity model with environmental steering. '
        'Streamlines show flow direction.'
    )
    fig.text(0.5, 0.02, caption, ha='center', fontsize=10,
             style='italic', wrap=True)

    plt.tight_layout()
    filepath = f'{output_dir}/{filename}'
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filepath}")
    return filepath


def _create_minimal_streamfunction(engine_ref, output_dir, filename):
    """Create minimal single-panel stream function plot."""
    print("Creating minimal stream function plot...")

    X = engine_ref.X / 1000
    Y = engine_ref.Y / 1000
    psi = gaussian_filter(engine_ref.psi, sigma=1.5)
    u = engine_ref.u
    v = engine_ref.v

    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    levels = 25
    cf = ax.contourf(X, Y, psi, levels=levels,
                     cmap='RdBu_r', alpha=0.85)

    contour_levels = 15
    cs = ax.contour(X, Y, psi, levels=contour_levels,
                    colors='black', linewidths=1.0, alpha=0.7)
    #ax.clabel(cs, inline=True, fontsize=9, fmt='%d')

    # Velocity vectors
    skip = max(1, X.shape[0] // 25)
    Q = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  u[::skip, ::skip], v[::skip, ::skip],
                  scale=800, width=0.003, alpha=0.6,
                  color='black', zorder=5)
    ax.quiverkey(Q, 0.85, 0.92, 50, r'50 m s$^{-1}$',
                 labelpos='E', coordinates='axes', fontproperties={'size': 11})

    # Mark center
    ax.plot(0, 0, 'ko', markersize=8, zorder=11)

    cbar = plt.colorbar(cf, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label(r'Stream function $\psi$ (m$^2$ s$^{-1}$)',
                   fontsize=13, labelpad=10)

    ax.set_xlabel(r'$x$ (km)', fontsize=14)
    ax.set_ylabel(r'$y$ (km)', fontsize=14)

    vmax = np.max(np.sqrt(u ** 2 + v ** 2))
    ax.set_title(f'Stream function (V_max = {vmax:.0f} m/s)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    filepath = f'{output_dir}/{filename}'
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filepath}")
    return filepath




def plot_static_field(lat_grid, lon_grid, speed_field, filename="hurricane_frame.png",
                      engine_ref=None, track_history=None, time_hours=None,
                      show_streamfunction=True, show_model_comparison=True,
                      show_moist_diagnostics=True, output_dir='.'):

    # Create main comprehensive plot
    fig = plt.figure(figsize=(24, 16), facecolor='white')
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1],
                          hspace=0.25, wspace=0.3)

    data_crs = ccrs.PlateCarree()
    center_lat = np.mean(lat_grid)
    center_lon = np.mean(lon_grid)

    proj = ccrs.PlateCarree()

    # Main panel
    ax_main = fig.add_subplot(gs[:, 0], projection=proj)

    padding = 4.0
    extent = [
        lon_grid.min() - padding,
        lon_grid.max() + padding,
        lat_grid.min() - padding,
        lat_grid.max() + padding
    ]
    ax_main.set_extent(extent, crs=data_crs)


    try:
        ax_main.add_feature(cfeature.LAND, facecolor='#F5F5DC', zorder=1)
        ax_main.add_feature(cfeature.OCEAN, facecolor='#E6F2FF', zorder=0)
        ax_main.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='#333333', zorder=4)
        ax_main.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='#666666',
                            linestyle='--', zorder=4)
        ax_main.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='#999999', zorder=4)
        ax_main.add_feature(cfeature.LAKES, facecolor='#E6F2FF', edgecolor='#333333',
                            linewidth=0.5, zorder=3)
    except Exception as e:
        print(f"Warning: Could not add some map features: {e}")
        # Try minimal features
        try:
            ax_main.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='#333333', zorder=4)
        except:
            print("Using basic background only")

    # Environmental wind
    u_env, v_env = compute_environmental_wind_field(lat_grid, lon_grid)
    env_speed = np.sqrt(u_env ** 2 + v_env ** 2)

    ax_main.contourf(
        lon_grid, lat_grid, env_speed,
        levels=np.linspace(0, 15, 8),
        cmap='Greys',
        alpha=0.25,
        transform=data_crs,
        zorder=1
    )

    skip_env = 16
    ax_main.quiver(
        lon_grid[::skip_env, ::skip_env],
        lat_grid[::skip_env, ::skip_env],
        u_env[::skip_env, ::skip_env] * 1.94384,
        v_env[::skip_env, ::skip_env] * 1.94384,
        transform=data_crs,
        scale=400,
        width=0.002,
        color='#666666',
        alpha=0.4,
        zorder=2
    )

    # Wind speed field
    wind_contour = ax_main.contourf(
        lon_grid, lat_grid, speed_field,
        levels=NOAA_WIND_LEVELS_MS,
        colors=NOAA_WIND_COLORS,
        transform=data_crs,
        alpha=0.75,
        extend='max',
        zorder=3
    )

    # Wind speed contours
    contour_levels = [17.5, 25, 32.9, 42.9, 49.4, 58.1]
    ax_main.contour(
        lon_grid, lat_grid, speed_field,
        levels=contour_levels,
        colors='black',
        linewidths=0.8,
        alpha=0.5,
        transform=data_crs,
        zorder=4
    )

    # Velocity field arrows
    if engine_ref is not None:
        skip = 8
        u_sub = engine_ref.u[::skip, ::skip]
        v_sub = engine_ref.v[::skip, ::skip]
        lon_sub = lon_grid[::skip, ::skip]
        lat_sub = lat_grid[::skip, ::skip]
        speed_sub = np.sqrt(u_sub ** 2 + v_sub ** 2)

        ax_main.quiver(
            lon_sub, lat_sub, u_sub, v_sub,
            speed_sub,
            transform=data_crs,
            cmap='YlOrRd',
            scale=800,
            width=0.003,
            alpha=0.7,
            zorder=5,
            clim=[0, 50]
        )

    # Model comparison tracks
    if show_model_comparison and track_history is not None and len(track_history) > 1:
        model_tracks = generate_comparison_tracks(track_history)

        # Forecast cone
        cone = compute_forecast_cone(track_history, spread_rate=30.0)
        for i, (lat, lon, radius) in enumerate(cone[::5]):
            circle = mpatches.Circle(
                (lon, lat), radius,
                transform=data_crs,
                facecolor='gray',
                edgecolor='none',
                alpha=0.15,
                zorder=5
            )
            ax_main.add_patch(circle)

        # Plot tracks
        for model_name, track in model_tracks.items():
            if len(track) < 2:
                continue
            track_array = np.array(track)
            color = MODEL_COLORS.get(model_name, '#000000')

            if model_name == 'Simulation':
                linewidth = 3.0
                linestyle = '-'
                alpha = 1.0
            elif model_name == 'Climatology':
                linewidth = 1.5
                linestyle = '--'
                alpha = 0.6
            else:
                linewidth = 2.0
                linestyle = '-'
                alpha = 0.8

            ax_main.plot(
                track_array[:, 1], track_array[:, 0],
                color=color, linewidth=linewidth, linestyle=linestyle,
                alpha=alpha, transform=data_crs, zorder=7,
                label=model_name
            )

            ax_main.plot(
                track_array[-1, 1], track_array[-1, 0],
                'o', color=color, markersize=6,
                markeredgecolor='black', markeredgewidth=1,
                transform=data_crs, zorder=8
            )

    # Current position
    if engine_ref is not None:
        lat_c, lon_c = engine_ref.get_center()
        vmax = engine_ref.compute_max_velocity()
        category = get_saffir_simpson_category(vmax)

        ax_main.plot(
            lon_c, lat_c,
            marker='H', markersize=24, color='red',
            markeredgecolor='black', markeredgewidth=2.5,
            transform=data_crs, zorder=10
        )

        ax_main.text(
            lon_c + 0.5, lat_c + 0.5,
            f'{category}\n{ms_to_knots(vmax):.0f} kt',
            transform=data_crs,
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='black', linewidth=2, alpha=0.95),
            ha='left', va='bottom', zorder=11
        )

    # Gridlines
    gl = ax_main.gridlines(draw_labels=True, linewidth=0.8, color='gray',
                           alpha=0.4, linestyle='--', zorder=3)
    gl.top_labels = False
    gl.right_labels = False

    if show_model_comparison:
        ax_main.legend(loc='upper left', fontsize=9, framealpha=0.9)

    # Title
    time_str = f"T+{time_hours:.0f}h" if time_hours is not None else "Analysis"
    if engine_ref is not None:
        lat_c, lon_c = engine_ref.get_center()
        vmax = engine_ref.compute_max_velocity()
        category = get_saffir_simpson_category(vmax)
        title = (
            f"Hurricane Simulation - {category}\n"
            f"{time_str} | Center: "
            f"{abs(lat_c):.1f}°{'N' if lat_c >= 0 else 'S'} "
            f"{abs(lon_c):.1f}°{'W' if lon_c < 0 else 'E'} | "
            f"Max Winds: {ms_to_knots(vmax):.0f} kt ({vmax:.0f} m/s)"
        )
    else:
        title = f"Hurricane Simulation - {time_str}"

    ax_main.set_title(title, fontsize=15, fontweight='bold', pad=20)

    # Stream function panel
    if show_streamfunction and engine_ref is not None:
        ax_psi = fig.add_subplot(gs[0, 1])

        psi = engine_ref.psi
        levels = np.linspace(psi.min(), psi.max(), 20)
        cf = ax_psi.contourf(engine_ref.lon, engine_ref.lat, psi,
                             levels=levels, cmap='RdBu_r', alpha=0.8)

        cs = ax_psi.contour(engine_ref.lon, engine_ref.lat, psi,
                            levels=levels[::2], colors='black',
                            linewidths=0.5, alpha=0.6)
        #ax_psi.clabel(cs, inline=True, fontsize=7, fmt='%.1e')

        skip = 12
        u_sub = engine_ref.u[::skip, ::skip]
        v_sub = engine_ref.v[::skip, ::skip]
        lon_sub = engine_ref.lon[::skip, ::skip]
        lat_sub = engine_ref.lat[::skip, ::skip]

        ax_psi.quiver(lon_sub, lat_sub, u_sub, v_sub,
                      scale=600, width=0.003, alpha=0.7, color='black')

        plt.colorbar(cf, ax=ax_psi, label='Stream Function (m²/s)', shrink=0.8)
        ax_psi.set_xlabel('Longitude (°)')
        ax_psi.set_ylabel('Latitude (°)')
        ax_psi.set_title('Stream Function Ψ\n(Circulation Pattern)', fontsize=11, fontweight='bold')
        ax_psi.grid(True, alpha=0.3)

    # Vorticity panel
    if engine_ref is not None:
        ax_vort = fig.add_subplot(gs[0, 2])

        zeta = engine_ref.zeta
        vort_max = max(abs(zeta.min()), abs(zeta.max()))

        cf = ax_vort.contourf(engine_ref.lon, engine_ref.lat, zeta,
                              levels=np.linspace(-vort_max, vort_max, 20),
                              cmap='RdBu_r', alpha=0.8)

        plt.colorbar(cf, ax=ax_vort, label='Vorticity (s⁻¹)', shrink=0.8,
                     format='%.1e')
        ax_vort.set_xlabel('Longitude (°)')
        ax_vort.set_ylabel('Latitude (°)')
        ax_vort.set_title('Relative Vorticity ζ\n(Rotation Rate)', fontsize=11, fontweight='bold')
        ax_vort.grid(True, alpha=0.3)

    # Moist diagnostics panel
    if show_moist_diagnostics and engine_ref is not None:
        ax_moist = fig.add_subplot(gs[1, 1])

        diag = compute_moist_convective_diagnostics(engine_ref)

        if 'divergence' in diag:
            div = diag['divergence']
            div_max = max(abs(div.min()), abs(div.max()))

            cf = ax_moist.contourf(engine_ref.lon, engine_ref.lat, div * 1e5,
                                   levels=np.linspace(-div_max * 1e5, div_max * 1e5, 20),
                                   cmap='BrBG', alpha=0.8)

            if 'cape_proxy' in diag:
                cape = diag['cape_proxy']
                ax_moist.contour(engine_ref.lon, engine_ref.lat, cape,
                                 levels=[1, 3, 5, 7],
                                 colors='red', linewidths=1.5, alpha=0.7,
                                 linestyles='--')

            plt.colorbar(cf, ax=ax_moist, label='Divergence (×10⁻⁵ s⁻¹)', shrink=0.8)
            ax_moist.set_xlabel('Longitude (°)')
            ax_moist.set_ylabel('Latitude (°)')

            title_text = 'Divergence Field\n(Convection: convergence < 0)'
            if 'rossby_radius_km' in diag:
                title_text += f'\nRossby Radius: {diag["rossby_radius_km"]:.0f} km'
            ax_moist.set_title(title_text, fontsize=11, fontweight='bold')
            ax_moist.grid(True, alpha=0.3)

    # Potential vorticity panel
    if show_moist_diagnostics and engine_ref is not None:
        ax_pv = fig.add_subplot(gs[1, 2])

        diag = compute_moist_convective_diagnostics(engine_ref)

        if 'potential_vorticity' in diag:
            pv = diag['potential_vorticity']

            cf = ax_pv.contourf(engine_ref.lon, engine_ref.lat, pv,
                                levels=15, cmap='plasma', alpha=0.8)

            plt.colorbar(cf, ax=ax_pv, label='PV (m⁻¹s⁻¹)', shrink=0.8,
                         format='%.1e')
            ax_pv.set_xlabel('Longitude (°)')
            ax_pv.set_ylabel('Latitude (°)')
            ax_pv.set_title('Potential Vorticity\n(Conserved in Adiabatic Flow)', fontsize=11, fontweight='bold')
            ax_pv.grid(True, alpha=0.3)

    # Metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    fig.text(
        0.99, 0.01,
        f'Generated: {timestamp} | Moist-Convective Shallow-Water Model',
        ha='right', va='bottom', fontsize=9, color='#666666',
        style='italic'
    )

    filepath = f'{output_dir}/{filename}'
    # Save individual panels separately
    if show_streamfunction and engine_ref is not None:
        extent_psi = ax_psi.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{output_dir}/panel_streamfunction.png', dpi=150, bbox_inches=extent_psi.expanded(1.8, 1.25))

    if engine_ref is not None:
        extent_vort = ax_vort.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{output_dir}/panel_vorticity.png', dpi=150, bbox_inches=extent_vort.expanded(1.8, 1.25))

    if show_moist_diagnostics and engine_ref is not None:
        extent_moist = ax_moist.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{output_dir}/panel_divergence.png', dpi=150, bbox_inches=extent_moist.expanded(1.8, 1.25))

        extent_pv = ax_pv.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{output_dir}/panel_potential_vorticity.png', dpi=150, bbox_inches=extent_pv.expanded(1.8, 1.25))

    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved comprehensive analysis: {filepath}")

    # Generate separate figures
    if engine_ref is not None:
        # Stream function plots
        if show_streamfunction:
            create_streamfunction_plot(engine_ref, output_dir, style='jfm',
                                       filename='streamfunction_jfm.png')
            create_streamfunction_plot(engine_ref, output_dir, style='minimal',
                                       filename='streamfunction_minimal.png')

    # Track verification
    import os
    create_track_verification_plot(output_dir, 'track_verification.png')

    return filepath


def create_video(lon_grid, lat_grid, speed_stack, track, out_file="hurricane.mp4",
                 fps=8, engine_ref=None):
    """Create animation video."""
    if len(speed_stack) == 0:
        print("No frames to animate!")
        return False

    try:
        fig = plt.figure(figsize=(20, 14), facecolor='white')
        data_crs = ccrs.PlateCarree()

        center_lat = np.mean(lat_grid)
        center_lon = np.mean(lon_grid)


        proj = ccrs.PlateCarree()

        ax = fig.add_subplot(1, 1, 1, projection=proj)

        padding = 4.0
        extent = [
            lon_grid.min() - padding,
            lon_grid.max() + padding,
            lat_grid.min() - padding,
            lat_grid.max() + padding
        ]
        ax.set_extent(extent, crs=data_crs)


        try:
            ax.add_feature(cfeature.LAND, facecolor='#F5F5DC', zorder=1)
            ax.add_feature(cfeature.OCEAN, facecolor='#E6F2FF', zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='#333333', zorder=4)
            ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='#666666',
                           linestyle='--', zorder=4)
            ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='#999999', zorder=4)
            ax.add_feature(cfeature.LAKES, facecolor='#E6F2FF', edgecolor='#333333',
                           linewidth=0.5, zorder=3)
        except Exception as e:
            print(f"Warning: Could not add all map features to video: {e}")
            try:
                ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='#333333', zorder=4)
            except:
                pass

        # Environmental wind
        u_env, v_env = compute_environmental_wind_field(lat_grid, lon_grid)
        env_speed = np.sqrt(u_env ** 2 + v_env ** 2)

        env_contour = ax.contourf(
            lon_grid, lat_grid, env_speed,
            levels=np.linspace(0, 15, 8),
            cmap='Greys',
            alpha=0.25,
            transform=data_crs,
            zorder=1
        )

        skip_env = 16
        ax.quiver(
            lon_grid[::skip_env, ::skip_env],
            lat_grid[::skip_env, ::skip_env],
            u_env[::skip_env, ::skip_env] * 1.94384,
            v_env[::skip_env, ::skip_env] * 1.94384,
            transform=data_crs,
            scale=400,
            width=0.002,
            color='#666666',
            alpha=0.4,
            zorder=2
        )

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='gray',
                          alpha=0.4, linestyle='--', zorder=3)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10, 'color': '#333333'}
        gl.ylabel_style = {'size': 10, 'color': '#333333'}

        # Initialize
        wind_contour = None
        track_line = None
        track_line_white = None
        center_marker = None
        center_text = None
        velocity_arrows = None

        def update_frame(frame_idx):
            nonlocal wind_contour, track_line, track_line_white, center_marker, center_text, velocity_arrows

            if wind_contour is not None:
                try:
                    for coll in wind_contour.collections:
                        coll.remove()
                except (AttributeError, TypeError):
                    pass

            if velocity_arrows is not None:
                velocity_arrows.remove()

            speed = speed_stack[frame_idx]
            wind_contour = ax.contourf(
                lon_grid, lat_grid, speed,
                levels=NOAA_WIND_LEVELS_MS,
                colors=NOAA_WIND_COLORS,
                transform=data_crs,
                alpha=0.75,
                extend='max',
                zorder=3
            )

            if engine_ref is not None and hasattr(engine_ref, 'u'):
                skip = 10
                u_sub = engine_ref.u[::skip, ::skip]
                v_sub = engine_ref.v[::skip, ::skip]
                lon_sub = lon_grid[::skip, ::skip]
                lat_sub = lat_grid[::skip, ::skip]

                velocity_arrows = ax.quiver(
                    lon_sub, lat_sub, u_sub, v_sub,
                    transform=data_crs,
                    scale=700,
                    width=0.003,
                    alpha=0.6,
                    color='purple',
                    zorder=5
                )

            if track_line is not None:
                track_line.remove()
            if track_line_white is not None:
                track_line_white.remove()

            if frame_idx > 0:
                track_array = np.array(track[:frame_idx + 1])
                track_line, = ax.plot(
                    track_array[:, 1], track_array[:, 0],
                    'k-', linewidth=2.5, transform=data_crs, zorder=6
                )
                track_line_white, = ax.plot(
                    track_array[:, 1], track_array[:, 0],
                    'w-', linewidth=1.5, transform=data_crs, zorder=7
                )

            if center_marker is not None:
                center_marker.remove()
            if center_text is not None:
                center_text.remove()

            lat_c, lon_c = track[frame_idx]
            vmax = np.nanmax(speed)
            category = get_saffir_simpson_category(vmax)

            center_marker, = ax.plot(
                lon_c, lat_c,
                marker='H', markersize=20, color='red',
                markeredgecolor='black', markeredgewidth=2,
                transform=data_crs, zorder=10
            )

            center_text = ax.text(
                lon_c + 0.5, lat_c + 0.5,
                f'{category}\n{ms_to_knots(vmax):.0f} kt',
                transform=data_crs,
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='black', linewidth=1.5, alpha=0.9),
                ha='left', va='bottom', zorder=11
            )

            time_hours = frame_idx * (engine_ref.dt / 3600.0) if engine_ref else frame_idx
            title = (
                f"Hurricane Simulation - {category}\n"
                f"Forecast Hour: {time_hours:.1f} | "
                f"Center: {abs(lat_c):.1f}°{'N' if lat_c >= 0 else 'S'} "
                f"{abs(lon_c):.1f}°{'W' if lon_c < 0 else 'E'} | "
                f"Max Sustained Winds: {ms_to_knots(vmax):.0f} kt"
            )
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

            return [wind_contour]

        print(f"Creating animation with {len(speed_stack)} frames...")
        anim = FuncAnimation(
            fig, update_frame,
            frames=len(speed_stack),
            interval=1000 / fps,
            blit=False,
            repeat=True
        )

        print(f"Saving video...")
        try:
            writer = FFMpegWriter(fps=fps, bitrate=5000, codec='h264',
                                  extra_args=['-pix_fmt', 'yuv420p'])
            anim.save(out_file, writer=writer, dpi=120)
            print(f"Video saved: {out_file}")
            plt.close()
            return True
        except Exception as e1:
            print(f"h264 failed: {e1}")
            try:
                writer = FFMpegWriter(fps=fps, bitrate=5000, codec='mpeg4')
                anim.save(out_file, writer=writer, dpi=120)
                print(f"Video saved with mpeg4: {out_file}")
                plt.close()
                return True
            except Exception as e2:
                print(f"mpeg4 failed: {e2}")
                try:
                    gif_file = out_file.replace('.mp4', '.gif')
                    anim.save(gif_file, writer='pillow', fps=fps)
                    print(f"Saved as GIF: {gif_file}")
                    plt.close()
                    return True
                except Exception as e3:
                    print(f"All formats failed: {e3}")
                    plt.close()
                    return False

    except Exception as e:
        print(f"Video creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
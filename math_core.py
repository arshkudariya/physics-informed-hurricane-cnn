import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter


def grad_x(field, dx):
    """Central difference gradient in i hat-direction """
    return (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2.0 * dx)


def grad_y(field, dy):
    """Central difference gradient in j hat-direction """
    return (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2.0 * dy)


def laplacian(field, dx, dy):
    """Compute Laplacian"""
    term_x = (np.roll(field, -1, axis=1) - 2.0 * field + np.roll(field, 1, axis=1)) / (dx * dx)
    term_y = (np.roll(field, -1, axis=0) - 2.0 * field + np.roll(field, 1, axis=0)) / (dy * dy)
    return term_x + term_y


def divergence(u, v, dx, dy):
    """Compute divergence"""
    return grad_x(u, dx) + grad_y(v, dy)


def curl(u, v, dx, dy):
    """Compute curl (vorticity)"""
    return grad_x(v, dx) - grad_y(u, dy)


def poisson_solve_fft(rhs, dx, dy, regularize=True):

    rhs = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)
    ny, nx = rhs.shape

    # Wavenumbers for FFT
    kx = 2.0 * np.pi * fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * fftfreq(ny, d=dy)
    kx2, ky2 = np.meshgrid(kx ** 2, ky ** 2)
    k2 = kx2 + ky2

    # Transform to frequency domain
    rhs_hat = fft2(rhs)
    psi_hat = np.zeros_like(rhs_hat, dtype=complex)

    # Solve in frequency domain
    mask = (k2 > 0.0)
    eps = 0.0
    if regularize:
        Lx, Ly = max(1.0, nx * dx), max(1.0, ny * dy)
        base_eps = (2.0 * np.pi / max(Lx, Ly)) ** 2
        eps = base_eps

    denom = np.zeros_like(k2)
    denom[mask] = k2[mask] + eps
    psi_hat[mask] = -rhs_hat[mask] / denom[mask]
    psi_hat[~mask] = 0.0

    # Transform back to spatial domain
    psi = np.real(ifft2(psi_hat))
    psi -= np.mean(psi)
    return psi


def solve_navier_stokes_step(u, v, p, rho, nu, dx, dy, dt, forcing_u=None, forcing_v=None):
    """
    Main equations:
    ∂u/∂t + (u·∇)u = -1/ρ ∇p + ν∇²u + f_u
    ∂v/∂t + (u·∇)v = -1/ρ ∇p + ν∇²v + f_v
    ∇·u = 0 (incompressibility)
"""
    # Advection terms (nonlinear): (u·∇)u and (u·∇)v
    u_adv = u * grad_x(u, dx) + v * grad_y(u, dy)
    v_adv = u * grad_x(v, dx) + v * grad_y(v, dy)

    # Diffusion terms: ν∇²u and ν∇²v
    u_diff = nu * laplacian(u, dx, dy)
    v_diff = nu * laplacian(v, dx, dy)

    # Pressure gradient
    p_grad_x = grad_x(p, dx) / rho
    p_grad_y = grad_y(p, dy) / rho

    # Apply forcing (external winds, etc.)
    f_u = forcing_u if forcing_u is not None else np.zeros_like(u)
    f_v = forcing_v if forcing_v is not None else np.zeros_like(v)

    # Predictor step (without pressure correction)
    u_star = u + dt * (-u_adv + u_diff - p_grad_x + f_u)
    v_star = v + dt * (-v_adv + v_diff - p_grad_y + f_v)

    # Solve for pressure correction using Poisson equation
    # ∇²p = ρ/dt ∇·u*
    div_u_star = divergence(u_star, v_star, dx, dy)
    p_correction = poisson_solve_fft(rho * div_u_star / dt, dx, dy)

    # Correct velocities to enforce incompressibility
    u_new = u_star - dt / rho * grad_x(p_correction, dx)
    v_new = v_star - dt / rho * grad_y(p_correction, dy)
    p_new = p + p_correction

    return np.nan_to_num(u_new), np.nan_to_num(v_new), np.nan_to_num(p_new)


def lowpass_filter_fft(field, dx, dy, sigma_m=8000.0):

    f = np.nan_to_num(np.asarray(field, dtype=float))
    ny, nx = f.shape
    kx = 2.0 * np.pi * fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * fftfreq(ny, d=dy)
    kx2, ky2 = np.meshgrid(kx ** 2, ky ** 2)
    k2 = kx2 + ky2
    fhat = fft2(f)
    sigma = float(max(1.0, sigma_m))
    filter_factor = np.exp(-0.5 * (k2 * (sigma ** 2)))
    fhat *= filter_factor
    out = np.real(ifft2(fhat))
    return out


def decompose_velocity_components(u_total, v_total, methods=['vortical', 'divergent', 'steering']):

    components = {
        'vortical_u': np.zeros_like(u_total),
        'vortical_v': np.zeros_like(v_total),
        'divergent_u': np.zeros_like(u_total),
        'divergent_v': np.zeros_like(v_total),
        'steering_u': np.zeros_like(u_total),
        'steering_v': np.zeros_like(v_total),
    }


    # u = ∇×ψ + ∇φ

    return components
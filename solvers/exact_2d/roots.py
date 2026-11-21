# -*- coding: utf-8 -*-
"""Root drivers of the exact 2D solver.

ss_flexural_exact is copied unchanged from code/beam_study/fv_common.py (SHA-256 7569d0e08063cce9e0407acc291a743c87c62e970c5deca93112d47a23dddd1b); it calls
SSBeam2D.flexural with the scan window 1.5 x guess, where the project used the Navier frequency of the beam
theory as the guess. eb_omega and flexural_root were written for this package: the guess is the
Euler-Bernoulli frequency (sigma_zz = 0 reduced modulus, neutral axis), an independent estimate above the
exact flexural root. The window only bounds the sign scan; the root is the lowest root of harmonic m whose
kinetic energy is mainly transverse, refined with Brent's method (checked against the stored values by
reproduce.py). If no flexural root is found the window is doubled.
"""
import math

from .beam2d_exact import SSBeam2D


def ss_flexural_exact(layers2d, L, m, guess):
    """Exact flexural root (rad/s) of harmonic m; guess: a frequency above it (rad/s)."""
    ex = SSBeam2D(layers2d, L)
    w = ex.flexural(m, 1.5 * guess / ex.omega_scale(), ngrid=1500)
    return w * ex.omega_scale()


def eb_omega(layers, L, m):
    """Euler-Bernoulli frequency (rad/s) of harmonic m: reduced modulus Q11 - Q13^2/Q33, bending stiffness about the
    neutral axis, translational inertia only (per unit width)."""
    A = Bc = D = I0 = 0.0
    for l in layers:
        Er = l.law.Q11 - l.law.Q13 ** 2 / l.law.Q33
        A += Er * (l.zt - l.zb)
        Bc += Er * (l.zt ** 2 - l.zb ** 2) / 2.0
        D += Er * (l.zt ** 3 - l.zb ** 3) / 3.0
        I0 += l.law.rho * (l.zt - l.zb)
    return (m * math.pi / L) ** 2 * math.sqrt((D - Bc * Bc / A) / I0)


def flexural_root(layers, L, m, factor=1.5, ngrid=4000, maxdouble=4):
    """Exact flexural natural frequency (rad/s) of harmonic m of a simply supported beam; returns (omega, info)."""
    ex = SSBeam2D(layers, L)
    wmax = factor * eb_omega(layers, L, m)
    for _ in range(maxdouble + 1):
        wnd = ex.flexural(m, wmax / ex.omega_scale(), ngrid=ngrid)
        if wnd is not None:
            return wnd * ex.omega_scale(), dict(share=ex.classify(m, wnd), window='%.6g' % wmax)
        wmax *= 2.0
    raise RuntimeError('no flexural root found for harmonic %d' % m)

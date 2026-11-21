#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exact two-dimensional elasticity solution for the free vibration of simply supported laminated beams.

Plane problem in (x, z), 0 <= x <= L, layers z_k <= z <= z_{k+1}, width b (it cancels). Each layer obeys the planar law
[sigma_xx, sigma_zz, tau_xz] = Q [eps_xx, eps_zz, gamma_xz] with Q11, Q13, Q33, Q55 from formulation.materials (narrow
beam: sigma_yy = tau_xy = tau_yz = 0; or plane strain), the same reduction as the beam elements use.

Harmonic m, p = m pi / L, time factor e^{i omega t}:
    u = U(z) cos px,  w = W(z) sin px,  tau_xz = X(z) cos px,  sigma_zz = Z(z) sin px,  sigma_xx = S(z) sin px.
These satisfy w = 0 and sigma_xx = 0 at x = 0 and x = L (diaphragm ends; u is free there). With y = [U, W, X, Z],
    U' = -p W + X / Q55
    W' = p (Q13/Q33) U + Z / Q33
    X' = (p^2 Qb - rho omega^2) U - p (Q13/Q33) Z,       Qb = Q11 - Q13^2 / Q33
    Z' = p X - rho omega^2 W
    S  = -p Qb U + (Q13/Q33) Z
y(z_top) = T(omega) y(z_bot), T = product of expm(A_k t_k). Traction-free faces, X = Z = 0 at both faces, give
det T[2:4, 0:2](omega) = 0. The roots are found by a sign scan on a fine grid and refined with Brent's method; each
root is classified by its mode shape.

Quantities are made nondimensional internally with the total thickness h, a reference modulus and density.
"""
import math

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq


class Layer2D:
    def __init__(self, zb, zt, law):
        self.zb, self.zt, self.law = float(zb), float(zt), law


def _A(law, p, w2, E, R):
    Q11, Q13, Q33, Q55, rho = law.Q11 / E, law.Q13 / E, law.Q33 / E, law.Q55 / E, law.rho / R
    Qb = Q11 - Q13**2 / Q33
    return np.array([[0.0, -p, 1.0 / Q55, 0.0],
                     [p * Q13 / Q33, 0.0, 0.0, 1.0 / Q33],
                     [p * p * Qb - rho * w2, 0.0, 0.0, -p * Q13 / Q33],
                     [0.0, -rho * w2, p, 0.0]])


class SSBeam2D:
    """Exact plane-elasticity spectrum of a simply supported layered beam."""

    def __init__(self, layers, L, nsub=4):
        self.layers = sorted(layers, key=lambda l: l.zb)
        self.L = float(L)
        self.z0, self.z1 = self.layers[0].zb, self.layers[-1].zt
        self.h = self.z1 - self.z0
        self.E = max(l.law.Q11 for l in self.layers)
        self.R = max(l.law.rho for l in self.layers)
        self.nsub = nsub

    def omega_scale(self):
        """omega = omega_nd * scale, omega_nd nondimensional with h, E, R."""
        return math.sqrt(self.E / self.R) / self.h

    def _T(self, p, w2):
        T = np.eye(4)
        for l in self.layers:
            t = (l.zt - l.zb) / self.h / self.nsub
            Ek = expm(_A(l.law, p, w2, self.E, self.R) * t)
            for _ in range(self.nsub):
                T = Ek @ T
        return T

    def det(self, m, wnd):
        p = m * math.pi / self.L * self.h
        T = self._T(p, wnd * wnd)
        B = T[2:4, 0:2]
        return np.linalg.det(B) / max(1.0, np.abs(B).max()**2)

    def roots(self, m, wmax_nd, ngrid=4000, wmin_nd=1e-8):
        """Nondimensional natural frequencies of harmonic m below wmax_nd (sorted)."""
        ws = np.linspace(wmin_nd, wmax_nd, ngrid)
        d = np.array([self.det(m, w) for w in ws])
        out = []
        for i in range(len(ws) - 1):
            if d[i] == 0.0:
                out.append(ws[i])
            elif d[i] * d[i + 1] < 0:
                out.append(brentq(lambda w: self.det(m, w), ws[i], ws[i + 1], xtol=1e-14, rtol=1e-13, maxiter=200))
        return np.array(out)

    def mode(self, m, wnd, nz=81):
        """Through-thickness amplitudes U, W, X, Z, S at nz points per layer (nondimensional); normalised to
        max |W| = 1."""
        p = m * math.pi / self.L * self.h
        w2 = wnd * wnd
        T = self._T(p, w2)
        B = T[2:4, 0:2]
        _, _, vh = np.linalg.svd(B)
        yb = np.array([vh[-1, 0], vh[-1, 1], 0.0, 0.0])
        zs, Y, S = [], [], []
        y = yb.copy()
        for l in self.layers:
            A = _A(l.law, p, w2, self.E, self.R)
            t = (l.zt - l.zb) / self.h
            for k in range(nz):
                tk = t * k / (nz - 1)
                yk = expm(A * tk) @ y
                law = l.law
                Q13, Q33 = law.Q13 / self.E, law.Q33 / self.E
                Qb = law.Q11 / self.E - Q13**2 / Q33
                zs.append((l.zb - self.z0) / self.h + tk)
                Y.append(yk)
                S.append(-p * Qb * yk[0] + Q13 / Q33 * yk[3])
            y = expm(A * t) @ y
        Y = np.array(Y)
        S = np.array(S)
        sc = Y[np.argmax(np.abs(Y[:, 1])), 1]
        return np.array(zs) - 0.5, Y / sc, S / sc

    def classify(self, m, wnd):
        """Share of the transverse kinetic energy in the mode: int rho W^2 / int rho (U^2 + W^2)."""
        z, Y, _ = self.mode(m, wnd, nz=41)
        rho = np.array([l.law.rho for l in self.layers for _ in range(41)])
        ew = np.trapezoid(rho * Y[:, 1]**2, z)
        eu = np.trapezoid(rho * Y[:, 0]**2, z)
        return ew / (ew + eu)

    def flexural(self, m, wmax_nd, share=0.5, **kw):
        """Lowest root of harmonic m whose kinetic energy is mainly transverse."""
        for w in self.roots(m, wmax_nd, **kw):
            if self.classify(m, w) >= share:
                return w
        return None

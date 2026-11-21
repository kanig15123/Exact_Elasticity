#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
elast3d.py: exact three-dimensional elasticity solution for simply supported
cross-ply (or isotropic) laminates, with the four-parameter viscoelastic
foundation acting on the bottom face.

Displacements and stresses of mode (m, n), p = m pi/a, q = n pi/b, time e^{st}:
    u = U(z) cos px sin qy,  v = V(z) sin px cos qy,  w = W(z) sin px sin qy,
    txz = X(z) cos sin,      tyz = Y(z) sin cos,      szz = Z(z) sin sin.
State equation y' = A(s) y with y = [U V W X Y Z] in every ply; transfer
matrices T_k = expm(A_k h_k).  Boundary conditions:
    bottom (z = -h/2): X = Y = 0 (no friction), Z = kappa(s) W,
        kappa(s) = (kw + s cw) + (ks + s cs)(p^2 + q^2)   [sigma_z = R_f]
    top (z = +h/2):    X = Y = 0, Z = q0 (static load in +z) or 0 (free vibration).
Free vibration: det F(s) = 0, solved for complex s by Newton/secant iteration
from a starting value (for example the plate-theory root).

Units: any consistent set; scaled internally with L = a, E_ref, rho_ref.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm


class Elastic3D:
    def __init__(self, a, b, section, nsub=1):
        """section: mq3d.Section of plies with constant properties (0 or 90 deg, isotropic).

        nsub: sub-layers per ply (only for output resolution; expm is exact).
        """
        self.a, self.b, self.sec = a, b, section
        self.L = a
        self.E_ref = section.emax()
        self.rho_ref = section.rhomax()
        self.T = self.L * math.sqrt(self.rho_ref / self.E_ref)
        self.plies = []
        for ly in section.layers:
            zm = 0.5 * (ly.zb + ly.zt)
            Q = ly.Qf(zm) / self.E_ref
            if abs(Q[0, 2]) > 1e-12 * Q[0, 0] or abs(Q[3, 4]) > 1e-12 * Q[3, 3]:
                raise ValueError('only cross-ply / orthotropic plies aligned with the edges')
            rho = ly.rf(zm) / self.rho_ref
            for k in range(nsub):
                zb = ly.zb + (ly.zt - ly.zb) * k / nsub
                zt = ly.zb + (ly.zt - ly.zb) * (k + 1) / nsub
                self.plies.append(dict(zb=zb / self.L, zt=zt / self.L, Q=Q, rho=rho))
        self.as_, self.bs = 1.0, b / a

    # scaling of foundation constants (physical -> scaled), as in mq3d
    def kw_s(self, kw): return kw * self.L / self.E_ref
    def ks_s(self, ks): return ks / (self.E_ref * self.L)
    def cw_s(self, cw): return cw / math.sqrt(self.rho_ref * self.E_ref)
    def cs_s(self, cs): return cs / (self.L**2 * math.sqrt(self.rho_ref * self.E_ref))

    @staticmethod
    def state_matrix(Q, rho, p, q, s2):
        C11, C12, C13 = Q[0, 0], Q[0, 1], Q[0, 5]
        C22, C23, C33 = Q[1, 1], Q[1, 5], Q[5, 5]
        C44, C55, C66 = Q[3, 3], Q[4, 4], Q[2, 2]
        R11 = C11 - C13 * C13 / C33
        R12 = C12 - C13 * C23 / C33
        R22 = C22 - C23 * C23 / C33
        A = np.zeros((6, 6), dtype=complex)
        A[0, 2] = -p; A[0, 3] = 1.0 / C55
        A[1, 2] = -q; A[1, 4] = 1.0 / C44
        A[2, 0] = C13 * p / C33; A[2, 1] = C23 * q / C33; A[2, 5] = 1.0 / C33
        A[3, 0] = rho * s2 + p * p * R11 + q * q * C66
        A[3, 1] = p * q * (R12 + C66); A[3, 5] = -C13 * p / C33
        A[4, 0] = p * q * (R12 + C66)
        A[4, 1] = rho * s2 + q * q * R22 + p * p * C66; A[4, 5] = -C23 * q / C33
        A[5, 2] = rho * s2; A[5, 3] = p; A[5, 4] = q
        return A

    def _pq(self, m, n):
        return m * np.pi / self.as_, n * np.pi / self.bs

    def transfer(self, m, n, s_sc):
        p, q = self._pq(m, n)
        T = np.eye(6, dtype=complex)
        for ply in self.plies:
            A = self.state_matrix(ply['Q'], ply['rho'], p, q, s_sc**2)
            T = expm(A * (ply['zt'] - ply['zb'])) @ T
        return T

    def kappa(self, m, n, s_sc, kw=0.0, ks=0.0, cw=0.0, cs=0.0):
        p, q = self._pq(m, n)
        return (self.kw_s(kw) + s_sc * self.cw_s(cw)) + \
               (self.ks_s(ks) + s_sc * self.cs_s(cs)) * (p * p + q * q)

    def Fmat(self, m, n, s_sc, fnd):
        T = self.transfer(m, n, s_sc)
        G = np.zeros((6, 3), dtype=complex)
        G[0, 0] = G[1, 1] = G[2, 2] = 1.0
        G[5, 2] = self.kappa(m, n, s_sc, *fnd)
        return (T @ G)[3:6, :]

    def det(self, m, n, s_sc, fnd):
        return np.linalg.det(self.Fmat(m, n, s_sc, fnd))

    # ------------------------------------------------------------ eigenvalues
    def root(self, m, n, s0, kw=0.0, ks=0.0, cw=0.0, cs=0.0, tol=1e-13, maxit=100, deflate=()):
        """Complex root s (1/s, physical) of det F(s) = 0 near the guess s0 (1/s).

        deflate: roots (1/s) already found; they are divided out of det F so that
        the iteration cannot return to them (used to separate nearby branches).
        """
        fnd = (kw, ks, cw, cs)
        dfl = [complex(r) * self.T for r in deflate]

        def fun(x):
            v = self.det(m, n, x, fnd)
            for r in dfl:
                v = v / (x - r)
            return v

        x0 = complex(s0) * self.T
        x1 = x0 * (1.0 + 1e-4) + 1e-6j * abs(x0)
        f0, f1 = fun(x0), fun(x1)
        for it in range(maxit):
            if f1 == f0:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            x0, f0 = x1, f1
            x1, f1 = x2, fun(x2)
            if abs(x1 - x0) < tol * abs(x1):
                break
        return x1 / self.T

    def omega_undamped(self, m, n, w0, kw=0.0, ks=0.0, deflate=()):
        """Real natural frequency (rad/s) near w0 (rad/s) for the undamped case."""
        s = self.root(m, n, 1j * w0, kw, ks, deflate=[1j * d for d in deflate])
        return abs(s.imag)

    def flexural_share(self, m, n, s, kw=0.0, ks=0.0, cw=0.0, cs=0.0):
        """|W(0)| / max(|U|, |V|, |W|) of the mode at root s (1 for flexural modes)."""
        F = self.fields(m, n, s=s, kw=kw, ks=ks, cw=cw, cs=cs, npl=11)
        k0 = int(np.argmin(np.abs(F['z'])))
        big = max(np.max(np.abs(F['U'])), np.max(np.abs(F['V'])), np.max(np.abs(F['W'])))
        return abs(F['W'][k0]) / big

    # ------------------------------------------------------------------ fields
    def fields(self, m, n, s=0.0, q0=0.0, kw=0.0, ks=0.0, cw=0.0, cs=0.0, npl=41, wnorm=None):
        """Through-thickness amplitudes for a root s (free vibration, q0 = 0) or the
        static sinusoidal load q0 on the top face (s = 0).

        Returns dict z, U, V, W, txz, tyz, szz, sxx, syy, sxy (physical units).
        wnorm: if given, the solution is scaled so that W(0) = wnorm (modes).
        """
        fnd = (kw, ks, cw, cs)
        s_sc = complex(s) * self.T
        p, q = self._pq(m, n)
        F = self.Fmat(m, n, s_sc, fnd)
        if q0 != 0.0:
            c = np.linalg.solve(F, np.array([0.0, 0.0, q0 / self.E_ref], dtype=complex))
        else:
            # null vector of F (smallest singular value)
            _, _, Vh = np.linalg.svd(F)
            c = np.conj(Vh[-1])
        y = np.zeros(6, dtype=complex)
        y[0:3] = c
        y[5] = self.kappa(m, n, s_sc, *fnd) * c[2]
        out = {k: [] for k in ('z', 'U', 'V', 'W', 'txz', 'tyz', 'szz', 'sxx', 'syy', 'sxy')}
        for ply in self.plies:
            A = self.state_matrix(ply['Q'], ply['rho'], p, q, s_sc**2)
            Q = ply['Q']
            zz = np.linspace(ply['zb'], ply['zt'], npl)
            for z in zz:
                yz = expm(A * (z - ply['zb'])) @ y
                U, V, W, X, Y, Z = yz
                Wp = (Z + Q[0, 5] * p * U + Q[1, 5] * q * V) / Q[5, 5]
                out['z'].append(z); out['U'].append(U); out['V'].append(V); out['W'].append(W)
                out['txz'].append(X); out['tyz'].append(Y); out['szz'].append(Z)
                out['sxx'].append(-Q[0, 0] * p * U - Q[0, 1] * q * V + Q[0, 5] * Wp)
                out['syy'].append(-Q[0, 1] * p * U - Q[1, 1] * q * V + Q[1, 5] * Wp)
                out['sxy'].append(Q[2, 2] * (q * U + p * V))
            y = expm(A * (ply['zt'] - ply['zb'])) @ y
        res = {k: np.array(v) for k, v in out.items()}
        if wnorm is not None:
            j = int(np.argmin(np.abs(res['z'])))
            sc = wnorm / res['W'][j]
            for k in res:
                if k != 'z':
                    res[k] = res[k] * sc
        for k in res:
            if k == 'z' or k in ('U', 'V', 'W'):
                res[k] = res[k] * self.L
            else:
                res[k] = res[k] * self.E_ref
        return res

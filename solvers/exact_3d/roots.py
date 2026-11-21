# -*- coding: utf-8 -*-
"""Root drivers of the exact 3D solver.

exact_roots and exact_flexural are copied unchanged from code/plate_study/common.py (SHA-256 e141e10d82d145e841a8a618a8b0365c2519c602ff493683ba34210e83004761).
clpt_omega and flexural_root were written for this package. In the project the upper end of the sign scan was
span x (Navier frequency of the plate theory), which needs the plate-theory code; here it is span x the
classical laminated plate theory frequency, an independent estimate that lies above the exact flexural
root. The window only bounds the scan; the root is the lowest root of family (m, n) whose mode is flexural,
refined with Brent's method, so a wider window does not change it (checked against the stored values by
reproduce.py). If no flexural root is found the window is doubled.
"""
import math

import numpy as np
from scipy.optimize import brentq

from .elast3d import Elastic3D


def exact_roots(e3, m, n, wmax, ngrid=1500, wmin_frac=1e-4):
    """All real natural frequencies (rad/s) of family (m, n) below wmax by a sign scan of det F(i omega) and Brent
    refinement (undamped: F is real for s = i omega)."""
    T = e3.T

    def f(w):
        return np.real(e3.det(m, n, 1j * w * T, (0.0, 0.0, 0.0, 0.0)))

    ws = np.linspace(wmin_frac * wmax, wmax, ngrid)
    d = np.array([f(w) for w in ws])
    out = []
    for i in range(len(ws) - 1):
        if d[i] * d[i + 1] < 0:
            out.append(brentq(f, ws[i], ws[i + 1], xtol=1e-15 * wmax, rtol=1e-14, maxiter=200))
    return np.array(out)


def exact_flexural(e3, m, n, guess, span=1.6):
    """Lowest root of family (m, n) whose mode is flexural (|W(0)| largest displacement amplitude share > 0.5),
    searched below span*guess. Returns (omega, list of all roots found)."""
    r = exact_roots(e3, m, n, span * guess)
    for w in r:
        if e3.flexural_share(m, n, 1j * w) > 0.5:
            return w, r
    return np.nan, r


def clpt_omega(sec, a, b, m, n):
    """Classical laminated plate theory frequency (rad/s) of mode (m, n) of a simply supported cross-ply plate:
    plane-stress reduced ply stiffness, bending stiffness D about the mid-plane (membrane-bending coupling ignored,
    which can only raise the estimate), translational inertia only."""
    D = np.zeros((3, 3))
    I0 = 0.0
    for ly in sec.layers:
        zm = 0.5 * (ly.zb + ly.zt)
        Q = ly.Qf(zm)
        Qr = Q[:3, :3] - np.outer(Q[:3, 5], Q[:3, 5]) / Q[5, 5]
        D += Qr * (ly.zt ** 3 - ly.zb ** 3) / 3.0
        I0 += ly.rf(zm) * (ly.zt - ly.zb)
    p, q = m * math.pi / a, n * math.pi / b
    return math.sqrt((D[0, 0] * p ** 4 + 2.0 * (D[0, 1] + 2.0 * D[2, 2]) * p * p * q * q + D[1, 1] * q ** 4) / I0)


def flexural_root(sec, a, b, m, n, span=1.6, ngrid=3000, maxdouble=4):
    """Exact flexural natural frequency (rad/s) of mode (m, n); returns (omega, info)."""
    e3 = Elastic3D(a, b, sec)
    wmax = span * clpt_omega(sec, a, b, m, n)
    for _ in range(maxdouble + 1):
        for w in exact_roots(e3, m, n, wmax, ngrid=ngrid):
            share = e3.flexural_share(m, n, 1j * w)
            if share > 0.5:
                return w, dict(share=share, window='%.6g' % wmax)
        wmax *= 2.0
    raise RuntimeError('no flexural root found for mode (%d, %d)' % (m, n))

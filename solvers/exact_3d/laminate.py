# -*- coding: utf-8 -*-
"""Ply stiffness and layered-section objects used by the exact 3D solver.

Extracted unchanged (inspect.getsource) from code/plate/mq3d.py of the project, SHA-256 b6a86d5fd0f38ec1b770bbffef2a8e20d011d012a9032976dcfa361c571f96cc:
    Q3D, Q3D_iso, Layer, Section, laminate_section.
No other part of mq3d.py (the mixed finite element) is included in this package.
"""
import math

import numpy as np


def Q3D(E1, E2, E3, G12, G13, G23, nu12, nu13, nu23, theta_deg=0.0):
    """Transformed 3D stiffness, order [xx, yy, xy, yz, xz, zz]."""
    nu21 = nu12 * E2 / E1
    nu31 = nu13 * E3 / E1
    nu32 = nu23 * E3 / E2
    D = (1 - nu12 * nu21 - nu23 * nu32 - nu31 * nu13 - 2 * nu21 * nu32 * nu13) / (E1 * E2 * E3)
    Q11 = (1 - nu23 * nu32) / (E2 * E3 * D)
    Q12 = (nu21 + nu31 * nu23) / (E2 * E3 * D)
    Q13 = (nu31 + nu21 * nu32) / (E2 * E3 * D)
    Q22 = (1 - nu13 * nu31) / (E1 * E3 * D)
    Q23 = (nu32 + nu12 * nu31) / (E1 * E3 * D)
    Q33 = (1 - nu12 * nu21) / (E1 * E2 * D)
    Q44, Q55, Q66 = G23, G13, G12
    c = math.cos(math.radians(theta_deg))
    s = math.sin(math.radians(theta_deg))
    c2, s2 = c * c, s * s
    Qb = np.zeros((6, 6))
    Qb[0, 0] = Q11 * c2 * c2 + 2 * (Q12 + 2 * Q66) * s2 * c2 + Q22 * s2 * s2
    Qb[0, 1] = (Q11 + Q22 - 4 * Q66) * s2 * c2 + Q12 * (s2 * s2 + c2 * c2)
    Qb[1, 1] = Q11 * s2 * s2 + 2 * (Q12 + 2 * Q66) * s2 * c2 + Q22 * c2 * c2
    Qb[0, 2] = (Q11 - Q12 - 2 * Q66) * s * c2 * c + (Q12 - Q22 + 2 * Q66) * s2 * s * c
    Qb[1, 2] = (Q11 - Q12 - 2 * Q66) * s2 * s * c + (Q12 - Q22 + 2 * Q66) * s * c2 * c
    Qb[2, 2] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * s2 * c2 + Q66 * (s2 * s2 + c2 * c2)
    Qb[0, 5] = Q13 * c2 + Q23 * s2
    Qb[1, 5] = Q23 * c2 + Q13 * s2
    Qb[2, 5] = (Q13 - Q23) * c * s
    Qb[5, 5] = Q33
    Qb[3, 3] = Q44 * c2 + Q55 * s2
    Qb[4, 4] = Q55 * c2 + Q44 * s2
    Qb[3, 4] = (Q55 - Q44) * c * s
    for i, j in ((0, 1), (0, 2), (1, 2), (0, 5), (1, 5), (2, 5), (3, 4)):
        Qb[j, i] = Qb[i, j]
    return Qb

def Q3D_iso(E, nu):
    G = E / (2.0 * (1.0 + nu))
    return Q3D(E, E, E, G, G, G, nu, nu, nu, 0.0)

class Layer:
    def __init__(self, zb, zt, Qf, rf, name='', angle=None):
        self.zb, self.zt, self.Qf, self.rf, self.name, self.angle = zb, zt, Qf, rf, name, angle

class Section:
    def __init__(self, layers, name=''):
        self.layers = layers
        self.name = name
        self.zb = layers[0].zb
        self.zt = layers[-1].zt
        self.h = self.zt - self.zb

    def layer_at(self, z, side=0):
        tol = 1e-12 * self.h
        for L in self.layers:
            if side >= 0 and L.zb - tol <= z < L.zt - tol:
                return L
            if side < 0 and L.zb + tol < z <= L.zt + tol:
                return L
        return self.layers[-1] if z > 0 else self.layers[0]

    def Q(self, z, side=0):
        return self.layer_at(z, side).Qf(z)

    def rho(self, z, side=0):
        return self.layer_at(z, side).rf(z)

    def emax(self):
        zs = np.concatenate([np.linspace(L.zb, L.zt, 7) for L in self.layers])
        return max(np.max(np.abs(self.Q(z)[:3, :3])) for z in zs)

    def rhomax(self):
        zs = np.concatenate([np.linspace(L.zb, L.zt, 7) for L in self.layers])
        return max(self.rho(z) for z in zs)

    def scaled(self, L, E_ref, rho_ref):
        new = []
        for ly in self.layers:
            new.append(Layer(ly.zb / L, ly.zt / L,
                             (lambda zz, f=ly.Qf: f(zz * L) / E_ref),
                             (lambda zz, f=ly.rf: f(zz * L) / rho_ref), ly.name, ly.angle))
        return Section(new, self.name)

def laminate_section(props, angles, h, rho=1.0, thicknesses=None, name=''):
    """Orthotropic laminate, plies listed bottom -> top."""
    n = len(angles)
    t = np.full(n, h / n) if thicknesses is None else np.asarray(thicknesses, float)
    z = -h / 2 + np.concatenate([[0.0], np.cumsum(t)])
    layers = []
    for k, th in enumerate(angles):
        Qk = Q3D(props['E1'], props['E2'], props['E3'], props['G12'], props['G13'],
                 props['G23'], props['nu12'], props['nu13'], props['nu23'], th)
        layers.append(Layer(z[k], z[k + 1], (lambda zz, Q=Qk: Q), (lambda zz, r=rho: r),
                            '%g' % th, th))
    return Section(layers, name or '[' + '/'.join('%g' % a for a in angles) + ']')

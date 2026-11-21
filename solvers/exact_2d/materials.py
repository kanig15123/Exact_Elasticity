#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Material laws and their reduction to the planar (x-z) beam problem.

Notation: Voigt order [xx, yy, zz, yz, xz, xy] with engineering shear strains; the ply fibre (axis 1) lies at the
angle `angle_deg` from the beam axis x, measured in the x-y plane (rotation about z).

Reductions to the stresses (sigma_xx, sigma_zz, tau_xz) and strains (eps_xx, eps_zz, gamma_xz) of the beam:

  'narrow'        sigma_yy = tau_xy = tau_yz = 0 (lateral faces of a narrow beam free): the 3x3 compliance is the
                  (xx, zz, xz) sub-matrix of the 3D compliance, and the 3x3 stiffness its inverse.  This follows from
                  the stated stress conditions, not from deleting rows of a plate law.
  'plane_strain'  eps_yy = gamma_xy = gamma_yz = 0 (wide beam / plate strip in cylindrical bending): the 3x3
                  stiffness is the (xx, zz, xz) sub-matrix of the 3D stiffness.

For a ply rotated about z the (xx, zz) and (xz) parts are uncoupled in both reductions (checked numerically).

eps_zz = 0 limit (classical higher-order beam theory, transverse normal stress neglected): sigma_zz = 0 is imposed on
the 3x3 law, so that sigma_xx = Q11r eps_xx with Q11r = Q11 - Q13^2/Q33; with the 'narrow' reduction Q11r = 1/S11
= E_x (uniaxial modulus), with 'plane_strain' Q11r = E1/(1 - nu12 nu21) for a 0 degree ply.
"""
import numpy as np

VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]          # xx yy zz yz xz xy
BEAM = [0, 2, 4]                                                 # xx zz xz


def compliance_ortho(E1, E2, E3, G12, G13, G23, nu12, nu13, nu23):
    """6x6 compliance of an orthotropic ply in its material axes (engineering Voigt notation)."""
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1.0 / E1, 1.0 / E2, 1.0 / E3
    S[0, 1] = S[1, 0] = -nu12 / E1
    S[0, 2] = S[2, 0] = -nu13 / E1
    S[1, 2] = S[2, 1] = -nu23 / E2
    S[3, 3], S[4, 4], S[5, 5] = 1.0 / G23, 1.0 / G13, 1.0 / G12
    return S


def compliance_iso(E, nu):
    G = E / (2.0 * (1.0 + nu))
    return compliance_ortho(E, E, E, G, G, G, nu, nu, nu)


def _to_tensor(C):
    T = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(VOIGT):
        for J, (k, l) in enumerate(VOIGT):
            for (a, b) in {(i, j), (j, i)}:
                for (c, d) in {(k, l), (l, k)}:
                    T[a, b, c, d] = C[I, J]
    return T


def _to_voigt(T):
    C = np.zeros((6, 6))
    for I, (i, j) in enumerate(VOIGT):
        for J, (k, l) in enumerate(VOIGT):
            C[I, J] = T[i, j, k, l]
    return C


def stiffness_rotated(S_mat, angle_deg):
    """6x6 stiffness (engineering Voigt) in global axes of a ply whose axis 1 is at angle_deg from x (about z)."""
    C = np.linalg.inv(S_mat)                     # stiffness Voigt entries equal the tensor components
    if angle_deg == 0.0:
        return C
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])   # global = R material
    Ct = np.einsum('ia,jb,kc,ld,abcd->ijkl', R, R, R, R, _to_tensor(C), optimize=True)
    return _to_voigt(Ct)


def reduce_beam(C6, mode):
    """3x3 stiffness on (eps_xx, eps_zz, gamma_xz) for the reduction `mode` ('narrow' or 'plane_strain')."""
    if mode == 'narrow':
        S6 = np.linalg.inv(C6)
        Q3 = np.linalg.inv(S6[np.ix_(BEAM, BEAM)])
    elif mode == 'plane_strain':
        Q3 = C6[np.ix_(BEAM, BEAM)].copy()
    else:
        raise ValueError('unknown reduction %r' % mode)
    return Q3


class PointLaw:
    """Planar constants at one material point: Q11, Q13, Q33, Q55 (quasi-3D) and Q11r (eps_zz = 0 limit), rho."""

    __slots__ = ('Q11', 'Q13', 'Q33', 'Q55', 'Q11r', 'rho', 'Q3')

    def __init__(self, Q3, rho):
        if abs(Q3[0, 2]) > 1e-9 * abs(Q3[0, 0]) or abs(Q3[1, 2]) > 1e-9 * abs(Q3[0, 0]):
            raise ValueError('normal-shear coupling in the planar law (not a rotation about z?)')
        self.Q3 = Q3
        self.Q11, self.Q13, self.Q33, self.Q55 = Q3[0, 0], Q3[0, 1], Q3[1, 1], Q3[2, 2]
        self.Q11r = self.Q11 - self.Q13**2 / self.Q33
        self.rho = rho


def ply_law(props, angle_deg, rho, mode):
    """PointLaw of an orthotropic ply; props = dict(E1,E2,E3,G12,G13,G23,nu12,nu13,nu23)."""
    S = compliance_ortho(props['E1'], props['E2'], props['E3'], props['G12'], props['G13'], props['G23'],
                         props['nu12'], props['nu13'], props['nu23'])
    return PointLaw(reduce_beam(stiffness_rotated(S, angle_deg), mode), rho)


def iso_law(E, nu, rho, mode):
    """PointLaw of an isotropic material point.  The 'narrow' law is formed from the compliance sub-matrix directly,
    so that nu = 0.5 (singular 3D stiffness, regular planar compliance) is admissible."""
    S = compliance_iso(E, nu)
    if mode == 'narrow':
        return PointLaw(np.linalg.inv(S[np.ix_(BEAM, BEAM)]), rho)
    return PointLaw(reduce_beam(np.linalg.inv(S), mode), rho)


def iso_constants(E, nu, mode):
    """Closed forms (for tests): (Q11, Q13, Q33, Q55, Q11r)."""
    G = E / (2.0 * (1.0 + nu))
    if mode == 'narrow':
        q = E / (1.0 - nu**2)
        return q, nu * q, q, G, E
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam + 2.0 * G, lam, lam + 2.0 * G, G, E / (1.0 - nu**2)

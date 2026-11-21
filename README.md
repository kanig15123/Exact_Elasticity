# Exact elasticity reference solutions for the free vibration of laminated beams and plates

Reproducibility package of the manuscript *Mixed quasi-3D finite elements for the free vibration of laminated beams
and plates: a critical review with elasticity benchmarks* (prepared for Archives of Computational Methods in
Engineering).

**Status:** prepared for the author's review; not yet published. Repository URL: *to be supplied by the author.*
Licence: not yet chosen (see `LICENSE_NOTE.md`).

## 1. What this package is, and what it is not

The primary subject of the manuscript is the **mixed quasi-3D Hellinger-Reissner finite element formulation** for
laminated beams and plates. To assess it, the study computed independent **exact elasticity reference solutions**.
This package contains only those reference solvers, their inputs and their results, so that every exact reference
value of the manuscript can be regenerated and checked.

The package does **not** contain:
- the mixed quasi-3D finite element code;
- the Navier solutions of the beam and plate theories;
- the stability tests;
- the code of the two numerical finite element reference models (plane-stress Q9 beams, solid H27 plates).

The results of the two finite element reference models are included as data only, in
`reference_results/numerical_fe_references/`:
- Q9 values of the clamped and cantilever parametric beams (manuscript Tables 11, 15);
- Q9 values of the clamped data sets of Chen et al. (Table 10);
- H27 values of the clamped plates.

They are numerical approximations, not exact solutions, and `reproduce.py` does not check them.

## 2. Which solutions are exact, and under which assumptions

| Solver | Problem | Exact within these assumptions | Only numerical approximation |
|---|---|---|---|
| `solvers/exact_2d` (SSBeam2D) | simply supported layered beam, plane problem in (x, z) | linear elasticity; plane stress (narrow beam, sigma_yy = tau_xy = tau_yz = 0) or plane strain; orthotropic layers rotated about z only; perfect bonding; traction-free faces; diaphragm ends (w = 0, sigma_xx = 0, u free) | root of the frequency equation (floating point) |
| `solvers/exact_3d` (Elastic3D) | simply supported rectangular cross-ply plate | linear 3D elasticity; plies with principal axes along the edges (0/90 degrees or isotropic); perfect bonding; traction-free faces; diaphragm edges | root of the frequency equation (floating point) |

For these problems the displacement field separates exactly into a trigonometric function of the in-plane
coordinates and an amplitude through the thickness, which obeys a linear ordinary differential equation in each layer.
The equation is integrated exactly by the matrix exponential, and no thickness discretisation is involved. The only
approximation is the numerical evaluation of the frequency equation and of its roots (Section 3).

Clamped and free ends and clamped plates have no such solution. The manuscript uses finite element references for
them (plane-stress Q9, solid H27), which are not part of the exact solvers.

## 3. Numerical algorithms and precision

- **State space.** Beam: y = [U, W, X, Z] (displacements and transverse stresses), y' = A y per layer. Plate: y = [U, V,
  W, X, Y, Z]. The transfer matrix is the product of `scipy.linalg.expm(A_k t_k)` over the layers. The beam uses four
  sub-layers per layer, which affects only the output resolution.
- **Frequency equation.** Traction-free faces give det B(omega) = 0, with B a 2x2 block for beams (scaled by
  max|B|^2) and a 3x3 block for plates.
- **Roots.** A sign-change scan of det B on a uniform grid finds the roots, which are refined with Brent's method:
  - beams: `scipy.optimize.brentq`, xtol 1e-14, rtol 1e-13, 4000 grid points;
  - plates: xtol 1e-15 omega_max, rtol 1e-14, 3000 grid points.
- **Mode identification.** The flexural root is the lowest root whose mode is flexural:
  - beams: transverse share of the kinetic energy >= 0.5;
  - plates: |W(0)| / max(|U|, |V|, |W|) > 0.5.
- **Scan window.** The project computed the stored values with an upper scan limit of 1.5 (beams) or 1.6 (plates)
  times the Navier frequency of the plate or beam theory. This package uses 1.5 times the Euler-Bernoulli frequency
  (beams) and 1.6 times the classical-laminated-plate frequency (plates) instead. These estimates are independent of
  the theory code and lie above the exact flexural root; if no flexural root is found, the window is doubled.
  `solvers/*/roots.py` documents which functions are unchanged project code and which were written for the package.
- **Precision.** Agreement between the recomputed and the stored values:
  - better than 1e-8 relative for the values stored at full precision (largest observed 3e-10 for the parametric
    beams);
  - within the stored digits (6 to 8 significant digits) for the rest.

  The exception is short-wavelength modes of thick, high-contrast sections. For modes 8 and 9 of DATA-4 the
  determinant is about 1e-17 to 1e-18 near the root and its sign changes inside the rounding noise, so the computed
  root moves with the scan grid by up to 2e-7 (mode 8) and 5e-6 (mode 9). The project's original window reproduces
  the stored values exactly; the tolerance of these two cases is set from the observed scatter and stated in
  `reference_results/exact_reference_results.csv` (column `precision_note`). Neither mode is printed in a manuscript
  table; both are plotted in Fig. 18a, where the model errors are 22 to 49 per cent, so a relative change of 5e-6 in
  the reference cannot be seen.

## 4. How to reproduce

```
pip install -r requirements.txt
python reproduce.py            # all 266 cases, about 5 minutes on a workstation
python reproduce.py --quick    # one case per benchmark set
python reproduce.py --set P-L2 # one benchmark set
```

The script uses only the files of this package (paths relative to `reproduce.py`). It writes:
- `validation/reproduced_results.csv`: every case, recomputed and stored value, relative difference, check;
- `validation/validation_report.md`: summary by set and comparison with published values.

Its exit code is the number of cases outside tolerance. Verified on Python 3.13.9, NumPy 2.3.5, SciPy 1.16.3 from a
copy of this folder outside the project.

## 5. Benchmarks and the manuscript tables that use them

| Set | Definition | Solver | Cases | Manuscript |
|---|---|---|---|---|
| B-PAR | [0/90/0] and [0/90] beams (E1/E2 = 40, G12 = G13 = 0.6 E2, G23 = 0.5 E2, nu = 0.25) and the DATA-1 sandwich; L/h = 5-100; modes 1-3 | exact 2D, plane stress | 45 | Tables 11, 15, 20; Figs. 9, 10, 15, 16, 19 (S-S cases) |
| B-CHEN | Chen et al. (2003) DATA-1, DATA-2, DATA-4, S-S, modes 1-9 | exact 2D, plane stress | 27 | Table 9 (modes 1-5); Table 15 (DATA-2, mode 1); Fig. 18a (DATA-4); modes 6-9 of DATA-1 and DATA-2 are not printed |
| B-K5, B-K4, B-SW | Kapuria et al. (2004) five-ply beam, (90/0/90/0) beam and soft-core sandwich beam as defined by Wu Zhen and Chen Wanji (2008) | exact 2D, plane stress and plane strain | 72 | Tables 2, 4 |
| P-PAR | square SSSS [0/90/0], [0/90/90/0], [0/90] plates (same material), a/h = 5-100, modes (1,1) to (2,2) | exact 3D | 60 | Tables 12, 16, 20, 21, 23; Figs. 9, 12, 17, 20 (SSSS) |
| P-RP, P-NOOR | square SSSS cross-ply plates of Reddy and Phan (1985) and Noor (1973) as quoted | exact 3D | 39 | Tables 2, 3 |
| P-L2 | sandwich laminate L2 of Iurlaro et al. (2013), both normalisations printed there | exact 3D | 23 | Tables 2, 3 |

Table and figure numbers refer to the compiled manuscript of 21 November 2026; the column `manuscript_use` of
`reference_results/exact_reference_results.csv` gives them per case. The mode shapes and stresses of the same solvers
(Tables 18, 19, 22; Figs. 19 to 22) are not reproduced by this package.

The material constants, stacking sequences, thicknesses, boundary conditions and frequency normalisations are in
`benchmark_inputs/` (`materials.json`, `laminates.json`, `cases.json`). They are written from the project's own
definitions, and every packaged law was checked against the project's objects (largest difference 6e-17).

Assumptions needed for the Chen et al. data, because the paper does not state them:
- plane stress;
- isotropic cores with E = 2G(1 + nu), nu = 0.3;
- plies with E3 = E2, G13 = G23 = G12, nu13 = nu12, nu23 = 0.3.

They are recorded in the material notes.

## 6. Published benchmarks reproduced independently

From `validation/validation_report.md` (largest relative difference, recomputed against published value):

| Published values | Kind | n | Result |
|---|---|---|---|
| Kapuria et al. (2004) as printed by Wu Zhen and Chen Wanji (2008), Tables 2-3 | exact 2D | 22 | within 1.5e-4 |
| Rao et al. (2004) as printed by Iurlaro et al. (2013) and Grover et al. (2013) | exact 3D | 12 | within 2.3e-4 |
| Chen et al. (2003), DATA-2 and DATA-4 modes 2-9 | exact 2D (as printed) | 17 | within 1.2e-4 |
| Chen et al. (2003), DATA-1 | exact 2D (as printed) | 9 | 3e-4 to 7e-4, depending on the assumed core Young modulus |
| Desai et al. (2003) as quoted by Band and Desai (2015) | numerical 3D mixed FE | 7 | within 6.8e-4 |
| Rao and Desai (2004) as quoted by Iurlaro et al. (2013) | numerical mixed layerwise | 12 | within 5.7e-4 |

`validation/stored_project_checks/` holds two stored project records, which `reproduce.py` does not regenerate:
- the project's comparison with further published 3D values (Noor-type laminates as printed by later papers; the
  Srinivas et al. single-layer plate as quoted);
- the reproduction summary used for manuscript Table 2.

## 7. Unresolved discrepancies

These are reported as discrepancies whose cause is unknown, not as errors of the sources:

- **Noor (1973), as quoted** by Band and Desai (2015) and Grover et al. (2013): the 26 quoted values (25 cases)
  differ from the exact 3D solution by -0.81 % to +0.78 % (quoted value relative to the exact value, as in manuscript
  Table 2; `validation_report.md` gives the inverse ratio, -0.77 % to +0.82 %). The primary paper could not be
  obtained.
- **Chen et al. (2003), DATA-4 mode 1:** the exact solution gives 1.65495, the printed exact value is 1.6595
  (-2.7e-3). Modes 2-9 agree within 1.2e-4. The printed exact and SS-DQM values of mode 1 agree with each other.

## 8. Relation to the mixed quasi-3D finite element formulation

The exact solvers do not use the kinematics, the shear functions, the stress resultants or the discretisation of the
mixed formulation. They solve the equations of linear elasticity for the same material, geometry and boundary
conditions, and serve only as reference values. A reader can check every exact reference value of the manuscript
with this package. Checking the finite element results themselves requires the mixed finite element code, which is
not part of this package.

## 9. Contents and provenance

```
github_exact_elasticity/
├── README.md, LICENSE_NOTE.md, requirements.txt, MANIFEST.md
├── reproduce.py
├── solvers/exact_2d/   beam2d_exact.py, materials.py (unchanged project code), roots.py
├── solvers/exact_3d/   elast3d.py (unchanged), laminate.py (extracted unchanged), roots.py
├── benchmark_inputs/   materials.json, laminates.json, cases.json
├── reference_results/  exact_reference_results.csv; numerical_fe_references/ (Q9, H27 data, not exact)
└── validation/         published_values.csv, stored_project_checks/, outputs of reproduce.py
```

`MANIFEST.md` lists the origin of every solver file with the SHA-256 checksum of the project source, and the
checksum of every package file.

## 10. Not included or not reproducible here

- Code of the plane-stress Q9 and solid H27 reference models (data only).
- An exact solution for DATA-3 and the C-F case of DATA-1 of Chen et al. (2003): they have clamped or free ends, so
  none exists. Their Q9 values are included as numerical data only.
- The Navier solutions of the beam and plate theories quoted beside the exact values in manuscript Tables 2-4.
- The primary paper of Noor (1973), so the quoted values cannot be checked against their source.
- Plate thin-limit values (a/h up to 1000) and the exact modal stress profiles of the manuscript, which were
  computed with the same solvers but are not part of `cases.json`.

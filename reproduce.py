#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regenerate the exact elasticity reference frequencies of the ACME manuscript from the packaged solvers and inputs.

    python reproduce.py            all cases (about 5-10 minutes on a workstation)
    python reproduce.py --quick    one case per benchmark set (about one minute)
    python reproduce.py --set P-L2 only the cases of one set (see benchmark_inputs/cases.json)

For every case in benchmark_inputs/cases.json the script
  1. builds the layered section from benchmark_inputs/materials.json and laminates.json,
  2. finds the exact natural frequency with the packaged solver (solvers/exact_2d or solvers/exact_3d),
  3. converts it to the normalised quantity of the case,
  4. compares it with the value stored in reference_results/exact_reference_results.csv, which was copied from the
     full-precision project data, and with published values in validation/published_values.csv where they exist.
Outputs (overwritten): validation/reproduced_results.csv, validation/validation_report.md.
Exit code: number of cases whose recomputed value differs from the stored value by more than the tolerance.
Only the Python standard library, NumPy and SciPy are used; all paths are relative to this file.
"""
import argparse
import csv
import json
import math
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import scipy  # noqa: E402

from solvers.exact_2d import materials as mt  # noqa: E402
from solvers.exact_2d.beam2d_exact import Layer2D  # noqa: E402
from solvers.exact_2d.roots import flexural_root as beam_flexural_root  # noqa: E402
from solvers.exact_3d import laminate as lam3d  # noqa: E402
from solvers.exact_3d.elast3d import Elastic3D  # noqa: E402
from solvers.exact_3d.roots import flexural_root as plate_flexural_root  # noqa: E402

INP = os.path.join(HERE, 'benchmark_inputs')
REF = os.path.join(HERE, 'reference_results')
VAL = os.path.join(HERE, 'validation')


def load_json(name):
    with open(os.path.join(INP, name), encoding='utf-8') as fh:
        return json.load(fh)


def read_csv(path):
    with open(path, encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ sections
def beam_layers(spec, materials, reduction):
    """Layer2D list (bottom to top) of a 2D laminate; thicknesses in the units of the laminate (z from -H/2)."""
    H = sum(l['t'] for l in spec['layers'])
    z = -H / 2.0
    out = []
    for l in spec['layers']:
        m = materials[l['material']]
        if m['type'] == 'iso':
            law = mt.iso_law(m['E'], m['nu'], l['rho'], reduction)
        else:
            props = {k: m[k] for k in ('E1', 'E2', 'E3', 'G12', 'G13', 'G23', 'nu12', 'nu13', 'nu23')}
            law = mt.ply_law(props, l['angle'], l['rho'], reduction)
        out.append(Layer2D(z, z + l['t'], law))
        z += l['t']
    return out, H


def plate_section(spec, materials):
    """laminate.Section of a 3D laminate (plies bottom to top, constant properties per ply)."""
    H = sum(l['t'] for l in spec['layers'])
    z = -H / 2.0
    layers = []
    for l in spec['layers']:
        m = materials[l['material']]
        if m['type'] == 'iso':
            Q = lam3d.Q3D_iso(m['E'], m['nu'])
        else:
            Q = lam3d.Q3D(m['E1'], m['E2'], m['E3'], m['G12'], m['G13'], m['G23'], m['nu12'], m['nu13'], m['nu23'],
                          l['angle'])
        layers.append(lam3d.Layer(z, z + l['t'], (lambda zz, Q=Q: Q), (lambda zz, r=l['rho']: r), '', l['angle']))
        z += l['t']
    return lam3d.Section(layers, spec.get('name', '')), H


def normalise(omega, nrm):
    t = nrm['type']
    if t == 'omega_L2_over_H':          # omega L^2/H sqrt(rho_ref/E_ref)
        return omega * nrm['L'] ** 2 / nrm['H'] * math.sqrt(nrm['rho_ref'] / nrm['E_ref'])
    if t == 'omega_a2_over_h':          # omega a^2/h sqrt(rho_ref/E_ref); h as defined by the source (see cases.json)
        return omega * nrm['a'] ** 2 / nrm['h'] * math.sqrt(nrm['rho_ref'] / nrm['E_ref'])
    if t == 'hertz':
        return omega / (2.0 * math.pi)
    if t == 'rad_per_s':
        return omega
    raise ValueError('unknown normalisation %r' % t)


def solve_case(c, materials, laminates):
    if c['solver'].startswith('exact_2d'):
        layers, _ = beam_layers(laminates[c['laminate']], materials, c['reduction'])
        w, info = beam_flexural_root(layers, c['L'], c['m'])
    else:
        sec, _ = plate_section(laminates[c['laminate']], materials)
        w, info = plate_flexural_root(sec, c['a'], c['b'], c['m'], c['n'])
    return w, info


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--quick', action='store_true', help='one case per set')
    ap.add_argument('--set', default=None, help='restrict to one benchmark set')
    args = ap.parse_args()
    materials = load_json('materials.json')
    laminates = load_json('laminates.json')
    cases = load_json('cases.json')['cases']
    if args.set:
        cases = [c for c in cases if c['set'] == args.set]
    if args.quick:
        seen, sel = set(), []
        for c in cases:
            if c['set'] not in seen:
                seen.add(c['set'])
                sel.append(c)
        cases = sel
    stored = {r['id']: r for r in read_csv(os.path.join(REF, 'exact_reference_results.csv'))}
    published = {}
    for r in read_csv(os.path.join(VAL, 'published_values.csv')):
        published.setdefault(r['id'], []).append(r)
    rows, fails = [], 0
    t0 = time.time()
    for k, c in enumerate(cases):
        omega, info = solve_case(c, materials, laminates)
        val = normalise(omega, c['normalisation'])
        s = stored[c['id']]
        ref = float(s['value'])
        tol = float(s['tolerance_rel'])
        d = val / ref - 1.0
        ok = abs(d) <= tol
        fails += not ok
        rows.append(dict(id=c['id'], set=c['set'], solver=c['solver'], case=c['case'], mode=c['mode'],
                         quantity=c['normalisation']['label'], recomputed='%.12g' % val, stored=s['value'],
                         rel_diff='%.3e' % d, tolerance_rel=s['tolerance_rel'], check='OK' if ok else 'FAIL',
                         flexural_share='%.4f' % info['share'], window_estimate=info['window'],
                         precision_note=s.get('precision_note', '')))
        print('[%3d/%3d] %-44s %s  recomputed %.8g  stored %s  rel %.1e  %s' % (
            k + 1, len(cases), c['id'], c['normalisation']['label'][:24], val, s['value'], d, 'OK' if ok else 'FAIL'),
            flush=True)
    with open(os.path.join(VAL, 'reproduced_results.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # published comparison
    rec = {r['id']: float(r['recomputed']) for r in rows}
    prow = []
    for cid, lst in published.items():
        if cid not in rec:
            continue
        for p in lst:
            pv = float(p['published'])
            prow.append((p, rec[cid], rec[cid] / pv - 1.0))
    write_report(rows, prow, fails, time.time() - t0, args)
    print('\n%d cases, %d outside tolerance; report: validation/validation_report.md' % (len(rows), fails))
    return fails


def write_report(rows, prow, fails, secs, args):
    L = ['# Validation report (generated by reproduce.py)', '',
         'Environment: Python %s, NumPy %s, SciPy %s, %s. Run time %.0f s. Options: %s.' % (
             platform.python_version(), np.__version__, scipy.__version__, platform.platform(), secs,
             ('--quick ' if args.quick else '') + ('--set %s' % args.set if args.set else '') or 'all cases'),
         '', '## 1. Recomputed against the stored project values', '',
         '| Set | Cases | Largest relative difference | Outside tolerance |', '|---|---|---|---|']
    by = {}
    for r in rows:
        by.setdefault(r['set'], []).append(r)
    for s, rr in by.items():
        L.append('| %s | %d | %.1e | %d |' % (s, len(rr), max(abs(float(r['rel_diff'])) for r in rr),
                                            sum(1 for r in rr if r['check'] != 'OK')))
    L += ['', 'Tolerance per case: 1e-8 relative for values stored at full precision, half a unit of the last stored '
          'digit otherwise (column tolerance_rel of reference_results/exact_reference_results.csv). '
          'Total: %d cases, %d outside tolerance.' % (len(rows), fails), '']
    lim = [r for r in rows if r.get('precision_note')]
    if lim:
        L += ['Cases with a documented precision limit (tolerance set from the observed root scatter, see README):', '']
        L += ['- %s: relative difference %s (tolerance %s). %s' % (r['id'], r['rel_diff'], r['tolerance_rel'],
                                                                  r['precision_note']) for r in lim]
        L += ['']
    L += [
          '## 2. Recomputed against published values', '',
          'Published values are transcribed from the cited tables (validation/published_values.csv); "kind" says '
          'whether the published value is itself an exact elasticity solution or a numerical solution.', '',
          '| Published source (transcribed from) | Kind | n | Largest relative difference | Status |',
          '|---|---|---|---|---|']
    grp = {}
    for p, v, d in prow:
        key = (p['published_source'], p['transcribed_from'], p['kind'], p['status'])
        grp.setdefault(key, []).append(d)
    for (src, tr, kind, st), ds in sorted(grp.items()):
        L.append('| %s (%s) | %s | %d | %+.2e | %s |' % (src, tr, kind, len(ds), max(ds, key=abs), st))
    L += ['', 'Status "reproduced" means agreement to the precision of the published digits or of the published '
          'numerical method; "unresolved discrepancy" means that the cause of the difference is not known; it is not '
          'claimed to be an error of the source.']
    with open(os.path.join(VAL, 'validation_report.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    sys.exit(main())

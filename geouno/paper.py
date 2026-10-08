"""Recompute the manuscript's headline numbers from the bundled Source Data.

Inputs are the packaged Source Data in ``geouno/resources/source_data`` (copied from the manuscript's
scientific audit, ``scientific_audit_20261002/source_data``). Each recomputed value is
rounded as printed and compared with the manuscript (revision of 7 October 2026).
"""
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .artifacts import resource

SOURCE = resource('source_data')

# Values as printed in the manuscript (Table 1, Table 2, Supplementary Table 1, abstract).
PRINTED = {
    'Table 1': {
        'GeoUNO': ((.369, .006), (.413, .005), (.964, .011), (.412, .004)),
        'DeepONet': ((1.650, .003), (0.779, .014), (2.258, .020), (0.790, .023)),
        'FNO': ((1.618, .026), (1.431, .050), (2.115, .011), (1.564, .037)),
        'U-FNO': ((0.603, .042), (0.693, .010), (1.271, .022), (0.806, .019)),
        'U-Net': ((0.566, .027), (0.559, .019), (1.218, .026), (0.596, .030)),
        'ResNet': ((1.145, .046), (0.635, .012), (1.447, .059), (0.730, .025)),
        'PCNO': ((0.587, .009), (0.526, .014), (1.358, .136), (0.537, .013)),
    },
    'Table 2 (24 reservoirs)': {
        'Prior': ((1.013, .162), (12.664, 3.001), (17.233, 12.273)),
        'Static conditioning': ((0.783, .149), (9.423, 2.564), (3.452, 5.115)),
        'MAP': ((0.663, .170), (6.369, 2.600), (0.489, .314)),
        'ES-MDA': ((0.679, .173), (6.324, 2.192), (0.562, .368)),
    },
    'Table 2 (six reservoirs)': {
        'Prior': ((0.973, .121), (13.010, 1.552), (20.970, 12.796)),
        'Static conditioning': ((0.771, .152), (11.968, .883), (6.281, 8.394)),
        'MAP': ((0.657, .133), (8.775, 3.453), (0.596, .474)),
        'ES-MDA': ((0.667, .163), (8.199, 2.793), (0.673, .444)),
        'NUTS': ((0.624, .131), (6.615, 1.657), (0.654, .426)),
    },
    'Supplementary Table 1': {
        'DeepONet': ((1.784, .030), (0.597, .013), (4.285, .082), (0.851, .018)),
        'FNO': ((1.031, .009), (0.742, .008), (2.479, .010), (1.089, .016)),
        'U-FNO': ((0.628, .014), (0.579, .018), (1.467, .040), (0.836, .039)),
        'U-Net': ((0.563, .012), (0.447, .007), (1.359, .039), (0.643, .008)),
        'ResNet': ((1.276, .019), (0.487, .012), (3.246, .068), (0.723, .023)),
    },
    'Abstract': {
        'pressure-field error reduction vs prior (%)': 97,
        'temperature-field error reduction vs prior (%)': 50,
        'temperature-field RMSE at 33 locations, unrestricted multiwell (K)': 4.65,
        'temperature-field RMSE at 33 locations, vertical monitoring well (K)': 10.08,
        'temperature-field RMSE at 33 locations, inclined monitoring well (K)': 6.79,
    },
}

MODEL_NAMES = {'deeponet': 'DeepONet', 'fno': 'FNO', 'ufno': 'U-FNO', 'unet': 'U-Net', 'resnet': 'ResNet',
               'pcno': 'PCNO', 'geouno': 'GeoUNO'}
SIX = [f'test_{i:02d}' for i in range(1, 7)]
# Reporting policy of 4 October 2026: the author retired the attention U-Net comparator from all
# manuscript tables; its archived runs remain in the Source Data but are not reported.
RETIRED = {'attention_unet'}


def _mean_sd(values):
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite Source Data')
    return statistics.mean(values), statistics.stdev(values)


def monitoring_table(path=None):
    """``{(configuration, metric): {reservoir: value}}`` from the monitoring Source Data."""
    rows = defaultdict(dict)
    with Path(path or SOURCE / 'monitoring_primary_source_data.csv').open() as handle:
        for r in csv.DictReader(handle):
            group = rows[(r['configuration'], r['metric'])]
            if r['reservoir'] in group:
                raise ValueError('Duplicate reservoir in monitoring Source Data')
            group[r['reservoir']] = float(r['value'])
    return rows


def geouno_test_runs():
    """Independent equal-scenario/time means from all three frozen FP32 test runs.

    The transferred application model is deliberately excluded. Missing, duplicated,
    non-finite or mismatched scenario/time records fail instead of changing the average.
    """
    folder = SOURCE / 'table1_three_seed_20261007'
    with gzip.open(folder / 'cohort.json.gz', 'rt') as handle:
        cohort = json.load(handle)
    expected = {(name, int(t)): len(members) for name, t, members in cohort}
    if len(expected) != 618 * 11 or len(cohort) != len(expected):
        raise ValueError('Incomplete or duplicate test cohort')
    columns = ('T_field_rmse', 'P_field_rmse', 'T_nearwell_rmse', 'P_nearwell_rmse')
    runs = defaultdict(dict)
    with gzip.open(folder / 'scenario_time_scores.csv.gz', 'rt') as handle:
        for row in csv.DictReader(handle):
            seed, key = int(row['seed']), (row['scenario'], int(row['time_index']))
            if key in runs[seed] or expected.get(key) != int(row['members']):
                raise ValueError('Duplicate or mismatched test record')
            values = tuple(float(row[k]) for k in columns)
            if not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError('Invalid test metric')
            runs[seed][key] = values
    if set(runs) != {20260925, 20260926, 20260927} or any(set(r) != set(expected) for r in runs.values()):
        raise ValueError('Incomplete three-seed test results')
    return [tuple(statistics.mean(c) for c in zip(*runs[s].values())) for s in sorted(runs)]


def table1():
    by = defaultdict(list)
    with (SOURCE / 'table1_nearwell' / 'run_summary.csv').open() as handle:
        for r in csv.DictReader(handle):
            if r['model'] in RETIRED or r['model'] == 'geouno':
                continue
            by[MODEL_NAMES[r['model']]].append([float(r[k]) for k in ('T_field_rmse', 'P_field_rmse',
                                                                      'T_nearwell_rmse', 'P_nearwell_rmse')])
    by['GeoUNO'] = geouno_test_runs()
    out = {}
    for name, runs in by.items():
        cols = list(zip(*runs))
        out[name] = tuple(c[0] for c in cols) if len(runs) == 1 else tuple(_mean_sd(c) for c in cols)
    return out


def table2():
    m = monitoring_table()
    nuts = {}
    with (SOURCE / 'nuts_20261005' / 'per_reservoir.csv').open() as handle:
        for r in csv.DictReader(handle):
            nuts[r['case']] = r
    methods = {'Prior': 'prior', 'Static conditioning': 'static_prior', 'MAP': 'multiwell_opt17/map',
               'ES-MDA': 'multiwell_opt17/esmda'}
    metrics = ('log10_permeability_RMSE_dec', 'temperature_field_RMSE_K', 'pressure_field_RMSE_MPa')
    out = {'Table 2 (24 reservoirs)': {}, 'Table 2 (six reservoirs)': {}}
    for label, config in methods.items():
        values = [m[(config, k)] for k in metrics]
        assert all(len(v) == 24 for v in values)
        out['Table 2 (24 reservoirs)'][label] = tuple(_mean_sd(list(v.values())) for v in values)
        out['Table 2 (six reservoirs)'][label] = tuple(_mean_sd([v[c] for c in SIX]) for v in values)
    cols = ('logk_rmse', 'temperature_field_rmse', 'pressure_field_rmse')
    out['Table 2 (six reservoirs)']['NUTS'] = tuple(_mean_sd([float(nuts[c][k]) for c in SIX]) for k in cols)
    return out


def supplementary_table1():
    by = defaultdict(list)
    with (SOURCE / 'uniform_baselines_20261004' / 'per_seed.csv').open() as handle:
        for r in csv.DictReader(handle):
            if r['model'] in RETIRED:
                continue
            by[MODEL_NAMES[r['model']]].append([float(r[k]) for k in (
                'all_times_T_paired_rmse', 'all_times_P_paired_rmse', 'ten_years_T_paired_rmse',
                'ten_years_P_paired_rmse')])
    return {name: tuple(_mean_sd(c) for c in zip(*runs)) for name, runs in by.items()}


def abstract():
    m = monitoring_table()

    def mean(config, metric):
        return statistics.mean(m[(config, metric)].values())
    t, p = 'temperature_field_RMSE_K', 'pressure_field_RMSE_MPa'
    return {
        'pressure-field error reduction vs prior (%)': 100 * (1 - mean('multiwell_opt17/esmda', p) / mean('prior', p)),
        'temperature-field error reduction vs prior (%)':
            100 * (1 - mean('multiwell_opt17/esmda', t) / mean('prior', t)),
        'temperature-field RMSE at 33 locations, unrestricted multiwell (K)': mean('multiwell_opt33/esmda', t),
        'temperature-field RMSE at 33 locations, vertical monitoring well (K)': mean('vertical_opt33/esmda', t),
        'temperature-field RMSE at 33 locations, inclined monitoring well (K)': mean('inclined_opt33/esmda', t),
    }


def check():
    """Return a list of ``(table, row, column, recomputed, printed, ok)``."""
    computed = {'Table 1': table1(), **table2(), 'Supplementary Table 1': supplementary_table1(),
                'Abstract': abstract()}
    report = []
    if set(computed) != set(PRINTED):
        raise ValueError('Computed table set differs from printed tables')
    for table, rows in PRINTED.items():
        if set(computed[table]) != set(rows):
            raise ValueError(f'Computed rows differ from printed rows: {table}')
        for row, printed in rows.items():
            value = computed[table][row]
            if not isinstance(printed, tuple):
                ok = round(value) == printed if isinstance(printed, int) else round(value, 2) == printed
                report.append((table, row, '', value, printed, ok))
                continue
            for col, (v, p) in enumerate(zip(value, printed, strict=True)):
                if isinstance(p, tuple):
                    if len(v) != len(p):
                        raise ValueError(f'Wrong statistic count: {table}/{row}')
                    ok = round(v[0], 3) == p[0] and round(v[1], 3) == p[1]
                else:
                    ok = round(v, 3) == p
                report.append((table, row, col, v, p, ok))
    return report


def main():
    report = check()
    width = max(len(f'{t} / {r}') for t, r, *_ in report)
    for table, row, col, value, printed, ok in report:
        shown = (f'{value[0]:.3f} +/- {value[1]:.3f}' if isinstance(value, tuple)
                 else f'{value:.4f}')
        label = f'{table} / {row}' + (f' [{col}]' if col != '' else '')
        print(f'{"OK " if ok else "BAD"} {label:<{width + 5}} recomputed {shown:<22} printed {printed}')
    failures = [r for r in report if not r[-1]]
    print(json.dumps(dict(checked=len(report), mismatches=len(failures))))
    return 1 if failures else 0

# SPDX-License-Identifier: LGPL-2.1-only
"""Run the supplied geothermal FE case with an installed cavity executable."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

import numpy as np

from common import HERE, generator, sha256, verify_files


def compare(output):
    reference = np.load(HERE / 'reference/fields.npz')
    model = generator()
    collected = output / 'collected'
    collected.mkdir()
    if not model.collect_and_clean_v4('case', output, collected, reference['z'], rc=0):
        raise ValueError('The solver output did not reach the required final time or is incomplete.')
    result_path = collected / 'case.npz'
    result = np.load(result_path)
    comparisons = {}
    for name in reference.files:
        actual, expected = result[name], reference[name]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f'Output layout changed: {name}')
        item = dict(shape=list(actual.shape), dtype=str(actual.dtype),
                    byte_equal=np.array_equal(actual, expected))
        if np.issubdtype(actual.dtype, np.number):
            if not np.isfinite(actual).all():
                raise ValueError(f'Non-finite output: {name}')
            error = actual.astype('f8') - expected.astype('f8')
            item.update(max_absolute=float(np.abs(error).max()),
                        rmse=float(np.sqrt(np.square(error).mean())))
        comparisons[name] = item
    return dict(comparisons=comparisons,
                all_arrays_byte_equal=all(item['byte_equal'] for item in comparisons.values()),
                output_sha256=sha256(result_path), reference_sha256=sha256(HERE / 'reference/fields.npz'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', required=True, help='Installed cavity-opt executable')
    parser.add_argument('--launcher', default='mpiexec', help='MPI launcher and options, e.g. "srun --mpi=pmi2"')
    parser.add_argument('--ranks', type=int, default=6)
    parser.add_argument('--out', required=True, type=Path, help='New output directory')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare the input and print the command')
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error('--ranks must be positive')
    output = args.out.resolve()
    if output.exists():
        parser.error('--out already exists; choose a new directory')
    executable = shutil.which(args.executable)
    if executable is None:
        parser.error('--executable could not be found')
    verify_files()
    output.mkdir(parents=True)
    deck = (HERE / 'case/case.i').read_text()
    for name, path in {'mesh_file': HERE / 'case/mesh.msh',
                       'perm_block_file': HERE / 'case/permeability.i',
                       'prod_bh_file': HERE / 'case/producer.bh'}.items():
        deck, count = re.subn(rf'^{name}\s*=.*$', lambda _: name + " = '" + str(path) + "'",
                             deck, count=1, flags=re.M)
        if count != 1:
            raise ValueError(f'Input path declaration changed: {name}')
    deck_path = output / 'case.i'
    deck_path.write_text(deck)
    command = shlex.split(args.launcher) + ['-n', str(args.ranks), executable, '-i', str(deck_path),
                                          f'Outputs/file_base={output}/case',
                                          f'Outputs/snap_out/file_base={output}/case_snap']
    report = dict(status='prepared', case_id='v4m02656_r012', command=command,
                  ranks=args.ranks, executable_sha256=sha256(executable),
                  input_sha256=sha256(deck_path), permeability_sha256=sha256(HERE / 'case/permeability.i'))
    (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.prepare_only:
        print(shlex.join(command))
        return
    environment = dict(os.environ)
    for prefix in ('OMP', 'MKL', 'OPENBLAS', 'NUMEXPR'):
        environment[prefix + '_NUM_THREADS'] = '1'
    started = time.monotonic()
    with (output / 'solver.log').open('w') as log:
        subprocess.run(command, cwd=output, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    report.update(compare(output))
    report.update(status='complete', seconds=time.monotonic() - started)
    (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

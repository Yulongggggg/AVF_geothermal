# SPDX-License-Identifier: LGPL-2.1-only
"""Reconstruct the supplied 400 by 400 FE permeability from the original seed."""
import argparse
import json
from pathlib import Path

import numpy as np

from common import HERE, generator, sha256, verify_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factor', type=Path, required=True,
                        help='FE sampling factor fe_sampling_cholesky_fdc8c5e.npy')
    parser.add_argument('--out', type=Path, required=True, help='New permeability text file')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out already exists')
    verify_files()
    provenance = json.loads((HERE / 'provenance.json').read_text())
    factor = provenance['sampling_factor']
    if args.factor.stat().st_size != factor['bytes'] or sha256(args.factor) != factor['sha256']:
        raise ValueError('The FE sampling factor does not match the original realization generator.')
    model = generator()
    base = model.G
    stored = np.load(HERE / 'reference/scenario.npz')
    reference = np.load(HERE / 'reference/fields.npz')
    scenario = model.scenario_params_v4(2656)
    image = HERE / 'case/fractures.png'
    field, _, _, recipe = base.build_heatmap_struct_logk(image, scenario['rng'], base.ellipse_kernel())
    mean = base.coarsen(field / np.log(10.), 400)
    mean128 = base.upsample(mean, base.x400, base.y400, base.xs, base.ys)
    density = base.coarsen(base.frac_density_full(base.fracture_mask(image)), 400)
    configuration = base.sample_confidence_cfg(scenario['rng'])
    confidence128 = base.eval_confidence(
        configuration, base.XS, base.YS,
        base.upsample(density, base.x400, base.y400, base.xs, base.ys))
    confidence64 = base.eval_confidence(
        configuration, base.XP, base.YP,
        base.upsample(density, base.x400, base.y400, base.xp, base.yp))
    if not np.array_equal(mean128.astype('f4'), stored['M']):
        raise ValueError('Reconstructed characterization mean differs from the stored case.')
    if not np.array_equal(confidence128.astype('f4'), stored['C']):
        raise ValueError('Reconstructed confidence differs from the stored case.')
    if recipe != str(stored['recipe']) or base.cfg_desc(configuration) != str(stored['cfg_desc']):
        raise ValueError('The characterization recipe differs from the stored case.')
    rng = np.random.default_rng(770000 + 2656 * 1000 + 12)
    z = rng.standard_normal(4096)
    z = rng.standard_normal(4096)  # The stored realization follows one screening rejection.
    if not np.array_equal(z.astype('f4'), reference['z']):
        raise ValueError('The latent realization differs from the stored case.')
    deviation = (base.confidence_to_S(confidence64, scenario['s_max']).ravel()
                 * (np.load(args.factor) @ z)).reshape(64, 64)
    permeability = np.clip(10 ** (mean + base.upsample(
        deviation, base.xp, base.yp, base.x400, base.y400)), base.CLIP_MIN, base.CLIP_MAX)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base.write_pml(permeability, base.x400, base.y400, args.out)
    actual = sha256(args.out)
    if actual != provenance['reconstructed_permeability_sha256']:
        raise ValueError(f'Reconstructed permeability text differs: {actual}')
    print(json.dumps(dict(status='complete', case_id=provenance['case_id'],
                          output=str(args.out), sha256=actual), indent=2))


if __name__ == '__main__':
    main()

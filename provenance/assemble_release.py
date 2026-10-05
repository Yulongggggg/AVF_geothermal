"""Maintainer record: how the release assets, packaged resources and reference results were built.

Inputs are the archival sources described in ``docs/PROVENANCE.md``:

--handoff     OL-geothermal-final-scheme @ codex/workstationd-nuts-20261001,
              handoffs/workstationd-nuts-20261001 (with Git LFS content)
--extension   same branch, handoffs/workstationd-nuts-extension-20261005
--results     OL-geothermal-final-scheme @ claude/workstationd-nuts-results-20261005,
              handoffs/workstationd-nuts-20261001-results
--prior-fit   OL-geothermal-final-scheme @ main, geouno/artifacts/prior
--parent      OL-geothermal-final-scheme @ main, geouno/artifacts/model/model.pt
--evidence    extracted AVF-Nature-communication.../scripts/operator_final_20260930/evidence.tar.gz
--audit       AVF-Nature-communication.../scientific_audit_20261002

Every copied input is checked against the hashes recorded in the archival bundles. Tar
archives are written with fixed ordering, ownership and modification times so that their
SHA-256 is reproducible from the same inputs.
"""
import argparse
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

CHECKPOINT = 'ed04348969fe27903bc9bd084781164f65c610c7e46da0ff75bee60553fdc18e'
PARENT = 'cf278e646fbe53db11ec4a03f0545b6354ef81daa9d8c2282ebaf24704ccff9b'
PRIOR_FACTOR = '64eb46fb1200c68b2f715084ce4248db119bce68df4ed7591286bd3d6c28fb7c'
CALIBRATION = 'b22e64c3148695ea645f94a96a5ba0c5f4cc6bb8a3480d36f0243d4a396ab3c7'
MONITORING_PROTOCOL = '968aa31e4f8e6512e1b63c4d854719647dfab9aa8b24abec3296efecc654b27b'
INFERENCE_PROTOCOL = 'c09205d5fbfbcf8cbf69bc3851ec63d78c0f2f47ed2b7b5b82461267fb28977a'
DESIGN = '8d9d89d9a8568da6e615a1d6c561438666e5eec559388fe86b27ed2ec9d1442f'
REPOSITORY = 'Yulongggggg/AVF_geothermal'
TAG = 'artifacts-v1'
MTIME = 1791158400  # 2026-10-05T00:00:00Z, fixed archive timestamp


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked_copy(src, dest, digest=None):
    if digest is not None and sha256(src) != digest:
        raise RuntimeError(f'{src}: hash mismatch')
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return sha256(dest)


def deterministic_tar(dest, members):
    """``members``: list of (archive name, source path). Uncompressed, sorted, fixed metadata."""
    with tarfile.open(dest, 'w', format=tarfile.PAX_FORMAT) as tar:
        for name, src in sorted(members):
            data = Path(src).read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = MTIME
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            tar.addfile(info, io.BytesIO(data))
    return {name: sha256(src) for name, src in members}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('handoff', 'extension', 'results', 'prior-fit', 'parent', 'evidence', 'audit', 'out', 'repo'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    out, repo = a.out, a.repo
    out.mkdir(parents=True, exist_ok=True)
    bundle, ext = read(a.handoff / 'bundle.json'), read(a.extension / 'bundle.json')
    case_manifest = read(a.evidence / 'data/monitoring/case_manifest.json')
    assets = {}

    # 1. Checkpoint, prior factor, parent.
    for name, src, digest, group, rel in [
            ('geouno_pod1024_ed043489.pt', a.handoff / 'data/model.pt', CHECKPOINT, 'core', 'model'),
            ('prior_factor_64eb46fb.npy', a.handoff / 'data/prior_factor.npy', PRIOR_FACTOR, 'core', 'prior'),
            ('geouno_parent_cf278e64.pt', a.parent, PARENT, 'training', 'model')]:
        checked_copy(src, out / name, digest)
        assets[name] = dict(group=group, path=f'{rel}/{name}', sha256=digest, bytes=(out / name).stat().st_size)

    # 2. Reservoir cases: the exact multiwell_opt17 case files and base inputs of the paper.
    members, registry = [], dict(pilot=[], test=[])
    protocol = read(a.handoff / 'provenance/monitoring_protocol.json')
    entries = {e['label']: (a.handoff, e) for e in bundle['cases']}
    entries.update({e['label']: (a.extension, e) for e in ext['cases']})
    for role in ('pilot', 'test'):
        for e in protocol['cases'][role]:
            record = dict(e)
            if e['label'] in entries:
                root, be = entries[e['label']]
                assert be['scenario'] == e['scenario'] and be['case_id'] == e['case_id']
                case_sha = sha256(root / be['case_path'])
                assert case_sha == be['case_sha256'] == case_manifest['multiwell_opt17/' + e['label']]['sha256']
                assert sha256(root / be['base_path']) == be['base_sha256']
                members += [(f"cases/{e['label']}.npz", root / be['case_path']),
                            (f"base_inputs/{e['label']}.npz", root / be['base_path'])]
                record.update(released=True, multiwell_opt17_case_sha256=case_sha, base_sha256=be['base_sha256'])
            else:
                record.update(released=False, note='Validation design reservoir; inputs are in the FE data release.')
            registry[role].append(record)
    ref = a.handoff / bundle['reference']['path']
    assert sha256(ref) == bundle['reference']['sha256']
    members.append(('validation/pilot_01_value_gradient.npz', ref))
    files = deterministic_tar(out / 'reservoirs_v1.tar', members)
    assets['reservoirs_v1.tar'] = dict(group='core', extract='reservoirs', files=files,
                                       sha256=sha256(out / 'reservoirs_v1.tar'),
                                       bytes=(out / 'reservoirs_v1.tar').stat().st_size)

    # 3. NUTS results (six reservoirs x four chains), merged latent duplicate omitted.
    sums = {}
    for line in (a.results / 'SHA256SUMS').read_text().splitlines():
        digest, rel = line.split(None, 1)
        sums[rel.strip().lstrip('./')] = digest
    for case in [f'test_{i:02d}' for i in range(1, 7)]:
        members = []
        base = a.results / 'results' / case
        for src in sorted(base.rglob('*')):
            if src.is_dir() or src.name.endswith('.lock') or src.relative_to(base).as_posix() == 'latent_samples.npz':
                continue
            rel = src.relative_to(a.results).as_posix()
            assert sums[rel] == sha256(src), rel
            members.append((src.relative_to(a.results / 'results').as_posix(), src))
        name = f'nuts_{case}_v1.tar'
        files = deterministic_tar(out / name, members)
        assets[name] = dict(group='nuts', extract='nuts', files=files, sha256=sha256(out / name),
                            bytes=(out / name).stat().st_size)
    members = [('README_return.md', a.results / 'README.md'), ('SHA256SUMS_return', a.results / 'SHA256SUMS')]
    for n in ('report.json', 'preflight.json', 'DEVIATION_smoke_20261001.md', 'smoke_cpu.log', 'smoke_cuda.log',
              'merge.log', 'worker_0.log'):
        src = a.results / 'results' / n
        assert sums['results/' + n] == sha256(src), n
        members.append((n, src))
    files = deterministic_tar(out / 'nuts_summary_v1.tar', members)
    assets['nuts_summary_v1.tar'] = dict(group='nuts', extract='nuts', files=files,
                                         sha256=sha256(out / 'nuts_summary_v1.tar'),
                                         bytes=(out / 'nuts_summary_v1.tar').stat().st_size)

    # 4. Prior-fit inputs (1,024 training fields, original smooth factor, fit record).
    names = ('fit_rows.json', 'fit_fields.npz', 'original_factor_lower.npz', 'prior_fit.json', 'stats.json',
             'training_power.npy', 'prior_factor.npy')
    assert sha256(a.prior_fit / 'prior_factor.npy') == PRIOR_FACTOR
    files = deterministic_tar(out / 'prior_fit_inputs_v1.tar', [(n, a.prior_fit / n) for n in names])
    assets['prior_fit_inputs_v1.tar'] = dict(group='prior', extract='prior_fit', files=files,
                                             sha256=sha256(out / 'prior_fit_inputs_v1.tar'),
                                             bytes=(out / 'prior_fit_inputs_v1.tar').stat().st_size)

    # 5. Packaged resources.
    res = repo / 'geouno/resources'
    stats_sha = checked_copy(a.handoff / 'data/stats_test.json', res / 'stats.json')
    assert sha256(a.handoff / 'data/stats_val.json') == stats_sha
    checked_copy(a.handoff / 'data/calibration.json', res / 'calibration.json', CALIBRATION)
    checked_copy(a.evidence / 'data/monitoring/design.json', res / 'design.json', DESIGN)
    write(res / 'cases.json', registry)
    write(res / 'release_manifest.json', dict(
        repository=REPOSITORY, release_tag=TAG,
        release_url=f'https://github.com/{REPOSITORY}/releases/download/{TAG}',
        checkpoint_sha256=CHECKPOINT, assets=assets))

    # 6. Frozen protocols (archival copies).
    proto = repo / 'protocols'
    checked_copy(a.handoff / 'provenance/monitoring_protocol.json', proto / 'monitoring_protocol.json',
                 MONITORING_PROTOCOL)
    checked_copy(a.handoff / 'provenance/inference_protocol.json', proto / 'inference_protocol.json',
                 INFERENCE_PROTOCOL)
    for n in ('hamiltonian_protocol.json', 'model_selection.json'):
        checked_copy(a.handoff / 'provenance' / n, proto / n)
    checked_copy(a.evidence / 'data/monitoring/design.json', proto / 'monitoring_design.json', DESIGN)
    checked_copy(a.handoff / 'data/calibration.json', proto / 'calibration.json', CALIBRATION)
    write(proto / 'case_manifest.json', {k: v['sha256'] for k, v in case_manifest.items()})

    # 7. Reference per-task scores of the paper runs (corrected static-only overlay applied).
    formal = a.evidence / 'data/monitoring/formal'
    scores = {}
    for path in sorted(formal.rglob('scores.json')):
        rel = path.parent.relative_to(formal).as_posix()
        parts = rel.split('/')
        if parts[0] == 'references':
            key = f'{parts[2]}/{parts[1]}'          # prior|static_prior / test_xx
        else:
            config, case, method = parts
            if config == 'multiwell_opt9_k':
                key = f'{config}/{case}/static_historical_member_order'
            else:
                key = f'{config}/{case}/{method}'
        scores[key] = read(path)
    for path in sorted((a.audit / 'source_data/corrected_static').glob('test_*/scores.json')):
        scores[f'multiwell_opt9_k/{path.parent.name}/static'] = read(path)
    nuts_root = a.results / 'results'
    for case in [f'test_{i:02d}' for i in range(1, 7)]:
        scores[f'multiwell_opt17/{case}/nuts'] = read(nuts_root / case / 'scores.json')
    write(repo / 'reference/paper_task_scores.json', scores)
    print(json.dumps({k: (v['bytes'], v['sha256'][:12]) for k, v in assets.items()}, indent=1))
    print('reference tasks:', len(scores))


if __name__ == '__main__':
    main()

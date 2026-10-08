"""Check the installed wheel from outside the source checkout."""
import json
import os
import resource
import subprocess
import sys
from pathlib import Path


def main():
    import geouno
    from geouno import artifacts, paper
    from geouno.model.geology import GeologyJointOperator
    from geouno.model.geothermal import GeothermalJointOperator
    from geouno.tasks import reference_scores

    root = Path(os.environ['GEOUNO_REVIEW_ROOT']).resolve()
    installed = Path(os.environ['GEOUNO_REVIEW_INSTALLED']).resolve()
    package = Path(geouno.__file__).resolve().parent
    assert package.is_relative_to(installed), package
    assert not Path.cwd().is_relative_to(root), Path.cwd()
    source_files = {p.relative_to(root / 'geouno') for p in (root / 'geouno').rglob('*.py')}
    installed_files = {p.relative_to(package) for p in package.rglob('*.py')}
    assert source_files == installed_files
    for filename in source_files:
        assert artifacts.sha256(package / filename) == artifacts.sha256(root / 'geouno' / filename), filename
    checked = paper.check()
    assert len(checked) == 80 and all(row[-1] for row in checked)
    hashes = {}
    for row in (paper.SOURCE / 'SHA256SUMS').read_text().splitlines():
        expected, filename = row.split(maxsplit=1)
        hashes[filename] = artifacts.sha256(paper.SOURCE / filename) == expected
    assert len(hashes) == 23 and all(hashes.values())
    assert len(reference_scores()) == 1206
    assert GeologyJointOperator is GeothermalJointOperator
    core = artifacts.verify(root=root / 'artifacts', groups=('core',))
    assert len(core) == 53 and all(core.values())
    command = [str(installed / 'bin/geouno'), 'paper-numbers']
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout.splitlines()[-1]) == dict(checked=80, mismatches=0)
    wheel = Path(os.environ['GEOUNO_REVIEW_WHEEL'])
    report = dict(status='passed', job_id=os.environ.get('SLURM_JOB_ID'), cwd=str(Path.cwd()),
                  installed_package=str(package), matched_python_files=len(source_files),
                  checked_paper_cells=len(checked), verified_data_files=len(hashes),
                  verified_core_artifact_files=len(core), reference_tasks=len(reference_scores()),
                  geothermal_compatibility=True, console_entry_point=True,
                  wheel_sha256=artifacts.sha256(wheel), max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    destination = root / 'provenance/repository_review_20261008/installed_wheel.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())

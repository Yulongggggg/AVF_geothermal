"""Run each test file in a separate process and retain the complete test results."""
import ast
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path


def main():
    destination = Path(__file__).resolve().parent
    root = destination.parents[1]
    files = sorted(root.glob('tests/test_*.py'))
    checks = ['tests/test_workflows.py']
    checks += [str(p.relative_to(root)) for p in files if p.name not in ('test_release.py', 'test_workflows.py')]
    release = root / 'tests/test_release.py'
    checks += [f'tests/test_release.py::{node.name}' for node in ast.parse(release.read_text()).body
               if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')]
    report = dict(job_id=os.environ.get('SLURM_JOB_ID'), host=platform.node(), python=sys.version,
                  planned_runs=len(checks), passed=0, skipped=0, failed=0, runs=[])
    for check in checks:
        command = [sys.executable, '-m', 'pytest', '-q', check]
        started = time.monotonic()
        result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report['runs'].append(dict(command=command, exit_code=result.returncode,
                                   seconds=time.monotonic() - started, output=result.stdout))
        for key in ('passed', 'skipped', 'failed'):
            report[key] += sum(map(int, re.findall(r'(\d+) ' + key, result.stdout)))
        report['max_child_rss_kib'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        report['status'] = ('running' if len(report['runs']) < len(checks) else
                            'passed' if all(r['exit_code'] == 0 for r in report['runs']) else 'failed')
        (destination / 'serial_tests.json').write_text(json.dumps(report, indent=2) + '\n')
        print(check, result.returncode, result.stdout, flush=True)
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())

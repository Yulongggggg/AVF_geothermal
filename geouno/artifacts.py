"""Release artifacts: locations, download and SHA-256 verification.

The model checkpoint, inverse-prior factor, reservoir cases and NUTS results are
published as assets of the GitHub release recorded in ``resources/release_manifest.json``.
They are stored under ``$GEOUNO_DATA`` (default: ``artifacts/`` in a source checkout, else
``~/.cache/geouno``). Downloads and extracted files are verified against SHA-256;
``path`` also verifies individual manifest-listed files before returning them.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
RESOURCES = PACKAGE / 'resources'
REPOSITORY = PACKAGE.parent


def read_json(path):
    return json.loads(Path(path).read_text())


def resource(name):
    """Path of a packaged resource (``stats.json``, ``calibration.json``, ``design.json`` ...)."""
    return RESOURCES / name


def manifest():
    return read_json(resource('release_manifest.json'))


def data_dir():
    env = os.environ.get('GEOUNO_DATA')
    if env:
        return Path(env).expanduser().resolve()
    if (REPOSITORY / 'pyproject.toml').exists():
        return REPOSITORY / 'artifacts'
    return Path.home() / '.cache' / 'geouno'


def sha256(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def asset_names(group='core'):
    """Asset names in ``group`` (``core``, ``nuts``, ``prior``, ``training`` or ``all``)."""
    assets = manifest()['assets']
    return [name for name, record in assets.items() if group == 'all' or record['group'] == group]


def _download(url, dest, repo=None, tag=None):
    """Download ``url`` to ``dest``; fall back to ``gh release download`` (private repository).

    The destination file name always equals the release asset name.
    """
    tmp = dest.with_name(dest.name + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'geouno-downloader'})
    try:
        with urllib.request.urlopen(request) as response, tmp.open('wb') as out:
            shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
        os.replace(tmp, dest)
    except urllib.error.HTTPError as error:
        tmp.unlink(missing_ok=True)
        if error.code not in (401, 403, 404) or not shutil.which('gh') or not repo:
            raise
        # Anonymous asset URLs only work for public repositories; use the GitHub CLI's login.
        subprocess.run(['gh', 'release', 'download', tag, '-R', repo, '-p', dest.name,
                        '-D', str(dest.parent), '--clobber'], check=True)


def _extract(archive_path, target):
    with tarfile.open(archive_path) as archive:
        if hasattr(tarfile, 'data_filter'):
            archive.extractall(target, filter='data')
        else:  # Python without extraction filters: reject unsafe members explicitly.
            for member in archive.getmembers():
                name = Path(member.name)
                if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
                    raise RuntimeError(f'Unsafe archive member: {member.name}')
            archive.extractall(target)


def fetch(names, root=None, force=False, log=print, source_dir=None, keep_archives=False):
    """Download, verify and (for archives) extract assets; return the data directory.

    ``source_dir`` installs from a local directory holding the asset files (e.g. a mirror
    or the Zenodo record) instead of downloading them. Archives are deleted after their
    extracted files pass verification unless ``keep_archives`` is set.
    """
    root = Path(root) if root else data_dir()
    root.mkdir(parents=True, exist_ok=True)
    info = manifest()
    for name in names:
        record = info['assets'][name]
        archive = bool(record.get('extract'))
        if archive:
            marker = root / record['extract'] / f'.{name}.complete'
            valid_files = (not force and marker.exists()
                           and marker.read_text().strip() == record['sha256']
                           and all((root / record['extract'] / rel).is_file()
                                   and sha256(root / record['extract'] / rel) == digest
                                   for rel, digest in record['files'].items()))
            if valid_files:
                log(f'{name}: already installed')
                continue
            marker.unlink(missing_ok=True)
        dest = root / 'downloads' / name if archive else root / record['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if force or not dest.exists() or sha256(dest) != record['sha256']:
            if source_dir is not None:
                log(f'copying {name} from {source_dir}')
                shutil.copyfile(Path(source_dir) / name, dest)
            else:
                log(f'downloading {name} ({record["bytes"] / 2**20:.1f} MiB)')
                _download(f"{info['release_url']}/{name}", dest, info.get('repository'), info.get('release_tag'))
        actual = sha256(dest)
        if actual != record['sha256']:
            raise RuntimeError(f'{name}: SHA-256 {actual} != {record["sha256"]}')
        if archive:
            target = root / record['extract']
            target.mkdir(parents=True, exist_ok=True)
            _extract(dest, target)
            for rel, digest in record['files'].items():
                if sha256(target / rel) != digest:
                    raise RuntimeError(f'{name}: extracted {rel} failed SHA-256 verification')
            marker.write_text(record['sha256'] + '\n')
            if not keep_archives:
                dest.unlink()
        log(f'verified {name}')
    return root


def verify(root=None, groups=('core',)):
    """Check every extracted file listed in the manifest; return ``{path: ok}``."""
    root = Path(root) if root else data_dir()
    info = manifest()
    results = {}
    for record in info['assets'].values():
        if record['group'] not in groups and 'all' not in groups:
            continue
        files = record.get('files') or {record['path']: record['sha256']}
        base = root / record['extract'] if record.get('extract') else root
        for rel, digest in files.items():
            path = base / rel
            results[str(path)] = path.exists() and sha256(path) == digest
    return results


def path(rel, root=None):
    """Path of an extracted artifact file, e.g. ``path('model/geouno_pod1024.pt')``."""
    p = (Path(root) if root else data_dir()) / rel
    if not p.exists():
        raise FileNotFoundError(f'{p} is missing; run `geouno download` first')
    for record in manifest()['assets'].values():
        files = ({str(Path(record['extract']) / name): digest for name, digest in record['files'].items()}
                 if record.get('extract') else {record['path']: record['sha256']})
        expected = files.get(str(Path(rel)))
        if expected is not None and sha256(p) != expected:
            raise RuntimeError(f'{p}: SHA-256 mismatch; run `geouno download` to repair the artifact')
    return p


class Reservoirs:
    """Access to the 24 test reservoirs and the validation pilot shipped with the release."""

    def __init__(self, root=None):
        self.root = Path(root) if root else data_dir()
        self.registry = read_json(resource('cases.json'))

    def labels(self, role='test'):
        return [e['label'] for e in self.registry[role]]

    def entry(self, label):
        for role in ('test', 'pilot'):
            for e in self.registry[role]:
                if e['label'] == label:
                    return e
        raise KeyError(label)

    def case_path(self, label):
        return path(f'reservoirs/cases/{label}.npz', self.root)

    def base_path(self, label):
        return path(f'reservoirs/base_inputs/{label}.npz', self.root)

    def case(self, label):
        import numpy as np
        with np.load(self.case_path(label), allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    def base(self, label):
        import numpy as np
        with np.load(self.base_path(label), allow_pickle=False) as z:
            return z['base'].copy()

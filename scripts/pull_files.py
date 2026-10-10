#!/usr/bin/env python3
"""Pull arbitrary files from ghcr.io/peonist-ai/halogen-flash-server tags.
Usage: pull_files.py <tag> <tag> ... --file NAME [--file NAME]...
Extracts every layer member whose path ends with /NAME (or == NAME) to
work/versions/<tag>/NAME (basename). Logs to work/pull.status.
"""
import json, os, subprocess, sys, urllib.request, datetime, hashlib, tarfile, io

ROOT = '/home/rangizingo/re-halogen'
S = os.path.join(ROOT, 'work/pull.status')
REPO = 'peonist-ai/halogen-flash-server'
ACC = 'application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json'

def log(msg):
    with open(S, 'a') as f:
        f.write(msg + '\n')
    print(msg, flush=True)

def get(url, token=None, binary=False):
    req = urllib.request.Request(url)
    req.add_header('Accept', ACC)
    if token:
        req.add_header('Authorization', 'Bearer ' + token)
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    return data if binary else data.decode('utf8', 'replace')

def main():
    args = sys.argv[1:]
    tags = []
    names = []
    i = 0
    while i < len(args):
        if args[i] == '--file':
            names.append(args[i+1]); i += 2
        else:
            tags.append(args[i]); i += 1
    log('files-pull start ' + datetime.datetime.utcnow().strftime('%FT%TZ') + ' ' + ','.join(tags) + ' files=' + ','.join(names))
    token = json.loads(get(f'https://ghcr.io/token?scope=repository:{REPO}:pull'))['token']
    for tag in tags:
        out = os.path.join(ROOT, 'work/versions', tag)
        os.makedirs(out, exist_ok=True)
        man = json.loads(get(f'https://ghcr.io/v2/{REPO}/manifests/{tag}', token))
        if 'layers' not in man:
            log(f'{tag} not a direct image manifest')
            continue
        found = {n: False for n in names}
        for l in man['layers']:
            if all(found.values()):
                break
            data = get(f'https://ghcr.io/v2/{REPO}/blobs/{l["digest"]}', token, binary=True)
            try:
                tf = tarfile.open(fileobj=io.BytesIO(data), mode='r:*')
            except Exception:
                continue
            for m in tf.getmembers():
                base = os.path.basename(m.name)
                if base in names and m.isfile():
                    f = tf.extractfile(m)
                    open(os.path.join(out, base), 'wb').write(f.read())
                    found[base] = True
                    log(f'{tag} {m.name} -> {base} ({m.size} B)')
            tf.close()
        for n, ok in found.items():
            if not ok:
                log(f'{tag} {n} NOT FOUND')
    log('files-pull done ' + datetime.datetime.utcnow().strftime('%FT%TZ'))

main()

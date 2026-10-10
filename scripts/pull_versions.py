#!/usr/bin/env python3
"""Pull flash_serve ELF from ghcr.io/peonist-ai/halogen-flash-server tags
into work/versions/<tag>/flash_serve. Anonymous ghcr pull token.
Logs to work/pull.status (appends)."""
import json, os, subprocess, sys, urllib.request, datetime, hashlib

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
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    return data if binary else data.decode('utf8', 'replace')

def main():
    tags = sys.argv[1:] or ['0.17.2', '0.17.0', '0.16.4', '0.15.0']
    log('py-pull start ' + datetime.datetime.utcnow().strftime('%FT%TZ'))
    tok = json.loads(get(f'https://ghcr.io/token?scope=repository:{REPO}:pull'))
    token = tok.get('token') or tok.get('access_token')
    log(f'token_len={len(token)}')
    for tag in tags:
        out = os.path.join(ROOT, 'work/versions', tag)
        if os.path.exists(os.path.join(out, 'flash_serve')) and os.path.getsize(os.path.join(out, 'flash_serve')) > 1000000:
            log(f'{tag} cached')
            continue
        os.makedirs(out, exist_ok=True)
        try:
            man = json.loads(get(f'https://ghcr.io/v2/{REPO}/manifests/{tag}', token))
        except Exception as e:
            log(f'{tag} manifest fetch failed: {e}')
            continue
        if 'layers' in man:
            digest = f'sha256:{hashlib.sha256(json.dumps(man,separators=(",",":")).encode()).hexdigest()}'
            log(f'{tag} direct image manifest')
        else:
            digest = None
            for m in man.get('manifests', []):
                p = m.get('platform', {})
                if p.get('os') == 'linux' and p.get('architecture') == 'amd64':
                    digest = m['digest']
            if not digest:
                log(f'{tag} NO-AMD64-MANIFEST')
                continue
            man = json.loads(get(f'https://ghcr.io/v2/{REPO}/manifests/{digest}', token))
        log(f'{tag} amd64={digest}')
        with open(os.path.join(out, 'manifest.json'), 'w') as f:
            json.dump(man, f)
        found = 0
        for l in man.get('layers', []):
            ld = l['digest']
            if l.get('size', 0) < 1000000:
                continue
            lp = os.path.join(out, 'layer.tar.zst')
            data = get(f'https://ghcr.io/v2/{REPO}/blobs/{ld}', token, binary=True)
            open(lp, 'wb').write(data)
            tf = subprocess.run(['tar', '--zstd', '-tf', lp], capture_output=True, text=True)
            if tf.returncode != 0:
                os.remove(lp)
                continue
            hits = [x for x in tf.stdout.splitlines() if x.endswith('/flash_serve') or x == 'flash_serve']
            if not hits:
                os.remove(lp)
                continue
            ex = subprocess.run(['tar', '--zstd', '-xf', lp, '--to-stdout', hits[0]],
                                capture_output=True)
            open(os.path.join(out, 'flash_serve'), 'wb').write(ex.stdout)
            log(f'{tag} found layer {ld[:19]}… size={len(data)} extracted={len(ex.stdout)}')
            os.remove(lp)
            found = 1
            break
        if not found:
            log(f'{tag} flash_serve NOT FOUND in layers')
    log('py-pull done ' + datetime.datetime.utcnow().strftime('%FT%TZ'))

main()

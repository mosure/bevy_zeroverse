#!/usr/bin/env python3
"""Check cached-model batch invariance and an upstream PyTorch image reference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from indoor_embedding_report import load_embeddings


def validate(batched, single, reference, output):
    bm, b = load_embeddings(batched)
    sm, s = load_embeddings(single)
    if bm['index_sha256'] != sm['index_sha256'] or bm['samples'] != sm['samples']:
        raise ValueError('batch comparison must use identical input rows')
    if bm['loaded_weights']['loaded_weight_sha256'] != sm['loaded_weights']['loaded_weight_sha256']:
        raise ValueError('batch comparison model weights differ')
    if bm['batch_size'] <= 1 or sm['batch_size'] != 1:
        raise ValueError('requires a multi-image batch and a singleton run')
    ref = json.loads(reference.read_text())
    tensor = reference.parent/ref['tensors']['file']
    if hashlib.sha256(tensor.read_bytes()).hexdigest() != ref['tensors']['file_sha256']:
        raise ValueError('upstream reference tensor checksum mismatch')
    expected = load_file(tensor)['output.image_embedding.normalized']
    rows = [i for i, r in enumerate(bm['samples'])
            if r['sha256'] == ref['inputs']['encoded_image']['file_sha256']]
    if len(rows) < 2 or expected.shape != (1, b.shape[1]):
        raise ValueError('include the upstream reference PNG at least twice')
    measured = dict(
        batch_invariance_max_absolute_error=float(np.max(np.abs(b-s))),
        reference_normalized_embedding_max_absolute_error=float(np.max(np.abs(b[rows]-expected))),
        duplicate_max_cosine_distance=float(max(0, 1-np.dot(b[rows[0]], b[rows[1]]))),
    )
    thresholds = dict(batch_invariance_max_absolute_error=5e-4,
                      reference_normalized_embedding_max_absolute_error=5e-4,
                      duplicate_max_cosine_distance=1e-6)
    passed = all(np.isfinite(v) and v <= thresholds[k] for k, v in measured.items())
    result = dict(passed=passed, measurements=measured, upper_limits=thresholds,
                  model_sha256=bm['loaded_weights']['loaded_weight_sha256'],
                  batched_metadata_sha256=hashlib.sha256(batched.read_bytes()).hexdigest(),
                  single_metadata_sha256=hashlib.sha256(single.read_bytes()).hexdigest(),
                  reference_manifest_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  reference=ref, batched_run=dict(path=str(batched), batch_size=bm['batch_size']),
                  singleton_run=str(single),
                  limits='One upstream deterministic image reference and eight-input batch controls; not a full model or downstream benchmark.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    if not passed:
        raise ValueError(f'embedding controls failed: {measured}')
    return measured


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['batched', 'single', 'reference', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.batched, args.single, args.reference, args.output), indent=2))

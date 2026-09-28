#!/usr/bin/env python3
"""Qualify glass quadrature against dense, matched linear-radiance references.

This isolates screen-space sampling error, not physical transport error. Each
filter is compared to its own dense quadrature; their reference difference is
reported separately so changing blur or clipping brightness cannot be called
noise reduction. Use compare_indoor_cycles.py for the independent optical test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from compare_indoor_cycles import LUMA, optical_identity, preview, raw

MODES = ('legacy', 'legacy-reference', 'default', 'reference')


def erode(mask):
    out = np.zeros_like(mask)
    out[1:-1, 1:-1] = np.logical_and.reduce([
        mask[y:y+mask.shape[0]-2, x:x+mask.shape[1]-2]
        for y in range(3) for x in range(3)])
    return out


def errors(image, reference, mask):
    difference = (image-reference) @ LUMA
    scale = max(float((reference @ LUMA)[mask].mean()), 1e-8)
    d = difference[mask]
    return {'pixels': int(mask.sum()), 'rmse': float(np.sqrt(np.mean(d*d))),
            'relative_rmse': float(np.sqrt(np.mean(d*d))/scale),
            'relative_bias': float(d.mean()/scale),
            'reference_mean_luminance': scale}, difference*difference


def compare(root, output):
    output.mkdir(parents=True, exist_ok=True)
    views, tiles, block_pairs = [], [], []
    sums = np.zeros(4)
    for directory in sorted((root/'glass_default').glob('seed_*')):
        name = directory.name
        dirs = [root/f'glass_{mode}'/name for mode in MODES]
        scenes = [json.loads((d/'reference/scene.json').read_text()) for d in dirs]
        identities = [optical_identity(s,d/'reference') for s,d in zip(scenes,dirs)]
        if len(set(identities)) != 1:
            raise ValueError(f'Optical inputs differ for {name}')
        captures = [json.loads((d/'capture.json').read_text()) for d in dirs]
        for mode, capture in zip(MODES,captures):
            if capture['capabilities']['glass_filter'] != mode.replace('-','_'):
                raise ValueError(f'Incorrect filter provenance for {name}/{mode}')
            if 'exposed scene-linear' not in capture['color_encoding']:
                raise ValueError('Noise qualification requires raw linear captures')
        size = scenes[0]['image_size']
        for camera in scenes[0]['cameras']:
            index = camera['index']
            prefix = f'view_{index:02}'
            semantic = [np.array(Image.open(d/f'{prefix}_semantic.png').convert('RGB')) for d in dirs]
            if not all(np.array_equal(semantic[0],s) for s in semantic[1:]):
                raise ValueError('Geometry/annotation mismatch')
            mask = erode(np.all(semantic[0] == (197,176,213),axis=2))
            if mask.sum() < 64:
                views.append({'seed': scenes[0]['seed'], 'view': index, 'pixels':int(mask.sum()), 'qualified':False})
                continue
            paths = [d/f'{prefix}_color.rgba32f' for d in dirs]
            images = [raw(p,size).astype(np.float64) for p in paths]
            if not all(np.isfinite(i).all() for i in images):
                raise ValueError('Non-finite capture')
            old, old_ref, new, new_ref = images
            old_metrics, old_error = errors(old,old_ref,mask)
            new_metrics, new_error = errors(new,new_ref,mask)
            common_metrics, common_error = errors(new,old_ref,mask)
            reference_change, _ = errors(new_ref,old_ref,mask)
            sums += (old_error[mask].sum(),new_error[mask].sum(),common_error[mask].sum(),mask.sum())
            for y in range(0,size[1],32):
                for x in range(0,size[0],32):
                    m = mask[y:y+32,x:x+32]
                    if m.sum() >= 16:
                        block_pairs.append([old_error[y:y+32,x:x+32][m].sum(),new_error[y:y+32,x:x+32][m].sum(),common_error[y:y+32,x:x+32][m].sum()])
            views.append({'seed': scenes[0]['seed'], 'view':index, 'qualified':True,
                          'optical_content_sha256':identities[0],
                          'capture_sha256':dict(zip(MODES,[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])),
                          'before':old_metrics,'after':new_metrics,'after_against_shared_legacy_reference':common_metrics,
                          'dense_reference_change':reference_change})
            tile=Image.new('RGB',(size[0]*3,size[1]+22))
            for col,(im,label) in enumerate([(old,'legacy'),(new,'new'),(new_ref,'dense reference')]):
                tile.paste(Image.fromarray(preview(im)),(col*size[0],22))
                ImageDraw.Draw(tile).text((col*size[0]+5,5),f'{name}/{index} {label}',fill='white')
            tiles.append(tile)
    if not sums[3] or not block_pairs:
        raise ValueError('No visible glass coverage')
    pairs = np.array(block_pairs)
    rng = np.random.default_rng(44014)
    ratios=[]
    common_ratios=[]
    for _ in range(1000):
        b=pairs[rng.integers(len(pairs),size=len(pairs))].sum(axis=0)
        ratios.append(float(np.sqrt(b[1]/max(b[0],1e-30))))
        common_ratios.append(float(np.sqrt(b[2]/max(b[0],1e-30))))
    before, after, common = np.sqrt(sums[:3]/sums[3])
    result={'scope':'Matched native screen-space quadrature error; not photographic realism, temporal stability, or physical transport error',
            'dense_samples_per_pixel':2048,'mask':'window semantic, eroded by one pixel, >=64 pixels/view',
            'filter_reference_warning':'Each filter uses its own dense reference; dense_reference_change records kernel bias separately',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'pixels':int(sums[3]),'rmse_before':float(before),'rmse_after':float(after),
            'rmse_after_against_shared_legacy_reference':float(common),
            'shared_reference_rmse_reduction_fraction':float(1-common/max(before,1e-30)),
            'shared_reference_after_before_ratio_95_percent_spatial_block_bootstrap':np.quantile(common_ratios,[.025,.975]).tolist(),
            'rmse_reduction_fraction':float(1-after/max(before,1e-30)),
            'after_before_rmse_ratio_95_percent_spatial_block_bootstrap':np.quantile(ratios,[.025,.975]).tolist(),
            'uncertainty_scope':'32x32 spatial block bootstrap within these consecutive scenes; not confidence over the entire procedural distribution',
            'views':views}
    (output/'glass_sampling.json').write_text(json.dumps(result,indent=2)+'\n')
    contact=Image.new('RGB',(tiles[0].width,sum(t.height for t in tiles)))
    y=0
    for tile in tiles: contact.paste(tile,(0,y));y+=tile.height
    contact.save(output/'glass_sampling.png')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.root,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='views'},indent=2))

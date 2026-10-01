#!/usr/bin/env python3
"""Check current project artifacts; optionally exercise the actual browser UI.

Static checks and browser checks have separate status. No GPU-render or viewer
qualification is inferred from an HTTP response, metadata or an image preview.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urljoin, urlparse
import zipfile

import numpy as np
from PIL import Image

from report_indoor_architecture import feature_flags, same_f32
from indoor_visibility import decode as decode_visibility, palette as visibility_palette

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "www/project"
MEDIA = PAGE / "static/media/architecture"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Document(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids, self.links, self.images = [], [], []
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a: self.ids.append(a["id"])
        if "href" in a: self.links.append(a["href"])
        if "src" in a: self.links.append(a["src"])
        if tag == "img": self.images.append(a)


def static_checks(captures=None):
    data = read(MEDIA/"gallery.json")
    current = int(re.search(r"GENERATOR_VERSION: u32 = (\d+)", (ROOT/"src/scene/procedural_indoor/layout.rs").read_text()).group(1))
    assert data['generator_version'] == current
    assert data['audit_rooms']==512 and data['rendered_rooms']==32 and data['rendered_views']==256
    assert [s['seed'] for s in data['scenes']] == list(range(32))
    assert data['modes']==['color','depth','normal','semantic','position','co_visibility']
    identities, images, lossless, metadata_count, masks_checked, peer_images = set(),0,0,0,0,0
    for scene in data['scenes']:
        metadata = read(MEDIA/scene['metadata'])
        manifest, capture, row = (metadata[k] for k in ('manifest','capture','architecture'))
        assert manifest['generator_version']==current
        assert manifest['seed']==capture['seed']==row['seed']==scene['seed']
        assert same_f32(manifest['envelope'],row['envelope'])
        assert scene['features']==[k for k,v in feature_flags(row).items() if v]
        assert [f['time'] for f in scene['frames']]==[0,1]
        assert capture['co_visibility_metadata']['legend']==data['co_visibility']['legend']
        for step,frame in enumerate(scene['frames']):
            assert (MEDIA/frame['plan']).is_file()
            assert len(frame['views'])==4
            for c,view in enumerate(frame['views']):
                camera = view['camera']
                assert camera['camera_index']==c and camera['step_index']==step and camera['time']==frame['time']
                assert camera in capture['views']
                pose = np.array(camera['world_from_view']).T
                assert np.isfinite(pose).all() and np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-5)
                assert abs(camera['fy_pixels'] - data['capture_size'][1]/(2*math.tan(camera['fov_y']/2))) < 1e-3
                assert camera['annotation_alignment']['reprojection_p99_pixels'] < .01
                for mode in data['modes']:
                    identity = (scene['seed'],step,c,mode)
                    assert identity not in identities
                    identities.add(identity)
                    path = MEDIA/view['images'][mode]
                    assert sha(path)==data['output_sha256'][path.name], path
                    with Image.open(path) as im:
                        im.load()
                        assert list(im.size)==data['capture_size']
                        if captures and mode!='color':
                            index = capture['views'].index(camera)
                            with Image.open(captures/f"seed_{scene['seed']:06}/view_{index:02}_{mode}.png") as source:
                                assert np.array_equal(np.array(im.convert('RGB')),np.array(source.convert('RGB')))
                            lossless += 1
                    images += 1
                index = capture['views'].index(camera)
                with zipfile.ZipFile(MEDIA/scene['masks']) as archive:
                    masks,valid,stats = decode_visibility(
                        archive.read(f'view_{index:02}_co_visibility_mask.png'),
                        archive.read(f'view_{index:02}_co_visibility_valid.png'),capture,index,4)
                    assert json.loads(archive.read('capture.json'))==metadata
                assert stats==view['visibility']
                with Image.open(MEDIA/view['images']['co_visibility']) as im:
                    assert np.array_equal(np.asarray(im.convert('RGB')),visibility_palette(capture['co_visibility_metadata'],4)[masks])
                masks_checked += 1
                for path in view['images']['peers']:
                    assert sha(MEDIA/path)==data['output_sha256'][path]
                    with Image.open(MEDIA/path) as im:
                        im.load(); assert list(im.size)==data['capture_size']
                    peer_images += 1
        metadata_count += 1
    source_count = 0
    if captures:
        for name,expected in data['source_sha256'].items():
            assert sha(captures/name)==expected, name
            source_count += 1
    documents = {p:Document(p) for p in (PAGE/'index.html',PAGE/'reference.html')}
    checked = 0
    for path,doc in documents.items():
        assert len(doc.ids)==len(set(doc.ids)), f'duplicate IDs in {path}'
        for im in doc.images: assert im.get('alt') and im.get('width') and im.get('height'), im
        for link in set(doc.links):
            url = urlparse(link)
            if url.scheme or url.netloc: continue
            target = (path.parent/unquote(url.path)).resolve() if url.path else path
            if target.is_dir(): target /= 'index.html'
            assert target.is_file(), f'{path}: missing {link}'
            if url.fragment and target.suffix=='.html':
                other = documents.get(target) or Document(target)
                assert unquote(url.fragment) in other.ids, f'{target}: missing #{url.fragment}'
            checked += 1
    main = (PAGE/'index.html').read_text()
    architecture = main.split('<!-- ARCHITECTURE_GALLERY_START -->')[1].split('<!-- ARCHITECTURE_GALLERY_END -->')[0]
    assert 's24005-' not in architecture and 'cohort-24000' not in architecture
    assert 'explore' in documents[PAGE/'index.html'].ids
    assert 'id="comparison"' not in main and 'id="scene-select"' not in main
    assert 'static/js/index.js' not in main
    assert 'data-architecture-mode="co_visibility"' in architecture
    assert 'id="architecture-visibility"' in architecture
    readme = (ROOT/'README.md').read_text()
    for example in ('bevy_zeroverse_dataloader_grid.webp','bevy_zeroverse_material_grid.webp'):
        assert f'(docs/{example})' in readme
        with Image.open(ROOT/'docs'/example) as im: im.load()
    assert 'static/media/architecture/hero.webp' in main
    assert 'ARCHITECTURE_GALLERY_START' in main and '@FEATURED@' not in main
    for seed in (7,8,6,2): assert f'data-architecture-seed="{seed}"' in main
    papers = PAGE/'static/papers'
    provenance = read(papers/'provenance.json')
    assert sha(papers/'bevy_zeroverse.pdf')==provenance['pdf_sha256']
    assert sha(papers/'whitepaper-source.zip')==provenance['source_archive_sha256']
    assert sha(MEDIA/'gallery.json')==provenance['architecture_gallery_sha256']
    with zipfile.ZipFile(papers/'whitepaper-source.zip') as archive:
        for name,expected in provenance['sources'].items():
            assert sha(ROOT/name)==expected
            assert hashlib.sha256(archive.read(str(Path(name).relative_to('tex')))).hexdigest()==expected
        for name in ('architecture_levels.jpg','architecture_envelopes.jpg','architecture_footprints.png',
                     'architecture_annotations.jpg','architecture_population.png','architecture_co_visibility.jpg','co_visibility_multiview_v21.png'):
            assert f'generated/{name}' in archive.namelist(), name
    population = read(MEDIA/'population.json')
    for key,total in data['population_plot_totals'].items():
        if key.endswith('/placement'):
            actual = sum(population['placement_heatmaps'][key.removesuffix('/placement')])
        elif key.startswith('main/'):
            actual = sum(population['object_counts_per_scene'][key].values())
        else:
            actual = sum(population['numeric'][key]['bin_counts'])
        assert actual==total, key
    return dict(passed=True, generator_version=current, audit_rooms=512, rendered_rooms=32,
                views=256, decoded_channel_images=images, unique_channel_identities=len(identities),
                lossless_annotation_roundtrips=lossless, source_hashes_checked=source_count,
                metadata_records=metadata_count, local_links=checked, pdf_and_source_archive_verified=True,
                population_denominators_verified=True, object_and_material_examples_preserved=True,
                co_visibility_masks_checked=masks_checked,co_visibility_peer_images=peer_images,
                unified_annotation_gallery=True, browser_checked=False, viewer_runtime_checked=False)


def browser_checks(url, output):
    from playwright.sync_api import sync_playwright
    output.mkdir(parents=True,exist_ok=True)
    errors, cases, peer_cases = [], 0, 0
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
        context = browser.new_context(viewport={'width':1440,'height':1000},reduced_motion='reduce')
        page = context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('response',lambda r:errors.append(f'HTTP {r.status}: {r.url}') if r.status>=400 else None)
        page.goto(url,wait_until='networkidle')
        page.wait_for_selector('#architecture-explorer[data-ready="true"]')
        data = page.request.get(urljoin(url,'static/media/architecture/gallery.json')).json()

        def selected(seed,step,mode,peer='all'):
            selection = f'{seed}:{step}:{mode}'
            page.wait_for_function('s => document.getElementById("architecture-explorer").dataset.selection === s',arg=selection)
            page.wait_for_function('s => document.getElementById("architecture-explorer").dataset.peer === s',arg=peer)
            page.wait_for_function('[...document.querySelectorAll(".architecture-annotation")].every(i => i.complete && i.naturalWidth===640 && i.naturalHeight===400)')
            scene = next(s for s in data['scenes'] if s['seed']==seed)
            frame = scene['frames'][step]
            assert page.locator('#architecture-explorer').get_attribute('aria-busy')=='false'
            assert page.locator('#architecture-plan').get_attribute('src').endswith(frame['plan'])
            assert page.locator('#architecture-metadata').get_attribute('href').endswith(scene['metadata'])
            assert page.locator('#architecture-masks').get_attribute('href').endswith(scene['masks'])
            assert page.locator('#architecture-visibility').is_visible()==(mode=='co_visibility')
            for c,v in enumerate(frame['views']):
                tile = page.locator(f'[data-architecture-camera="{c}"]')
                assert tile.locator('.architecture-rgb').get_attribute('src').endswith(v['images']['color'])
                expected = v['images'][mode]
                if mode=='co_visibility' and peer!='all':
                    expected = v['images']['color'] if c==int(peer) else v['images']['peers'][int(peer)]
                assert tile.locator('.architecture-annotation').get_attribute('src').endswith(expected)
                assert tile.locator('a').get_attribute('href').endswith(expected)
                if mode=='co_visibility':
                    assert tile.locator('.architecture-shared').inner_text()

        for scene in data['scenes']:
            page.select_option('#architecture-room',str(scene['seed']))
            for step in (0,1):
                page.click(f'[data-architecture-step="{step}"]')
                for mode in data['modes']:
                    page.click(f'[data-architecture-mode="{mode}"]')
                    selected(scene['seed'],step,mode)
                    cases += 1
                for peer in range(4):
                    page.click(f'[data-architecture-peer="{peer}"]')
                    selected(scene['seed'],step,'co_visibility',str(peer))
                    assert page.locator('#architecture-legend').is_hidden()
                    peer_cases += 1
                page.click('[data-architecture-peer="all"]')
                selected(scene['seed'],step,'co_visibility')
                assert page.locator('#architecture-legend .legend-swatch').count()==4
        for feature in data['feature_room_counts']:
            expected = [s for s in data['scenes'] if feature in s['features']]
            if not expected: continue
            page.select_option('#architecture-filter',feature)
            assert page.locator('#architecture-room option').count()==len(expected)
        page.evaluate('''() => {
          document.querySelector('[data-architecture-seed="8"]').click();
          document.querySelector('[data-architecture-mode="normal"]').click();
          document.querySelector('[data-architecture-seed="7"]').click();
          document.querySelector('[data-architecture-step="0"]').click();
          document.querySelector('[data-architecture-mode="semantic"]').click();
        }''')
        selected(7,0,'semantic')
        page.locator('#architecture-reveal').fill('75')
        page.locator('#architecture-reveal').focus();page.keyboard.press('ArrowLeft')
        assert page.locator('#architecture-reveal').input_value()=='74'
        page.locator('[data-architecture-jump="co_visibility"]').click()
        selected(7,0,'co_visibility')
        for width in (390,768,1440):
            page.set_viewport_size({'width':width,'height':1000})
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            assert page.locator('.architecture-views').evaluate('e => getComputedStyle(e).gridTemplateColumns.split(" ").length')==(1 if width<640 else 2)
            page.locator('#architecture').screenshot(path=str(output/f'architecture-{width}.png'))
            page.locator('#architecture-explorer').screenshot(path=str(output/f'co-visibility-{width}.png'))
        plain = browser.new_context(java_script_enabled=False,viewport={'width':390,'height':844})
        fallback = plain.new_page();fallback.goto(url,wait_until='networkidle')
        assert fallback.locator('#architecture-room').is_disabled()
        assert fallback.locator('.architecture-views figure').count()==4
        fallback.locator('.architecture-cohort summary').click()
        assert fallback.locator('.architecture-cohort img').count()==32
        assert fallback.locator('img[src$="architecture_co_visibility.jpg"]').count()==1
        assert fallback.locator('#architecture-masks').get_attribute('href').endswith('s7-visibility.zip')
        linked = context.new_page();linked.goto(url+'#explore',wait_until='networkidle')
        linked.wait_for_function('document.getElementById("architecture-explorer").dataset.selection === "7:0:co_visibility"')
        assert linked.locator('#architecture-visibility').is_visible()
        assert linked.locator('.skip-link').evaluate('e => getComputedStyle(e).clipPath')=='inset(50%)'
        linked.locator('.skip-link').focus()
        assert linked.locator('.skip-link').evaluate('e => getComputedStyle(e).clipPath')=='none'
        assert not errors,errors
        browser.close()
    return dict(passed=True,room_time_mode_cases=cases,view_cases=cases*4,feature_filters=True,
                atomic_switching=True,keyboard_reveal=True,responsive_widths=[390,768,1440],
                no_javascript_fallback=True,co_visibility_peer_cases=peer_cases,unified_annotation_gallery=True,errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static-only',action='store_true')
    parser.add_argument('--captures',type=Path,default=ROOT/'out/architecture_covisibility')
    parser.add_argument('--url',default='http://127.0.0.1:8770/project/')
    parser.add_argument('--output',type=Path,default=ROOT/'out/architecture_media_validation')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    report = {'static':static_checks(args.captures if args.captures.is_dir() else None)}
    if not args.static_only:
        try:
            report['browser'] = browser_checks(args.url,args.output)
        except Exception as error:
            report['browser'] = {'passed':False,'error':str(error)}
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            raise
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

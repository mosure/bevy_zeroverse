import json, hashlib, shutil
from pathlib import Path
root=Path(__file__).resolve().parents[3]
source=root/'out/human_review/verified_motion'
out=root/'docs/evidence/human_review_v11'
planning=json.loads((source/'planning.json').read_text())
scenes=[]
for seed in [0,1,13,34]:
 d=json.loads((source/str(seed)/'motion.json').read_text())
 c=json.loads((source/str(seed)/'capture.json').read_text())
 clips=dict(json.loads((source/str(seed)/'clips.json').read_text()))
 motions=[]
 for plan in d['accepted']:
  clip=clips[plan['actor_id']]
  ps=[f['root_translation'] for f in clip['frames']]
  distance=sum(((a[0]-b[0])**2+(a[2]-b[2])**2)**.5 for a,b in zip(ps,ps[1:]))
  displacement=((ps[-1][0]-ps[0][0])**2+(ps[-1][2]-ps[0][2])**2)**.5
  motions.append({'actor':plan['actor_id'],'behavior':plan['behavior'],'prompt':plan['request']['prompt'],'seed':plan['request']['seed'],'waypoints':plan['request']['waypoints'],'path_m':distance,'endpoint_displacement_m':displacement})
 b=c['object_obbs'];people=len(c['scene']['humans'])
 assert len(b)==len(c['scene']['objects'])+people
 assert sum(x['class_name']=='person' for x in b)==people
 scenes.append({'seed':seed,'humans':people,'requested':len(d['requested']),'accepted':len(d['accepted']),'rejected':d['rejected'],'objects':len(c['scene']['objects']),'object_and_person_boxes':len(b),'model_loads':d['model_loads'],'generated_batches':d['generated_batches'],'accepted_motions':motions,'flow':c['flow_checks'],'annotation_checks':c['annotation_checks']})
report={'schema':1,'generator_version':11,'status':'bounded local diagnostic; not a photographic-realism benchmark',
 'policy':{'fraction':1,'locomotion_fraction':1,'max_actors':16,'frames':120,'batch_size':4,'max_attempts':2},
 'planning':{'scenes':len(planning['scenes']),'humans':sum(x['humans'] for x in planning['scenes']),'planned':sum(len(x['plans']) for x in planning['scenes']),'rejected':sum(len(x['rejected']) for x in planning['scenes']),'behaviors':planning['behaviors']},
 'render':{'scenes':scenes,'humans':sum(x['humans'] for x in scenes),'accepted':sum(x['accepted'] for x in scenes),'views':sum(len(x['annotation_checks']) for x in scenes),'moving_person_flow_pixels':sum(x['flow']['moving_person_pixels'] for x in scenes),'missing_static_flow_correspondences':sum(v['missing_static_valid_pixels'] for x in scenes for v in x['flow']['views'])},
 'appearance':{'people':6,'views':12,'script':'examples/review_humans.rs','source':'out/human_review/verified','lighting':'850 lux key plus 250 lux fill; fixed studio controls; sRGB PNG encoding'},
 'capture_command':"target/debug/motion_validate --seeds 64 --render-seed-list 0,1,13,34 --flow --policy '{\"fraction\":1,\"locomotion_fraction\":1,\"max_actors\":16,\"frames\":120,\"batch_size\":4}' --output out/human_review/verified_motion",
 'limits':['3 clips rejected for overlap of conservative generated body bounds','Wasm was compile-checked; no new browser runtime qualification','Procedural skin/clothing/hair are approximate; no photographic realism claim'],
}
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
for kind in ['portrait','body']:
 for i in range(6):shutil.copyfile(root/f'out/human_review/verified/{kind}_{i}.png',out/f'{kind}_{i}.png')
html='''<!doctype html><meta charset="utf-8"><title>Zeroverse human review v11</title><style>body{background:#191b20;color:#eee;font:16px system-ui;margin:30px}main{display:grid;grid-template-columns:repeat(3,minmax(250px,1fr));gap:20px}img{width:100%;background:#111}a{color:#9bf}figcaption{padding:8px}figure{margin:0}p{max-width:80ch}</style><h1>Human appearance and motion review</h1><p>Six Anny phenotypes, fitted clothes, opaque scalp volumes, surface-fitted eyebrows and eyewear. Fixed studio lighting. These captures verify geometry and surface changes; they do not establish photographic realism.</p><p>64-room planning audit: 347/398 people received feasible plans. Real ARDY captures: 14/17 admitted, 3 retained static after overlapping body bounds. 40 annotated views; see <a href="summary.json">machine-readable metrics and rejection reasons</a>.</p><main>'''
for kind in ['portrait','body']:
 for i in range(6):html+=f'<figure><img src="{kind}_{i}.png"><figcaption>{kind.capitalize()} {i}</figcaption></figure>'
(out/'review.html').write_text(html+'</main>\n')
print(json.dumps({'planned':report['planning'],'render':{k:v for k,v in report['render'].items() if k!='scenes'}},indent=2))

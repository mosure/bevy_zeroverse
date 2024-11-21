import objaverse


annotations = objaverse.load_annotations()
print('total number of annotations:', len(annotations))

cc_by_uids = [uid for uid, annotation in annotations.items() if annotation["license"] == "by"]
print('number of CC-BY annotations:', len(cc_by_uids))

to_download = []
filter_tags = ['chair', 'table', 'sofa']
for uid in cc_by_uids:
    annotation = annotations[uid]

    if any(tag['name'] in filter_tags for tag in annotation["tags"]):
        to_download.append(uid)

print('number of CC-BY annotations with tags:', len(to_download))

locations = objaverse.load_objects(
    uids=to_download,
    download_processes=1,
)

import shutil
import os

os.makedirs('./assets/models', exist_ok=True)

for uid, location in locations.items():
    extension = location.split('.')[-1]

    if extension != 'glb':
        continue

    print('moving from location:', location)
    shutil.move(location, f'./assets/models/{uid}.glb')

import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0,'scripts')
from validate_indoor_ovoxel import tensors,audit
root=Path('out/ovoxel_review');dest=Path('docs/evidence/ovoxel')
for name in ['validation.json','validation128.json','validation_fs.json','capture_runs_v3.json','capture_runs_128.json']:
 (dest/name).write_bytes((root/name).read_bytes())
fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
for row,(folder,index) in enumerate([('cpu_v3',1),('cpu_128',0)]):
 report,samples=audit(root/folder)
 manifest,coords,dual,flags,semantic,labels,colors=samples[index]
 data=tensors(root/folder/'000000.safetensors')
 rgb=data['color'][index,0,0]
 axes[row,0].imshow(np.clip(rgb[:,:,:3],0,1));axes[row,0].axis('off')
 axes[row,0].set_title(f"Seed {manifest['seed']}: primary-room RGB")
 zone=report['samples'][index]['primary_zone'];lo=np.array(zone['min']);hi=np.array(zone['max'])
 ax=axes[row,1]
 for z in manifest['program']['zones']:
  zlo=np.array(z['min']);zhi=np.array(z['max'])
  ax.add_patch(Rectangle(zlo,*(zhi-zlo),facecolor='#f0f0f0',edgecolor='#999',lw=1))
 for o in manifest['objects']:
  pos=np.array(o['position'])[::2]
  inside=not o['neighbor'] and (lo<=pos).all() and (pos<=hi).all()
  ax.scatter(*pos,s=16,color='#186e9c' if inside else '#b9b9b9')
 ax.add_patch(Rectangle(lo,*(hi-lo),fill=False,edgecolor='#df6323',lw=2))
 ax.autoscale();ax.set_aspect('equal');ax.set_title(f"Selected objects {report['samples'][index]['primary_objects']}/{len(manifest['objects'])}")
 ax.set_xlabel('local X (m)');ax.set_ylabel('local Z (m)')
 bounds=data['ovoxel_aabb'][index];res=int(data['ovoxel_resolution'][index])
 world=bounds[0]+(coords+dual/255)*(bounds[1]-bounds[0])/res
 yaw=manifest['world_yaw'];c,s=np.cos(yaw),np.sin(yaw)
 local=world@np.array([[c,0,s],[0,1,0],[-s,0,c]])
 keep=(local[:,1]>.18)&(local[:,1]<1.7)
 linear=colors[keep,:3]/255
 srgb=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)
 ax=axes[row,2]
 ax.scatter(local[keep,0],local[keep,2],c=srgb,s=1.5,linewidths=0)
 ax.add_patch(Rectangle(lo-.2,*(hi-lo+.4),fill=False,edgecolor='#df6323',lw=1))
 ax.set_aspect('equal');ax.set_title(f"O-voxel cutaway (0.18–1.7 m), {res}³ grid")
 ax.set_xlabel('local X (m)');ax.set_ylabel('local Z (m)')
fig.suptitle('Primary-room scope: orange boundary; gray context excluded; voxel colors are semantic',fontsize=12)
fig.savefig(dest/'primary_room_review.png',dpi=150)
plt.close(fig)

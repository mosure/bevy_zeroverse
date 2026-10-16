/* Current material evidence. All data strings are inserted as text. */
(() => {
  const root=document.getElementById('material-atlas'); if(!root)return;
  const select=document.getElementById('material-category'),mode=document.getElementById('material-mode');
  const content=document.getElementById('material-content'),parameters=document.getElementById('material-parameters'),usagePanel=document.getElementById('material-usage');
  const base=root.dataset.source.replace(/[^/]+$/,'');
  const node=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
  const number=v=>v===null||v===undefined?'—':typeof v==='number'?Number(v.toPrecision(4)).toString():Array.isArray(v)?'['+v.map(number).join(', ')+']':typeof v==='object'?JSON.stringify(v):String(v);
  const label=s=>s.replace(/([a-z])([A-Z])/g,'$1 $2').replace(/_/g,' ');
  const get=(s,path)=>path.split('.').reduce((v,k)=>v?.[k],s);
  const bars=counts=>{const e=node('div',undefined,'object-bars'),max=Math.max(1,...counts);for(const [i,n]of counts.entries()){const b=node('span');b.style.height=(100*n/max)+'%';b.title=i+': '+n;e.append(b);}e.setAttribute('aria-label','Counts: '+counts.join(', '));return e;};
  let atlas,usage;
  function render(){
    const c=atlas.categories.find(c=>c.name===select.value),rows=c.selected;
    const metrics=node('div',undefined,'object-metrics');
    for(const [value,title]of [[atlas.sample_count,'Consecutive material seeds'],[c.structure_count,'Shared structures / room'],[c.finish_slots,'Finish slots / room'],[atlas.object_count.toLocaleString(),'Objects in assignment audit']]){const e=node('div');e.append(node('strong',value),node('span',title));metrics.append(e);}
    const note={color:'Fixed-light PBR swatches. Click any sample for the full-resolution render.',normal:'Rendered view-space normals, encoded in [0,1]. This is not a tangent-space texture.',albedo:'Raw sRGB base-color map before the material multiplier shown below. Untextured roles show their scalar color.',normal_map:'Tangent-space normal map, level zero. Neutral blue means no normal texture.',roughness:'Packed G-channel roughness before its scalar multiplier; untextured roles show the scalar value. White is rough, black is smooth.'};
    const grid=node('div',undefined,'material-grid');
    rows.forEach((r,i)=>{const fig=node('figure'),a=node('a'),img=node('img');a.href=base+c.name+'-'+i+'-'+mode.value+'.webp';a.target='_blank';a.rel='noopener';img.src=a.href;img.width=384;img.height=384;img.alt=label(c.name)+', material seed '+r.seed+', '+mode.selectedOptions[0].text; a.append(img);fig.append(a,node('figcaption','seed '+r.seed+' · '+(r.finish_slot===null?'base role':'finish '+r.finish_slot)+' · structure '+r.structure_group));grid.append(fig);});
    const wrap=node('div',undefined,'material-scalars'),table=node('table'),head=node('tr');head.append(node('th','Actual material binding'));rows.forEach(r=>head.append(node('th','seed '+r.seed)));table.append(head);
    for(const [key,title]of [['roughness_multiplier','Roughness multiplier'],['metallic','Metallic'],['clearcoat','Clearcoat'],['anisotropy','Anisotropy'],['specular_transmission','Specular transmission'],['diffuse_transmission','Diffuse transmission'],['ior','IOR'],['thickness_m','Thickness (m)'],['base_color_srgba','Base color (sRGBA)'],['attenuation_linear_rgb','Attenuation (linear RGB)'],['attenuation_distance_m','Attenuation distance (m)'],['emissive_linear_rgb','Emission (linear RGB)']]){const tr=node('tr');tr.append(node('th',title));for(const r of rows)tr.append(node('td',number(r.actual_material[key])));table.append(tr);}wrap.append(table);
    content.replaceChildren(metrics,node('p',note[mode.value],'object-status'),grid,wrap);
    parameters.replaceChildren();
    for(const [key,p]of Object.entries(c.parameters)){
      const detail=node('details');detail.append(node('summary',label(key)));
      if(p.type==='numeric'){detail.append(node('p',number(p.min)+' – '+number(p.max)+'; median '+number(p.median)+'; '+p.present+' recorded values.'),bars(p.bins));}
      else detail.append(node('p',Object.entries(p.counts).map(([k,v])=>k+': '+v).join('; ')));
      detail.append(node('p','Displayed seeds: '+rows.map(r=>number(get(r,key))).join(' · ')));parameters.append(detail);
    }
    const usageTable=node('table'),thead=node('thead'),titles=node('tr');['Object category','Objects using role / total','Material parts','Finish bindings'].forEach(t=>titles.append(node('th',t)));thead.append(titles);usageTable.append(thead);
    const body=node('tbody'),assignments=Object.entries(usage.usage).filter(([,o])=>o[c.name]).sort((a,b)=>b[1][c.name].parts-a[1][c.name].parts);
    for(const [kind,objects]of assignments){const r=objects[c.name],tr=node('tr'),td=node('td');td.append(bars(r.slots));const special=Object.entries(r.special_slots||{});if(special.length)td.append(node('p','Food: '+special.map(([slot,count])=>slot+': '+count).join(' · ')));if(r.base_parts)td.append(node('p','Base: '+r.base_parts));tr.append(node('td',label(kind)),node('td',r.objects+' / '+usage.object_counts_by_category[kind]+' ('+(100*r.objects/usage.object_counts_by_category[kind]).toFixed(1)+'%)'),node('td',r.parts),td);body.append(tr);}usageTable.append(body);usagePanel.replaceChildren(assignments.length?usageTable:node('p','This role was not present in the audited object assemblies. Architectural roles are assessed in the room captures.'));
    document.getElementById('material-usage-note').textContent=atlas.room_count+' consecutive rooms, density 0.65, primary objects only. '+assignments.length+' object categories use this role. Counts include actual constructed material parts; a single object can use several roles. Bars show generic slots 0–11; reserved food bindings and base-material parts are counted separately.';
    const url=new URL(location.href);url.searchParams.set('material',c.name);url.searchParams.set('material_mode',mode.value);history.replaceState(null,'',url);
  }
  const fetchJson=async url=>{const r=await fetch(url);if(!r.ok)throw new Error('Missing '+url);return r.json();};
  Promise.all([fetchJson(root.dataset.source),fetchJson(base+'usage.json')]).then(([a,u])=>{
    atlas=a;usage=u;for(const c of a.categories){const o=node('option',label(c.name));o.value=c.name;select.append(o);}
    const query=new URLSearchParams(location.search);select.value=a.categories.some(c=>c.name===query.get('material'))?query.get('material'):'Ceramic';
    if([...mode.options].some(o=>o.value===query.get('material_mode')))mode.value=query.get('material_mode');
    select.addEventListener('change',render);mode.addEventListener('change',render);render();
  }).catch(error=>content.replaceChildren(node('p','Material atlas could not load: '+error.message)));
})();

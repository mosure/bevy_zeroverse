use burn_human::{AnnyBody, AnnyInput};
fn main() {
 let b=AnnyBody::from_reference_paths("assets/burn_human/fullbody_default.safetensors","assets/burn_human/fullbody_default.meta.json").unwrap();
 println!("case,gender,height_anchor,native_stature_m,shoulder_span_m");
 for (case,age,muscle,weight,proportions) in [("neutral",0.7,0.5,0.5,0.5),("light",0.67,0.12,0.16,0.02),("heavy",0.7,0.5,0.84,0.5)] { for gender in [0.05,0.95] { for i in 0..=10 {
  let height=i as f64/10.;
  let p:Vec<f64>=b.metadata().metadata.phenotype_labels.iter().map(|n|match n.as_str(){"gender"=>gender,"age"=>age,"muscle"=>muscle,"weight"=>weight,"proportions"=>proportions,"height"=>height,_=>0.5}).collect();
  let out=b.forward(AnnyInput{phenotype_inputs:Some(&p),..Default::default()}).unwrap();
  let mut lo=f64::INFINITY; let mut hi=f64::NEG_INFINITY;
  for v in out.rest_vertices.data.chunks_exact(3) {lo=lo.min(v[2]);hi=hi.max(v[2]);}
  let (labels,_)=b.bone_hierarchy();
  let head=|n:&str| {let i=labels.iter().position(|x|x==n).unwrap();let m=&out.rest_bone_poses.data[i*16..];[m[3],m[7],m[11]]};
  let a=head("upperarm01.L");let c=head("upperarm01.R");let span=(0..3).map(|j|(a[j]-c[j]).powi(2)).sum::<f64>().sqrt();
  println!("{case},{gender},{height},{},{span}",hi-lo);
 }}}
}

"""Protect integer membership, validity and camera-order contracts in page media."""
from io import BytesIO
import unittest

import numpy as np
from PIL import Image

from indoor_visibility import decode, palette


def png(array):
    out = BytesIO()
    Image.fromarray(array).save(out,format='PNG')
    return out.getvalue()


def fixture(count=4, mask=14):
    legend=[]
    for bit in range(count):
        channel=bit%3; n=(count-channel+2)//3
        rgb=[0,0,0]; rgb[channel]=(255//((1<<n)-1))*(1<<(n-1-bit//3))
        legend.append(dict(bit=bit,camera_index=bit,mask=1<<bit,rgb8=rgb))
    metadata=dict(schema_version=1,camera_count=count,legend=legend)
    masks=np.array([[mask,0,0]],dtype=np.uint16)
    valid=np.array([[1,1,0]],dtype=np.uint8)
    peers=[int(bool(mask&(1<<i))) for i in range(count)]
    cardinality=[0]*count; cardinality[0]=1; cardinality[mask.bit_count()]+=1
    stats=dict(valid_pixels=2,shared_pixels=1,peer_pixels=peers,cardinality_pixels=cardinality,
               shared_fraction_valid=.5,peer_fraction_valid=[n/2 for n in peers])
    capture=dict(image_size=[3,1],co_visibility_metadata=metadata,
                 views=[dict(camera_index=0,co_visibility=stats)])
    return masks,valid,capture,palette(metadata,count)[masks]


class VisibilityTest(unittest.TestCase):
    def test_valid_unshared_surface_differs_from_background(self):
        m,v,c,rgb=fixture()
        masks,valid,_=decode(png(m),png(v),c,0,4,png(rgb))
        self.assertEqual(masks[0,1],masks[0,2])
        self.assertTrue(valid[0,1]);self.assertFalse(valid[0,2])

    def test_sixteen_camera_high_bit_survives(self):
        m,v,c,rgb=fixture(16,(1<<15)|(1<<1))
        masks,_,_=decode(png(m),png(v),c,0,16,png(rgb))
        self.assertEqual(int(masks[0,0]),32770)

    def test_reject_lossy_eight_bit_membership(self):
        m,v,c,rgb=fixture()
        with self.assertRaisesRegex(ValueError,'16-bit'):
            decode(png(m.astype(np.uint8)),png(v),c,0,4,png(rgb))

    def test_source_camera_is_excluded(self):
        m,v,c,rgb=fixture(mask=3)
        with self.assertRaisesRegex(ValueError,'source exclusion'):
            decode(png(m),png(v),c,0,4,png(rgb))

    def test_background_cannot_claim_membership(self):
        m,v,c,rgb=fixture();v[0,0]=0
        with self.assertRaisesRegex(ValueError,'background'):
            decode(png(m),png(v),c,0,4,png(rgb))

    def test_wrong_camera_legend_is_rejected(self):
        m,v,c,rgb=fixture();c['co_visibility_metadata']['legend'][0]['camera_index']=2
        with self.assertRaisesRegex(ValueError,'camera ordering'):
            decode(png(m),png(v),c,0,4,png(rgb))

    def test_corrupt_display_color_is_rejected(self):
        m,v,c,rgb=fixture();rgb[0,0]=0
        with self.assertRaisesRegex(ValueError,'preview'):
            decode(png(m),png(v),c,0,4,png(rgb))

    def test_changed_report_counts_are_rejected(self):
        m,v,c,rgb=fixture();c['views'][0]['co_visibility']['shared_pixels']=2
        with self.assertRaisesRegex(ValueError,'reported|report'):
            decode(png(m),png(v),c,0,4,png(rgb))


if __name__=='__main__':
    unittest.main()

"""Protect the matched-room comparison from accidental scene confounding."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from build_baseline_evaluation import scene_hash
from indoor_camera_group_report import group_geometry


class BaselineEvidence(unittest.TestCase):
    def test_matching_ignores_cameras_but_not_lighting_people_or_geometry(self):
        manifest = dict(seed=7, generator_version=21, objects=[{'position':[1,2,3]}],
                        humans=[{'pose':1}], lighting={'lux':100}, cameras=[{'start':[0,0,0]}],
                        camera_settings={'baseline':0}, camera_aspect_ratio=4/3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            def digest(value):
                path.write_text(json.dumps(value))
                return scene_hash(path)
            reference = digest(manifest)
            other = copy.deepcopy(manifest)
            other.update(cameras=[{'start':[1,0,0]}], camera_settings={'baseline':1}, camera_aspect_ratio=1)
            self.assertEqual(reference,digest(other))
            for key in ['seed','generator_version','objects','humans','lighting']:
                changed = copy.deepcopy(other)
                changed[key] = None
                self.assertNotEqual(reference,digest(changed),key)

    def test_reference_distance_is_distinct_from_closest_nonreference_pair(self):
        # A close pair of peers can still be far from the reference camera.
        result = group_geometry([[[0,0,0],[3,0,0],[3,0,1]]])
        self.assertAlmostEqual(result['min_pairwise_baseline_m'],1)
        self.assertAlmostEqual(result['min_reference_baseline_m'],3)
        self.assertAlmostEqual(result['max_reference_baseline_m'],10**.5)


if __name__ == '__main__':
    unittest.main()

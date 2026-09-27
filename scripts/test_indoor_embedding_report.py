import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from indoor_embedding_report import cohort_metrics, cross_scene_nearest, load_embeddings, spectrum


class EmbeddingAuditTests(unittest.TestCase):
    def test_excludes_all_views_of_same_scene(self):
        vectors=np.array([[1,0],[1,0],[0,1],[0,1]],dtype=np.float32)
        distances, neighbors=cross_scene_nearest(vectors,[1,1,2,2])
        np.testing.assert_allclose(distances,1)
        self.assertTrue(all(([1,1,2,2][i] != [1,1,2,2][j]) for i,j in enumerate(neighbors)))

    def test_trajectory_similarity_is_not_cross_scene_redundancy(self):
        vectors=np.array([[1,0],[1,0],[0,1],[0,1]],dtype=np.float32)
        rows=[dict(seed=s,camera=0,time=t,sha256=str(s)) for s in [1,2] for t in [0,1]]
        metrics=cohort_metrics(vectors,rows)
        self.assertEqual(metrics['same_camera_trajectory_cosine_distance']['max'],0)
        self.assertEqual(metrics['cross_scene_nearest_cosine_distance']['min'],1)
        self.assertEqual(metrics['exact_file_duplicate_cosine_distance']['max'],0)
        self.assertEqual(len(metrics['closest_cross_scene_pairs']),1)
        self.assertAlmostEqual(spectrum(np.array([[1,0],[1,0]]))['effective_rank'],0)

    def test_rejects_corrupted_tensor_and_nonunit_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); tensor=np.array([[2,0],[0,1]],dtype='<f4').tobytes()
            (path/'embeddings.f32').write_bytes(tensor)
            metadata=dict(schema_version=1,tensor_sha256=hashlib.sha256(tensor).hexdigest(),shape=[2,2],samples=[{},{}])
            (path/'embeddings.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'unnormalized'): load_embeddings(path/'embeddings.json')
            (path/'embeddings.f32').write_bytes(tensor[:-1])
            with self.assertRaisesRegex(ValueError,'checksum'): load_embeddings(path/'embeddings.json')


if __name__ == '__main__': unittest.main()

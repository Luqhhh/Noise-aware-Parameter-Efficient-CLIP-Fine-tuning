"""Synthetic checks for cached evidence semantics and safe output handling."""
import csv
import gzip
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('recheck', Path(__file__).resolve().parents[1]/'p75_evidence_recheck.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class RecheckTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for a, b, c in itertools.product((False, True), repeat=3):
            self.add(split='train', neighbor_votes=json.dumps({'0':12 if a else 5, '1':8 if a else 15}),
                     original_support_votes=str(12 if a else 5),
                     oof_centroid_prediction=str(0 if b else 1), oof_ridge_prediction=str(0 if c else 1),
                     channel='original' if a and b and c else 'uncertain',
                     original_supported=str(a and b and c))
        # 15 one-way errors must survive; center prediction deliberately differs from fixed decode.
        for _ in range(15):
            self.add(neighbor_votes='{"0":9,"1":6,"2":5}', original_support_votes='9')
        self.add(neighbor_votes='{"1":12,"0":8}', original_support_votes='8')
        self.add(neighbor_votes='{"2":12,"0":8}', original_support_votes='8')
        self.add(neighbor_votes='{"0":10,"1":10}', original_support_votes='10')
        self.add(neighbor_votes='{}', original_support_votes='0')
        self.add(l05_fixed_decode_prediction='0')

    def add(self, **kwargs):
        n = len(self.rows)
        row = dict(split='val', image_path=f'image{n}', content_group=f'group{n}',
                   original_label='0', channel='diagnostic_only', original_supported='False',
                   independent_neighbor_groups='20', original_support_votes='12',
                   neighbor_votes='{"0":12,"1":8}', oof_centroid_prediction='', oof_ridge_prediction='',
                   own_content_group_label_conflict='False', l05_prediction='0',
                   l05_fixed_decode_prediction='1', center_label_probability='0.2', unknown='True')
        row.update(kwargs)
        self.rows.append(row)

    def write(self):
        source = self.root/'sample_evidence.csv.gz'
        with gzip.open(source, 'wt', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader(); writer.writerows(self.rows)
        summary = self.root/'summary.json'
        summary.write_text(json.dumps(dict(identity=dict(num_classes=3, training_samples=8, validation_samples=20),
            baseline_errors=19, files={'sample_evidence.csv':m.digest(source, True)},
            archived_artifacts={source.name:m.digest(source)})))
        return source, summary

    def test_eight_masks_and_all_validation_relations(self):
        source, summary = self.write()
        result = m.run(source, summary, self.root/'out')
        self.assertEqual([r['rows'] for r in result['signal_masks']], [1]*8)
        self.assertEqual([r['final_snapshot_hard'] for r in result['signal_masks']], [1]*8)
        self.assertEqual(result['confusion']['errors_on_pairs_with_no_reverse'], 19)
        self.assertEqual(result['confusion']['P0_retained_errors'], 0)
        self.assertEqual(result['confusion']['top_directed_pair_cumulative_errors']['10'], 19)
        p = result['validation_neighbor_partitions']
        self.assertEqual([p[k]['rows'] for k in m.RELATIONS], [1,15,1,1,1,1])
        self.assertEqual(sum(v['prior_unknown_errors'] for v in p.values()), 19)
        self.assertEqual(result['counts']['val_errors_original_is_plurality_but_below_12_votes'], 15)
        self.assertEqual(len(list((self.root/'out').iterdir())), 4)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            m.run(source, summary, self.root/'out')

    def test_structural_veto_separate_from_mask(self):
        self.rows[7].update(own_content_group_label_conflict='True', channel='uncertain', original_supported='False')
        source, _ = self.write()
        result = m.analyze(source, 3)
        mask = result['signal_masks'][7]
        self.assertEqual((mask['rows'], mask['structural_eligible']), (1,0))
        self.assertEqual(result['counts'].get('channel_original', 0), 0)

    def test_hash_and_count_fail_without_outputs(self):
        source, summary = self.write()
        original = json.loads(summary.read_text())
        for field in ('hash', 'raw_hash', 'count', 'classes'):
            ref = json.loads(json.dumps(original))
            if field == 'hash': ref['archived_artifacts'][source.name] = 'bad'
            elif field == 'raw_hash': ref['files']['sample_evidence.csv'] = 'bad'
            elif field == 'count': ref['baseline_errors'] = 20
            else: ref['identity']['num_classes'] = 4
            summary.write_text(json.dumps(ref))
            with self.assertRaises(ValueError):
                m.run(source, summary, self.root/'out', classes=3)
            self.assertFalse((self.root/'out').exists())

    def test_reject_fabricated_validation_oof_and_tampered_votes(self):
        for change in ({'oof_centroid_prediction':'0'}, {'original_support_votes':'11'}):
            old = self.rows[8].copy()
            self.rows[8].update(change)
            source, _ = self.write()
            with self.assertRaisesRegex(ValueError, 'CSV line'):
                m.analyze(source, 3)
            self.rows[8] = old

    def test_reconstruct_old_symmetric_filter(self):
        for _ in range(2):
            self.add(original_label='1', l05_fixed_decode_prediction='0',
                     neighbor_votes='{"1":12,"0":8}', original_support_votes='12')
        source, _ = self.write()
        result = m.analyze(source, 3)
        self.assertEqual(result['confusion']['P0_retained_undirected_pairs'], 1)
        self.assertEqual(result['confusion']['P0_retained_errors'], 21)
        self.assertEqual(result['confusion']['errors_on_pairs_with_no_reverse'], 0)

    def test_no_errors_csv_still_has_schema(self):
        for r in self.rows:
            if r['split'] == 'val': r['l05_fixed_decode_prediction'] = r['original_label']
        source, summary = self.write()
        ref = json.loads(summary.read_text()); ref['baseline_errors'] = 0
        summary.write_text(json.dumps(ref))
        m.run(source, summary, self.root/'out')
        self.assertIn('reverse_errors', (self.root/'out'/'directed_confusions.csv').read_text())


if __name__ == '__main__':
    unittest.main()

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import json

spec=importlib.util.spec_from_file_location('visual',Path(__file__).resolve().parents[1]/'scripts/p75_visual_audit.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class VisualAuditTest(unittest.TestCase):
    def row(self,y=0,p=1,votes=None,i=0):
        return dict(split='val',image_path=f'train/{i}.jpg',original_label=str(y),
                    l05_fixed_decode_prediction=str(p),neighbor_votes=json.dumps(votes or {'0':12,'1':8}))

    def test_partition_never_uses_confidence_or_arbitrary_tie_break(self):
        cases=[self.row(),self.row(votes={'1':12}),self.row(votes={'2':12}),
               self.row(votes={'0':10,'1':10}),self.row(p=0)]
        self.assertEqual([m.classify(r)[0] for r in cases],list(m.BUDGET))
        self.assertEqual(m.classify(cases[3])[1],[0,1])

    def test_stratified_sampling_and_blind_shuffle_are_order_independent(self):
        rows=[]
        for n in range(15):
            i=n*5
            rows.extend([self.row(i=i),self.row(votes={'1':12},i=i+1),self.row(votes={'2':12},i=i+2),
                         self.row(votes={'0':10,'1':10},i=i+3),self.row(p=0,i=i+4)])
        with patch.object(m,'EXPECTED',{k:15 for k in m.BUDGET}),patch.object(m,'BUDGET',{k:3 for k in m.BUDGET}):
            a=m.sample(rows,42); b=m.sample(list(reversed(rows)),42)
            self.assertEqual(a,b);self.assertEqual(len(a),15)
            self.assertEqual(len({r['image_path'] for r in a}),15)
            self.assertNotEqual(a,m.sample(rows,43))
            with self.assertRaises(ValueError):m.sample(rows[:-1],42)

    def test_references_uniform_groups_and_exclude_query_group(self):
        train={0:[dict(image_path=f'train/{i}',content_group=str(i//2)) for i in range(12)]}
        result=m.reference_sample(train,{0},{'0'},42)[0]
        self.assertEqual(len(result),3)
        self.assertEqual(len({r['content_group'] for r in result}),3)
        self.assertNotIn('0',{r['content_group'] for r in result})
        self.assertEqual(result,m.reference_sample(train,{0},{'0'},42)[0])



spec2=importlib.util.spec_from_file_location('summary',Path(__file__).resolve().parents[1]/'scripts/summarize_p75_visual_audit.py')
s=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(s)


class WeightedSummaryTest(unittest.TestCase):
    def data(self):
        cases=[];obs=[];judgments=[]
        for i in range(200):
            cid=f'V{i+1:03d}';control=i>=100
            cases.append(dict(case_id=cid,stratum='control_agreement' if control else 'original_plurality',
                              population=1000 if control else 100,sample_size=100,weight=10 if control else 1))
            obs.append(dict(case_id=cid,flags=['crop_part_loss'] if i<50 else []))
            judgments.append(dict(case_id=cid,status='cannot_determine',training_use_allowed=False))
        return cases,obs,judgments

    def test_weights_not_naive_sample_fraction_and_unknown_retained(self):
        r=s.weighted(*self.data())['weighted']
        self.assertAlmostEqual(r['all_validation']['phenomena']['crop_part_loss']['weighted_fraction'],50/1100)
        self.assertEqual(r['all_validation']['status_estimated_counts']['cannot_determine'],1100)
        self.assertEqual(r['original_label_errors']['phenomena']['crop_part_loss']['weighted_fraction'],.5)
        self.assertEqual(r['agreement_controls']['phenomena']['crop_part_loss']['weighted_fraction'],0)

    def test_incomplete_or_training_use_refused(self):
        c,o,a=self.data()
        with self.assertRaises(ValueError):s.weighted(c,o[:-1],a)
        a[0]['training_use_allowed']=True
        with self.assertRaises(ValueError):s.weighted(c,o,a)


if __name__=='__main__':unittest.main()

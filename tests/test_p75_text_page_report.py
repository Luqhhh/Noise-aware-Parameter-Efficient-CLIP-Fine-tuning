import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def fixture(tmp):
    scan=tmp/'scan';scan.mkdir();images=tmp/'images';(images/'train').mkdir(parents=True)
    im=images/'train'/'one.jpg';Image.new('RGB',(600,768),'white').save(im)
    rows=[dict(split='train',image_path='train/one.jpg',label=0,content_group='g1',selected=True,file_sha256=sha(im)),
          dict(split='val',image_path='train/two.jpg',label=1,content_group='g2',selected=False,file_sha256='not-loaded')]
    pixels=scan/'pixels.jsonl';pixels.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary=dict(complete=True,pixels_sha256=sha(pixels),splits={'train_dev.csv':'abc'},rule=dict(id='test',review_seed=5,review_limit=16))
    (scan/'scan_summary.json').write_text(json.dumps(summary))
    evidence=tmp/'sample_evidence.csv.gz'
    with gzip.open(evidence,'wt') as f:
        writer=csv.DictWriter(f,fieldnames=['split','image_path','original_label','content_group','center_label_probability','l05_prediction','l05_fixed_decode_prediction'])
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(split=r['split'],image_path=r['image_path'],original_label=r['label'],content_group=r['content_group'],center_label_probability=.25,l05_prediction=0,l05_fixed_decode_prediction=0 if r['split']=='val' else ''))
    ref=tmp/'p0.json';ref.write_text(json.dumps(dict(archived_artifacts={evidence.name:sha(evidence)},identity=dict(num_classes=2,training_samples=1,validation_samples=1,split_hashes={'train_dev.csv':'abc'}),baseline_errors=1)))
    out=tmp/'out'
    cmd=[sys.executable,str(ROOT/'scripts/p75_text_page_report.py'),'--scan',str(scan),'--evidence',str(evidence),'--p0-summary',str(ref),'--image-root',str(images),'--out',str(out)]
    return cmd,out,evidence

def test_join_baseline_and_gradient_statistics_and_crop(tmp_path):
    cmd,out,_=fixture(tmp_path)
    done=subprocess.run(cmd,capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    result=json.loads((out/'report.json').read_text())
    group=result['groups']['train_detected_text_proxy']
    assert group['mean_sqrt_py']==.5
    assert group['logit_gradient_l1_sum']==.75
    assert result['groups']['val_not_detected_proxy']['original_label_errors']==1
    assert result['emptied_classes']==[0]
    assert result['training_started'] is False
    assert (out/'review_01.jpg').exists()
    assert subprocess.run(cmd,capture_output=True).returncode!=0

def test_changed_input_fails_before_output(tmp_path):
    cmd,out,evidence=fixture(tmp_path)
    evidence.write_bytes(evidence.read_bytes()+b'changed')
    assert subprocess.run(cmd,capture_output=True).returncode!=0
    assert not out.exists()

"""Use the real CLI checker so its stderr success channel cannot regress."""
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('delivery_verifier',ROOT/'scripts/verify_v1_768_full_delivery.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def package(tmp_path,label):
    test=tmp_path/'test';test.mkdir()
    (test/'first.png').touch();(test/'second.jpg').touch()
    mapping=tmp_path/'class_mapping.json'
    mapping.write_text(json.dumps({'0000':0,'0001':1}))
    destination=tmp_path/'submission';destination.mkdir()
    csv=destination/'pred_results.csv'
    csv.write_text(f'first.png, 0000\nsecond.jpg, {label}\n')
    with zipfile.ZipFile(destination/'submission.zip','w') as archive:
        archive.write(csv,'pred_results.csv')
    return test,mapping,destination


def test_real_checker_stderr_success_is_accepted(tmp_path):
    test,mapping,destination=package(tmp_path,'0001')
    output=module.check_package(ROOT,test,mapping,destination)
    assert 'All checks passed' in output
    assert 'Line count: 2' in output


def test_real_checker_invalid_label_still_fails(tmp_path):
    test,mapping,destination=package(tmp_path,'0002')
    with pytest.raises(subprocess.CalledProcessError):
        module.check_package(ROOT,test,mapping,destination)

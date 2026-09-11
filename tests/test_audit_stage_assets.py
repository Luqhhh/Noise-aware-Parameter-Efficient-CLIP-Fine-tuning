import hashlib
import pytest
from scripts.audit_stage_assets import audit_assets


def test_missing_asset_never_inherits_document_hash(tmp_path):
    item=audit_assets([{'artifact_id':'missing','path':'absent.pt','sha256':'a'*64}],base_dir=tmp_path)[0]
    assert item['exists'] is False and item['sha256'] is None and item['byte_size'] is None
    assert item['status']=='missing'


def test_hashes_actual_bytes_and_preserves_distinct_influence_roles(tmp_path):
    path=tmp_path/'asset';path.write_bytes(b'')
    item=audit_assets([{'artifact_id':'present','path':'asset','sha256':'0'*64,'learned_from_group_set_id':'train','encoded_group_set_id':'train_and_val'}],base_dir=tmp_path)[0]
    assert item['sha256']==hashlib.sha256(b'').hexdigest() and item['byte_size']==0
    assert item['status']=='hash_mismatch'
    assert item['learned_from_group_set_id']=='train'
    assert item['encoded_group_set_id']=='train_and_val'


def test_known_file_without_producer_is_not_reproducible(tmp_path):
    (tmp_path/'asset').write_bytes(b'file')
    item=audit_assets([{'artifact_id':'a','path':'asset'}],base_dir=tmp_path)[0]
    assert item['status']=='missing_producer'
    assert item['stage'] is None


def test_duplicate_ids_and_remote_paths_refused(tmp_path):
    with pytest.raises(ValueError):audit_assets([{'artifact_id':'a'},{'artifact_id':'a'}],base_dir=tmp_path)
    with pytest.raises(ValueError):audit_assets([{'artifact_id':'a','path':'https://example.com/a'}],base_dir=tmp_path)

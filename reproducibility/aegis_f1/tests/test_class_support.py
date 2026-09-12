import pytest
from aegis_clip.class_support import summarize_support


def test_conflicts_absent_classes_and_overlap_are_visible():
    raw = [dict(image_path=p, label=c) for p,c in [('a',0),('b',1),('c',1)]]
    rows = summarize_support(raw, raw[:1], raw[1:], {'a':'same','b':'same','c':'other'}, ['zero','one','absent'])
    assert rows[0]['conflict_group_samples'] == rows[1]['conflict_group_samples'] == 1
    assert rows[1]['unique_content_groups'] == 2
    assert rows[1]['validation_groups_seen_by_current_fit_samples'] == 1
    assert rows[2]['validation_support'] == 0
    assert rows[2]['raw_accuracy'] is None
    assert all(r['frequency_segment'] is None for r in rows)


def test_subset_label_changes_and_duplicate_paths_rejected():
    row = dict(image_path='a',label=0)
    with pytest.raises(ValueError,match='Subset differs'):
        summarize_support([row],[dict(image_path='a',label=1)],[],{'a':'g'},['a','b'])
    with pytest.raises(ValueError,match='Duplicate'):
        summarize_support([row,row],[],[],{'a':'g'},['a'])

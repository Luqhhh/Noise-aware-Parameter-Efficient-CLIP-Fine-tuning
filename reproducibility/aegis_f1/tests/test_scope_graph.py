import copy
import pytest
from aegis_clip.lineage import LineageAuditError, audit_declared_scope_graph


def node(identity, parents=(), learned=(), selected=(), encoded=()):
    return dict(artifact_id=identity, parent_artifact_ids=list(parents),
                stage='preliminary', dataset_id='official', class_mapping_sha256='mapping',
                scope='development_fit', producer_record='bound-producer.json',
                learned_from_groups=list(learned), selected_using_groups=list(selected),
                encoded_groups=list(encoded))


def audit(nodes, **kwargs):
    return audit_declared_scope_graph(nodes, target_id='child', stage='preliminary',
        dataset_id='official', class_mapping_sha256='mapping', evaluation_groups={'val'},
        official_test_groups={'test'}, **kwargs)


def test_fixed_encoding_is_not_fitting():
    r = audit([node('child', encoded=['train', 'val', 'test'])])
    assert r['status'] == 'checks_passed'
    assert r['evaluation_encoded_groups'] == 1
    assert r['evaluation_learned_groups'] == 0
    assert not r['authorizes_formal_execution']
    assert not r['source_authenticity_verified']


def test_oof_name_does_not_hide_teacher_exposure():
    r = audit([node('teacher', learned=['val']), node('OOF', ['teacher']), node('child', ['OOF'])])
    assert r['status'] == 'blocked'
    assert r['evaluation_learned_groups'] == 1
    assert any(x['dependency_path'] == ['child', 'OOF', 'teacher'] for x in r['exposures'])


def test_selection_propagates_and_diamond_is_counted_once():
    r = audit([node('parent', selected=['val']), node('a', ['parent']),
               node('b', ['parent']), node('child', ['a', 'b'])])
    assert r['evaluation_selected_groups'] == 1
    assert len(r['visited_artifacts']) == 4
    assert not r['independent_under_declared_graph']


@pytest.mark.parametrize('field,value', [('stage','repechage'), ('dataset_id','old'),
    ('class_mapping_sha256','permuted'), ('scope','synthetic_dryrun'), ('scope','final_fit'), ('scope','unverified')])
def test_scope_and_mapping_mismatch(field, value):
    n = node('child'); n[field] = value
    assert audit([n])['status'] == 'blocked'


@pytest.mark.parametrize('field', ['producer_record','parent_artifact_ids','learned_from_groups',
                                  'selected_using_groups','encoded_groups'])
def test_missing_declaration_is_unknown_not_empty(field):
    n = node('child'); del n[field]
    r = audit([n]); assert r['status'] == 'blocked' and r['unknown']


def test_cycles_and_missing_parents_block():
    assert audit([node('child', ['parent']), node('parent', ['child'])])['errors']
    assert audit([node('child', ['absent'])])['unknown']
    with pytest.raises(LineageAuditError): audit([node('child'), node('child')])


def test_overlap_diagnostic_is_explicit_and_does_not_allow_test_fit():
    n = node('child', learned=['val']); n['scope'] = 'final_fit'
    r = audit([n], require_independent=False)
    assert r['status'] == 'checks_passed'
    assert r['evaluation_role'] == 'overlap_diagnostic'
    assert not r['independent_under_declared_graph']
    n['selected_using_groups'] = ['test']
    assert audit([n], require_independent=False)['status'] == 'blocked'


def test_audit_does_not_mutate_declarations():
    nodes = [node('child', ['parent']), node('parent', encoded=['val'])]
    before = copy.deepcopy(nodes)
    audit(nodes)
    assert nodes == before


def test_bound_scope_files_reject_tampering(tmp_path):
    import json
    from aegis_clip.cli.audit_scope_graph import audit_manifest
    from aegis_clip.runtime import sha256_file
    groups = tmp_path / 'groups.json'; groups.write_text(json.dumps(['val']))
    binding = {'path': groups.name, 'sha256': sha256_file(groups)}
    n = node('child'); n['encoded_groups'] = binding
    manifest = dict(schema_version=1, nodes=[n], target_id='child', stage='preliminary',
                    dataset_id='official', class_mapping_sha256='mapping',
                    evaluation_groups=binding, official_test_groups=binding,
                    evaluation_role='development_evaluation')
    source = tmp_path / 'manifest.json'; source.write_text(json.dumps(manifest))
    assert audit_manifest(source)['status'] == 'checks_passed'
    groups.write_text(json.dumps(['different']))
    with pytest.raises(LineageAuditError, match='hash mismatch'): audit_manifest(source)

import copy
import json
import pytest
from aegis_clip.cli.build_teacher_trust import relocated_feature_cache


def test_explicit_cache_relocation_preserves_checkpoint_and_rejects_order(tmp_path):
    (tmp_path/'old_paths.json').write_text('["a", "b"]')
    (tmp_path/'image_paths.json').write_text('["a", "b"]')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(stage='preliminary',backbone='ViT-B/32',
         pretrained='openai',augmentation='none',normalized=True,feature_dim=512)))
    cfg=dict(project={'stage':'preliminary'},model={'feature_dim':512},
             features={'paths_path':str(tmp_path/'old_paths.json'),'tensor_path':'old.pt'})
    before=copy.deepcopy(cfg)
    result=relocated_feature_cache(cfg,str(tmp_path))
    assert result['tensor_path']==str(tmp_path/'features.pt') and cfg==before
    assert relocated_feature_cache(cfg,None)==cfg['features']
    (tmp_path/'image_paths.json').write_text('["b", "a"]')
    with pytest.raises(ValueError,match='order'):relocated_feature_cache(cfg,str(tmp_path))
    cfg['project']['stage']='repechage'
    with pytest.raises(ValueError,match='protocol'):relocated_feature_cache(cfg,str(tmp_path))

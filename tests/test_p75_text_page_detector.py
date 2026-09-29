import importlib.util
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('detector', ROOT/'scripts/p75_text_page_detector.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CFG = json.loads((ROOT/'configs/p75_text_page_mask_20260929/detector.json').read_text())

def page():
    im = Image.new('RGB',(600,768),'white')
    d = ImageDraw.Draw(im)
    # Artificial regular glyphs exercise geometry without any learned text model.
    for y in range(30,710,20):
        for x in range(30,565,10): d.rectangle((x,y,x+4,y+8),fill='black')
    return im

def test_dense_page_and_blank():
    assert m.detect(page(),CFG)['selected']
    assert not m.detect(Image.new('RGB',(600,768),'white'),CFG)['selected']

def test_large_object_on_page_is_not_suppressed():
    im=page(); d=ImageDraw.Draw(im)
    d.rectangle((100,200,490,530),fill='white')
    d.ellipse((200,220,400,510),outline='black',width=6)
    assert not m.detect(im,CFG)['selected']

def test_colored_organ_and_nontext_drawing_are_retained():
    im=page(); ImageDraw.Draw(im).ellipse((200,220,400,510),fill='green')
    assert not m.detect(im,CFG)['selected']
    im=Image.new('RGB',(600,768),'white'); d=ImageDraw.Draw(im)
    for x in range(50,550,10): d.line((300,30,x,700),fill='black',width=2)
    assert not m.detect(im,CFG)['selected']

def test_no_label_or_confidence_inputs_and_deterministic():
    assert list(__import__('inspect').signature(m.detect).parameters)==['image','cfg']
    assert m.detect(page(),CFG)==m.detect(page(),CFG)

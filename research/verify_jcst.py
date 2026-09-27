"""Verify format conversion preserved measured tables and supplied journal assets."""
from pathlib import Path
import re
import json
import hashlib
import zipfile
from pypdf import PdfReader
from PIL import Image

ROOT=Path(__file__).resolve().parent
paper=ROOT/'paper_jcst'
old=(ROOT/'paper/main.tex').read_text(encoding='utf-8')
new=(paper/'main.tex').read_text(encoding='utf-8')
pattern=r'\\begin\{tabular\}.*?\\end\{tabular\}'
old_tables=re.findall(pattern,old,re.S)
new_tables=re.findall(pattern,new,re.S)
retained_old=[t for t in old_tables if 'NLL' not in t]
assert all(t in new_tables for t in retained_old)
assert len(new_tables)==21
assert not re.search(r'\bNLL\b|negative log likelihood|Supporting predictive checks',new,re.I)
assert not any('label{tab:'+k+'}' in new for k in ['countgrid','countperf','elometrics','performance','stageperf'])
for label in re.findall(r'\\label\{(tab:[^}]+)\}',new):
    assert 'ref{'+label+'}' in new,label
assert (paper/'jcst.cls').read_bytes()==(ROOT/'jcst_template/jcst.cls').read_bytes()
assert 'Overfull' not in (paper/'main.log').read_text(errors='replace')
assert 'undefined' not in (paper/'main.log').read_text(errors='replace').lower()
assert min(Image.open(paper/'stage_values.png').info['dpi'])>=299
pdf=ROOT.parent/'output/pdf/jcst_manuscript.pdf'
reader=PdfReader(pdf)
assert len(reader.pages)<=15
for page in reader.pages:
    assert abs(float(page.mediabox.width)-595.276)<1
    assert abs(float(page.mediabox.height)-841.89)<1
text='\n'.join(p.extract_text() for p in reader.pages)
for value in ['Jungbin Kim','rlawjdqls1212@unist.ac.kr','0009-0000-4719-1228',
              'no competing interests','no external funding','Resumen','Citation:']:
    assert value in text,value
abstracts=re.findall(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',new,re.S)
counts=[len(s.split()) for s in abstracts]
assert len(counts)==2 and max(counts)<200
with zipfile.ZipFile(ROOT.parent/'output/jcst_manuscript_source.zip') as z:
    assert z.testzip() is None
    assert {'main.tex','main.bbl','references.bib','jcst.cls','ieeetran.bst','stage_values.png'}<=set(z.namelist())
report=dict(pages=len(reader.pages),retained_original_tables_unchanged=True,removed_nll_tables=5,total_tables=len(new_tables),all_tables_referenced=True,
            official_class_unchanged=True,abstract_word_counts=counts,figure_dpi=300,
            overfull_boxes=0,undefined_references=0,source_zip_integrity=True,
            pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
(paper/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

"""Author-selected final operator; all new results have a separate namespace."""
import os
from pathlib import Path
from operator_scratch_20260927_v2.common import ROOT, DATA_ROOT, PACK, BANK, BANK_SHA, read,write,save,sha

OUT=DATA_ROOT/'runs/operator_final_20260930'
SOURCE=DATA_ROOT/'runs/joint_refinement_20260915/applications/baseline'
FACTOR=DATA_ROOT/'runs/history_matching_dense_20260910/prior_factor.npy'
PAPER=Path('/mnt/home/yliu5/projects/AVF-operator-final-20260930')
SELECTED=DATA_ROOT/'runs/operator_campaign_20260925/formal/r1024_s20260925'
EXPECTED_SHA='ed04348969fe27903bc9bd084781164f65c610c7e46da0ff75bee60553fdc18e'
PYTHON='/mnt/home/yliu5/.venvs/olgeo-inference-compare/bin/python'


def environment():
    os.environ.update(OLGEO_PACK=str(PACK),OLGEO_SCREEN='0',OLGEO_GUARD='0',OLGEO_PAIRED='0')
    os.environ.pop('OLGEO_ABORT',None)


def candidate():
    m=read(OUT/'model_selection.json')
    assert m['author_selected_final_model'] and m['checkpoint_sha256']==EXPECTED_SHA
    assert sha(m['checkpoint'])==EXPECTED_SHA
    return m

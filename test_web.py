"""Boundary and UI checks for the revised research calculator."""
from pathlib import Path
import json
import math
import sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from inference import load_model,predict,validate_inputs,charge_af_5y,harms2_af,InputError


def example():return json.loads((HERE/'example.json').read_text())


def test_release_and_parity_audit():
    model=load_model()
    audit=json.loads((HERE/'verification.json').read_text())
    assert len(model['features'])==49 and len(model['fits'])==5
    assert model['source_model_sha256']==audit['source_model_sha256']
    assert audit['passed'] and audit['max_abs_error']<1e-10
    for v in audit['cohorts'].values():
        assert math.isfinite(v['frozen_ensemble_max_abs_error']) and v['frozen_ensemble_max_abs_error']<1e-10
        assert v['complete_input_participants']>0
        assert math.isfinite(v['complete_input_website_max_abs_error']) and v['complete_input_website_max_abs_error']<1e-10


@pytest.mark.parametrize('change',[{'prs':None},{'duration':float('nan')},{'age':float('inf')},{'fall':3},{'height':-1}])
def test_invalid_inputs_are_not_imputed(change):
    with pytest.raises(InputError):predict(load_model(),{**example(),**change})


def test_risk_is_bounded_and_monotone():
    risk=predict(load_model(),example())
    assert risk[0]==0 and all(0<=r<=1 for r in risk)
    assert all(a<=b for a,b in zip(risk,risk[1:]))


def test_score_boundaries():
    base={'age':59.99,'bmi':29.99,'alcohol_units_week':6.99,'hypertension':0,'sex':0,'sleepapnoea':0,'ever_smoking':0}
    assert harms2_af(base)==0
    assert harms2_af({**base,'age':60,'bmi':30,'alcohol_units_week':7})==3
    assert harms2_af({**base,'age':65,'bmi':30,'alcohol_units_week':15,'hypertension':1,'sex':1,'sleepapnoea':1,'ever_smoking':1})==14


def test_charge_requires_current_smoking_and_treatment():
    x={'age':65,'height':1.75,'weight':75,'sbp':120,'dbp':80,'race':1,
       'current_smoking':0,'antihypertensive_use':0,'diabete':0,'heartfailer':0,'infarc':0}
    base=charge_af_5y(x)
    assert 0<base<1
    assert charge_af_5y({**x,'current_smoking':1})>base
    assert charge_af_5y({**x,'antihypertensive_use':1})>base
    del x['antihypertensive_use']
    with pytest.raises(InputError):charge_af_5y(x)


def test_streamlit_flow_and_stale_predictions():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(str(HERE/'app.py'),default_timeout=15).run()
    assert not at.exception and not at.metric
    at.button(key='calculate_main').click().run()
    assert at.error and not at.metric
    at.button(key='load_example').click().run()
    at.button(key='calculate_main').click().run()
    assert not at.exception and not at.error
    expected=predict(load_model(),example())
    for metric,year in zip(at.metric,[1,3,5,10]):
        assert metric.value==f"{expected[load_model()['times'].index(float(year))]:.2%}"
    at.number_input(key='m_age').set_value(66.).run()
    assert not at.metric and any('Inputs have changed' in x.value for x in at.warning)
    at.button(key='clear_inputs').click().run()
    assert not at.metric and at.number_input(key='m_age').value is None


def test_score_forms_reject_missing_and_compute_examples():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(str(HERE/'app.py'),default_timeout=15).run()
    buttons={b.label:b for b in at.button}
    buttons['Calculate HARMS₂-AF'].click().run()
    assert at.error and not at.metric
    for key,value in {'h_age':65.,'h_bmi':30.,'h_alcohol':15.}.items():at.number_input(key=key).set_value(value)
    for key in ['h_sex','h_hypertension','h_sleep','h_smoking']:at.selectbox(key=key).set_value(1)
    next(b for b in at.button if b.label=='Calculate HARMS₂-AF').click().run()
    assert not at.exception and at.metric[0].value=='14 / 14'
    for key,value in {'c_age':65.,'c_height':1.75,'c_weight':75.,'c_sbp':120.,'c_dbp':80.}.items():at.number_input(key=key).set_value(value)
    for key in ['c_race','c_smoking','c_treatment','c_diabetes','c_hf','c_mi']:at.selectbox(key=key).set_value(0)
    next(b for b in at.button if b.label=='Calculate CHARGE-AF').click().run()
    assert not at.exception and len(at.metric)==2

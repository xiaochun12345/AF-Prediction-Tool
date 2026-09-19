"""Export only fitted parameters; validate against all frozen validation predictions."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import joblib

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
SOURCE=PROJECT/'outputs/revision_v2'
sys.path.insert(0,str(PROJECT/'code/revision_v2'))
from feature_schema import metadata
from inference import predict, fit_risk


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def export():
    source=SOURCE/'models/main_ElasticNet.joblib'
    b=joblib.load(source)
    assert b['model']=='ElasticNet' and b['analysis']=='main' and b['m']==5
    assert len(b['features'])==49
    times=np.r_[0.,b['grid']]
    train=pd.read_csv(SOURCE/'data/cohort.csv',low_memory=False)
    train=train[train.split.eq('train')]
    schema=[metadata(c) for c in b['features']]
    for field in schema:
        values=train[field['name']].dropna()
        if field['type']=='number':
            field.update(training_min=float(values.min()),training_max=float(values.max()))
        else:
            assert set(values.unique()).issubset(set(field['choices']))
    compiled=[]
    for fit in b['fits']:
        ct=fit.transformer.named_steps['columntransformer']
        support=fit.transformer.named_steps['variancethreshold'].get_support()
        beta=np.zeros(len(support));beta[support]=fit.estimator.coef_.ravel()
        offset=float(fit.estimator.offset_[0]);terms=[];j=0
        for kind,processor,columns in ct.transformers_:
            if kind=='numeric':
                for c,mean,scale in zip(columns,processor.mean_,processor.scale_):
                    coefficient=float(beta[j]/scale);j+=1
                    offset+=coefficient*mean
                    terms.append({'feature':c,'coefficient':coefficient,'category':None})
            elif kind=='categorical':
                for c,categories,drop in zip(columns,processor.categories_,processor.drop_idx_):
                    for i,category in enumerate(categories):
                        if i==drop:continue
                        terms.append({'feature':c,'coefficient':float(beta[j]),'category':float(category)});j+=1
            else:raise ValueError('Unexpected transformer '+kind)
        assert j==len(beta)
        base=fit.estimator._baseline_models[0].cum_baseline_hazard_(b['grid'])
        compiled.append({'offset':offset,'terms':terms,'baseline_hazard':[0.]+base.tolist()})
    model={'format_version':2,'release':'revision_v2_2026-09-19','model':'Elastic Net Cox',
           'source_model_sha256':digest(source),'parameters':b['params'],'imputations':5,
           'features':schema,'times':times.tolist(),'report_years':[1,3,5,10], 'fits':compiled,
           'target':'Net AF risk, with competing death treated as censoring; not a competing-risk cumulative incidence.',
           'missing_input_policy':'All 49 input fields are required. No website imputation.',
           'validation':'UK Biobank internal and temporal validation; no independent external validation.'}
    example={f['name']:float(train[f['name']].mode().iloc[0] if f['type']=='category' else train[f['name']].median()) for f in schema}
    audit={'source_model_sha256':model['source_model_sha256'],'m':5,'input_features':len(schema),
           'tolerance':1e-10,'cohorts':{},'only_model_parameters_exported':True,
           'example_is_synthetic_marginal_medians_and_modes':True}
    original=pd.read_csv(SOURCE/'data/cohort.csv',low_memory=False)
    frozen=pd.read_csv(SOURCE/'results/predictions_main.csv')
    frozen=frozen[frozen.model.eq('main_ElasticNet')]
    max_error=0.
    for split in ['internal','temporal']:
        each=[];fit_errors=[]
        for k,fit in enumerate(b['fits']):
            d=pd.read_csv(SOURCE/'data'/f'main_{split}_imp{k}.csv')
            rows=d[b['features']].to_dict('records')
            actual=np.array([fit_risk(compiled[k],row)[1:] for row in rows])
            expected=fit.predict(d,b['grid'])[1]
            fit_errors.append(float(np.max(np.abs(actual-expected))))
            each.append(actual)
        saved=frozen[frozen.split.eq(split)].pivot(index='id',columns='time',values='risk')
        saved.columns=np.round(saved.columns.to_numpy(float),10)
        saved=saved.reindex(index=d.id,columns=np.round(b['grid'],10))
        assert np.isfinite(saved.to_numpy()).all(), 'Frozen prediction alignment failed'
        ensemble_error=float(np.max(np.abs(np.mean(each,axis=0)-saved.to_numpy())))
        complete=original[original.split.eq(split)].dropna(subset=b['features'])
        assert len(complete)>0
        actual=np.array([predict(model,row)[1:] for row in complete.to_dict('records')])
        expected=saved.reindex(index=complete.id).to_numpy()
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        complete_error=float(np.max(np.abs(actual-expected)))
        max_error=max(max_error,*fit_errors,ensemble_error,complete_error)
        audit['cohorts'][split]={'participants':len(d),'each_fit_max_abs_errors':fit_errors,
            'frozen_ensemble_max_abs_error':ensemble_error,'complete_input_participants':len(complete),
            'complete_input_website_max_abs_error':complete_error}
    all_errors=[e for v in audit['cohorts'].values() for e in v['each_fit_max_abs_errors']+[v['frozen_ensemble_max_abs_error'],v['complete_input_website_max_abs_error']]]
    if not np.isfinite(all_errors).all() or max_error>audit['tolerance']:raise ValueError('Website prediction differs from locked results')
    audit['max_abs_error']=max_error;audit['passed']=True
    (HERE/'model.json').write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'example.json').write_text(json.dumps(example,indent=2),encoding='utf-8')
    (HERE/'verification.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
    (SOURCE/'audit/web_prediction_parity.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':export()

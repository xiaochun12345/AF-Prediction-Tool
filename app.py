"""Revised COPD–AF calculator; preserves the sidebar and three-panel layout."""
import json
from pathlib import Path
import plotly.graph_objects as go
import streamlit as st
from inference import (load_model, predict, validate_inputs, outside_training_range,
                       charge_af_5y, harms2_af, InputError)

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title='COPD–AF Risk Prediction | Revised Model', page_icon='🫁', layout='wide')
st.markdown('''<style>
[data-testid="stMetric"] {padding:12px;border:1px solid #8883;border-radius:10px;}
[data-testid="stSidebar"] .stButton > button {width:100%;}
</style>''', unsafe_allow_html=True)


@st.cache_data
def resources():
    return (load_model(), json.loads((HERE/'example.json').read_text()),
            json.loads((HERE/'verification.json').read_text()))


try:
    model, example, verification = resources()
except (OSError, ValueError, KeyError):
    st.error('The revised model files could not be loaded. Please contact the study team.')
    st.stop()


def set_example():
    for field in model['features']:
        st.session_state['m_'+field['name']] = example[field['name']]
    st.session_state.pop('main_result', None)
    st.session_state['example_loaded'] = True


def clear_inputs():
    for field in model['features']:
        st.session_state['m_'+field['name']] = None
    st.session_state.pop('main_result', None)
    st.session_state['example_loaded'] = False


def field_widget(field):
    key='m_'+field['name']
    if field['type']=='category':
        return st.selectbox(field['label'],field['choices'],index=None,key=key,
            format_func=lambda v:field['choice_labels'][str(int(v))],help=field['help'],placeholder='Select an answer')
    label=field['label'] + (' ('+field['unit']+')' if field['unit'] else '')
    return st.number_input(label,min_value=field['minimum'],value=None,step=0.0001 if field['name']=='prs' else .01,
                           format='%.4f' if field['name']=='prs' else '%.2f',key=key,help=field['help'])


st.sidebar.title('🧬 Baseline data entry')
st.sidebar.caption('49 predictors · revised Elastic Net model')
st.sidebar.button('Load synthetic example',on_click=set_example,key='load_example')
st.sidebar.button('Clear all inputs',on_click=clear_inputs,key='clear_inputs')
if st.session_state.get('example_loaded'):
    st.sidebar.info('Synthetic example loaded. Replace the values for your own research case.')
values={}
with st.sidebar:
    for group in dict.fromkeys(field['group'] for field in model['features']):
        with st.expander(group,expanded=group=='Demographics & body measures'):
            if group=='COPD & medication use':st.caption('Medication use at baseline; recorded COPD history.')
            for field in model['features']:
                if field['group']==group:values[field['name']]=field_widget(field)
    calculate=st.button('Calculate revised AF risk',type='primary',key='calculate_main')
    completed=sum(v is not None for v in values.values())
    st.caption(f'{completed} / 49 fields completed. All inputs are required.')

st.title('🫁 AF Risk Prediction in COPD')
st.caption('Revised model · 19 September 2026 · UK Biobank internal and temporal validation')
st.info('Research calculator for participants with baseline COPD and no previous AF. '
        'The model estimates net AF risk with death treated as censoring. It does not estimate '
        'competing-risk cumulative incidence and has not been validated for clinical decisions.')

if calculate:
    st.session_state.pop('main_result',None)
    try:
        clean=validate_inputs(model,values)
        risks=predict(model,clean)
        st.session_state['main_result']={'inputs':clean,'risk':risks}
    except InputError as exc:
        st.error(str(exc))

result=st.session_state.get('main_result')
if result and result['inputs']!=values:
    st.warning('Inputs have changed. Calculate again to update the results.')
    result=None

if result:
    outside=outside_training_range(model,result['inputs'])
    if outside:st.warning('Outside the observed training range: '+', '.join(outside)+'. These predictions involve extrapolation.')
    col_graph,col_summary=st.columns([3,1])
    fig=go.Figure(go.Scatter(x=model['times'],y=result['risk'],mode='lines',
                            line=dict(color='#2ca02c',width=3),fill='tozeroy',
                            fillcolor='rgba(44,160,44,0.1)',name='Net AF risk',
                            hovertemplate='Year %{x:.2f}<br>Net AF risk %{y:.2%}<extra></extra>'))
    fig.update_layout(title='<b>Predicted net AF risk over time</b>',xaxis_title='Time from baseline (years)',
                       yaxis_title='Net AF risk',yaxis=dict(range=[0,1],tickformat='.0%'),
                       height=350,margin=dict(l=20,r=20,t=45,b=20),template='plotly_white',showlegend=False)
    with col_graph:st.plotly_chart(fig,width='stretch',key='main_curve')
    with col_summary:
        st.markdown('### Predicted risk')
        for year in model['report_years']:
            risk=result['risk'][model['times'].index(float(year))]
            st.metric(f'{year}-year net AF risk',f'{risk:.2%}')
    st.caption('Probabilities are averaged across five separately fitted Elastic Net models. '
               'No high / medium / low treatment categories are assigned.')
    download={'model_release':model['release'],'model_sha256':model['source_model_sha256'],
              'estimand':model['target'],'times_years':model['times'],'predicted_net_AF_risk':result['risk']}
    st.download_button('Download prediction results',json.dumps(download,indent=2),
                       file_name='copd_af_revision_v2_prediction.json',mime='application/json')
else:
    st.markdown('### Enter baseline measurements to calculate risk')
    st.write('Use the grouped inputs in the sidebar. You can load a synthetic example to explore the calculator. '
             'A prediction is shown only after all 49 fields have been completed and Calculate is selected.')

st.divider()
st.subheader('Published comparison scores')
st.caption('These panels use published formulas, without refitting their coefficients in this cohort.')
left,right=st.columns(2)


def number(label,key,min_value=0.,help=None):
    return st.number_input(label,min_value=min_value,value=None,key=key,help=help)


def binary(label,key):
    return st.selectbox(label,[0,1],index=None,format_func=lambda v:'Yes' if v else 'No',key=key)


with left:
    st.markdown('### 📉 HARMS₂-AF')
    with st.form('harms_form',border=True):
        hc1,hc2=st.columns(2)
        with hc1:
            hage=number('Age (years)','h_age')
            hbmi=number('BMI (kg/m²)','h_bmi')
            halcohol=number('Alcohol (UK units/week; 8 g/unit)','h_alcohol',help='Enter ethanol units, not the number of glasses. One UK unit contains 8 g ethanol.')
            hsex=st.selectbox('Sex',[0,1],index=None,format_func=lambda v:'Male' if v else 'Female',key='h_sex')
        with hc2:
            hhypertension=binary('Hypertension','h_hypertension')
            hsleep=binary('Sleep apnoea','h_sleep')
            hsmoke=binary('Current or previous smoking','h_smoking')
        hsubmit=st.form_submit_button('Calculate HARMS₂-AF')
    if hsubmit:
        st.session_state.pop('harms_result',None)
        try:
            st.session_state['harms_result']=harms2_af({'age':hage,'bmi':hbmi,'alcohol_units_week':halcohol,
                'hypertension':hhypertension,'sex':hsex,'sleepapnoea':hsleep,'ever_smoking':hsmoke})
        except InputError as exc:st.error(str(exc))
    if 'harms_result' in st.session_state:st.metric('HARMS₂-AF score — last submitted inputs',f"{st.session_state['harms_result']} / 14")
    st.caption('Score only. No absolute risk or survival curve is calculated because this analysis does not use a validated absolute-risk mapping.')

with right:
    st.markdown('### ⚡ CHARGE-AF')
    with st.form('charge_form',border=True):
        cc1,cc2=st.columns(2)
        with cc1:
            cage=number('Age (years)','c_age')
            cheight=number('Height (m)','c_height')
            cweight=number('Weight (kg)','c_weight')
            csbp=number('Systolic BP (mmHg)','c_sbp')
            cdbp=number('Diastolic BP (mmHg)','c_dbp')
            crace=st.selectbox('Ethnic background',[0,1],index=None,
                format_func=lambda v:'White' if v else 'Other / non-white',key='c_race')
        with cc2:
            csmoke=binary('Current smoking','c_smoking')
            ctreatment=binary('Antihypertensive treatment','c_treatment')
            cdiabetes=binary('Diabetes','c_diabetes')
            chf=binary('Heart failure','c_hf')
            cmi=binary('Myocardial infarction','c_mi')
        csubmit=st.form_submit_button('Calculate CHARGE-AF')
    if csubmit:
        st.session_state.pop('charge_result',None)
        try:
            st.session_state['charge_result']=charge_af_5y({'age':cage,'height':cheight,'weight':cweight,
                'sbp':csbp,'dbp':cdbp,'race':crace,'current_smoking':csmoke,'antihypertensive_use':ctreatment,
                'diabete':cdiabetes,'heartfailer':chf,'infarc':cmi})
        except InputError as exc:st.error(str(exc))
    if 'charge_result' in st.session_state:st.metric('5-year CHARGE-AF risk — last submitted inputs',f"{st.session_state['charge_result']:.2%}")
    st.caption('Published 5-year formula. Current smoking and antihypertensive treatment are separate inputs; hypertension diagnosis is not substituted for treatment.')

st.divider()
with st.expander('Model version, input definitions and reproducibility'):
    st.write('The main calculator uses the 49 predictors selected in the revised analysis, including recorded COPD duration, '
             '11 medication classes and the baseline inpatient J44.1 indicator. ICS and the three lung-function measures are not main-model inputs.')
    st.write('Five fitted models: alpha = 0.01; l1_ratio = 0.1. The deployment contains model coefficients and baseline hazards, '
             'without individual UK Biobank records or the training-data imputation kernel. Missing inputs are not replaced by typical values.')
    st.write('Inputs are processed by the Streamlit server for the current session. This application does not write input records to a database or file. '
             'Do not enter identifiers. The synthetic example is assembled from marginal training summaries and is not a participant record.')
    st.write(f"Parity checks passed for {sum(x['participants'] for x in verification['cohorts'].values())} validation participants across five completed datasets. "
             f"Maximum absolute probability difference: {verification['max_abs_error']:.2e}.")
    st.code(model['source_model_sha256'],language=None)
    st.markdown('[Source code](https://github.com/xiaochun12345/AF-Prediction-Tool) · '
                '[CHARGE-AF formula](https://pmc.ncbi.nlm.nih.gov/articles/PMC3647274/) · '
                '[HARMS₂-AF publication](https://academic.oup.com/eurheartj/article/44/36/3443/7205602)')
st.caption('Research use · No independent external validation · Model release '+model['release'])

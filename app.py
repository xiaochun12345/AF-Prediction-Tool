import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ==========================================
# 1. 页面配置与样式
# ==========================================
st.set_page_config(
    page_title="Multi-Model AF Risk Prediction",
    page_icon="💓",
    layout="wide"
)


st.markdown("""
<style>
    /* 浅色模式样式（原有逻辑保留） */
    @media (prefers-color-scheme: light) {
        .stMetric {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #eee;
            box-shadow: 1px 1px 3px #eee;
            color: #2c3e50 !important;
        }
        div[data-testid="stVerticalBlockBordered"] {
            background-color: #f8f9fa !important;
            border: 1px solid #eee !important;
            color: #2c3e50 !important;
        }
        h3 {
            color: #2c3e50;
        }
    }

    /* 深色模式样式（新增适配逻辑） */
    @media (prefers-color-scheme: dark) {
        .stMetric {
            background-color: #2d3748 !important; /* 深灰背景，适配深色模式 */
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #4a5568 !important; /* 深灰边框 */
            box-shadow: 1px 1px 3px #1a202c;
            color: #e2e8f0 !important; /* 浅灰文字，与深灰背景对比明显 */
        }
        div[data-testid="stVerticalBlockBordered"] {
            background-color: #2d3748 !important; /* 深灰背景，适配深色模式 */
            border: 1px solid #4a5568 !important; /* 深灰边框 */
            color: #e2e8f0 !important; /* 浅灰文字，清晰可见 */
        }
        h3 {
            color: #e2e8f0; /* 浅灰标题，适配深色模式 */
        }
        /* 确保输入框、下拉框在深色模式下清晰 */
        div[data-testid="stNumberInput"], div[data-testid="stSelectbox"] {
            background-color: #1a202c !important; /* 黑色背景，适配深色模式 */
            border: 1px solid #4a5568 !important;
            color: #e2e8f0 !important;
        }
    }
</style>
""", unsafe_allow_html=True)




# ==========================================
# 2. 模型加载逻辑
# ==========================================
@st.cache_resource
def load_models():
    models = {}
    try:
        # 1. 主模型
        models['main'] = joblib.load('coxnet_pipeline.pkl')
    except:
        models['main'] = None
    
    try:
        # 2. HARMS2-AF
        models['harms'] = joblib.load('harms_cox.pkl') 
    except:
        models['harms'] = None

    try:
        # 3. CHARGE-AF
        models['charge'] = joblib.load('charge_cox.pkl')
    except:
        models['charge'] = None
        
    return models

models = load_models()

if models['main'] is None:
    st.error("⚠️ Error: 'coxnet_pipeline.pkl' not found.")
    st.stop()

# ==========================================
# 3. 辅助函数：绘制生存曲线
# ==========================================


def plot_survival_curve(model, input_df, title_suffix="", line_color='#2ca02c'):
    try:
        # --- 自动列名对齐与检查 ---
        if hasattr(model, "feature_names_in_"):
            required_cols = list(model.feature_names_in_)
            try:
                input_df = input_df[required_cols]
            except KeyError as e:
                return None, 0.0, f"列名不匹配: {e}"

        # --- 预测 ---
        surv_funcs = model.predict_survival_function(input_df)
        fn = surv_funcs[0]
        
        if isinstance(fn, np.ndarray) or isinstance(fn, list):
            fn = fn[-1]
        
        if hasattr(fn, 'x'): 
            times = fn.x
            probs = fn.y  # 原始数据：无房颤（AF-Free）概率，时间0时=1
        elif hasattr(fn, 'index'): 
            times = fn.index.values
            probs = fn.values  # 原始数据：无房颤（AF-Free）概率，时间0时=1
        else:
            return None, 0.0, f"未知返回格式: {type(fn)}"
        
        # ==========================================
        # 关键修改1：将无房颤概率转换为房颤（AF）概率
        # ==========================================
        af_probs = 1 - probs  # 反转概率：房颤概率=1-无房颤概率，时间0时=0，随时间上升
        
        # 原有风险值计算逻辑不变（保证后续st.info展示的数值准确）
        final_prob = probs[-1]
        final_risk = (1 - final_prob) * 100
        
        # ==========================================
        # 关键修改2：绘制房颤概率曲线（使用af_probs替代probs）
        # ==========================================
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, 
            y=af_probs,  # 替换为房颤概率数据
            mode='lines', 
            line=dict(color=line_color, width=3),
            fill='tozeroy',  # 填充到y=0轴，符合房颤概率从0上升的视觉效果
            fillcolor=f'rgba{tuple(int(line_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}',
            name='AF Probability'  # 修改图例名称
        ))
        
        # ==========================================
        # 关键修改3：调整图表标题和坐标轴标签，明确展示房颤概率
        # ==========================================
        fig.update_layout(
            title=f"<b>Probability of Developing Atrial Fibrillation (AF) ({title_suffix})</b>",
            xaxis_title="Time (Years)",
            yaxis_title="AF Probability",  # y轴标签改为房颤概率
            yaxis_range=[0, 1.05],  # 保持y轴范围0-1.05，适配房颤概率（0到1）
            height=350, 
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_white",
            showlegend=False
        )
        
        return fig, final_risk, "Success"
    except Exception as e:
        return None, 0.0, str(e)





# ==========================================
# 4. PART A: 主模型输入 (侧边栏)
# ==========================================
st.sidebar.title("🧬 Patient Data Entry")
st.sidebar.info("Model: Penalized Cox Regression (Elastic Net)")

# --- 分组 1: 人口学 & 身体指标 ---
with st.sidebar.expander("👤 Demographics & Body Measures", expanded=True):
    # 数值变量
    age = st.number_input("Age", 30, 100, 65, key="m1_age")
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0, key="m1_bmi")
    sbp = st.number_input("Systolic Blood Pressure (mmHg)", 80, 250, 120, key="m1_sbp")
    weight = st.number_input("Weight(kg)", 30.0, 200.0, 75.0, key="m1_weight")
    height = st.number_input("Height(m)", 1.0, 2.5, 1.75, key="m1_height")
    
    # 分类变量 - 性别
    sex_map = {"Female": 0, "Male": 1}
    sex_label = st.selectbox("Sex", options=list(sex_map.keys()), index=1, key="m1_sex")
    sex = sex_map[sex_label]
    
    # 分类变量 - 种族
    race_map = {"Other": 0, "White": 1}
    race_label = st.selectbox("Race", options=list(race_map.keys()), index=1, key="m1_race")
    race = race_map[race_label]

# --- 分组 2: 生活方式 & 活动 ---
with st.sidebar.expander("🏃 Lifestyle & Activity", expanded=False):
    # 分类变量 - 睡眠呼吸暂停
    sleepapnoea_map = {"No": 0, "Yes": 1}
    sleepapnoea = sleepapnoea_map[st.selectbox("Sleep Apnoea", options=list(sleepapnoea_map.keys()), key="m1_sleepap")]
    
    # 分类变量 - 体重减轻
    lossweight_map = {"No": 0, "Yes": 1}
    lossweight = lossweight_map[st.selectbox("Loss weight (vs 1yr ago)", options=list(lossweight_map.keys()), key="m1_lossw")]
    
    # 分类变量 - 跌倒
    fall_map = {"No": 0, "Yes": 1}
    fall = fall_map[st.selectbox("Fall history", options=list(fall_map.keys()), key="m1_fall")]
    
    # 数值变量 - 饮酒量（每周杯数，8g/杯）
    drinkcup = st.number_input("Drinks/week (8g per drink)", 0, 100, 5, key="m1_drink")
    
    # 分类变量 - 吸烟
    smoke_map = {"No": 0, "Yes": 1}
    smoke = smoke_map[st.selectbox("Smoke", options=list(smoke_map.keys()), key="m1_smoke")]
    
    # 分类变量 - 昼夜节律
    chronotype_map = {"Morning type": 0, "Evening type": 1}
    chronotype = chronotype_map[st.selectbox("Chronotype", options=list(chronotype_map.keys()), index=0, key="m1_chrono")]
    
    # 分类变量 - 失眠
    insomnia_map = {"No": 0, "Yes": 1}
    insomnia = insomnia_map[st.selectbox("Insomnia", options=list(insomnia_map.keys()), key="m1_insomnia")]

# --- 分组 3: 病史 & 用药 ---
with st.sidebar.expander("🏥 Medical History & Meds", expanded=False):
    # 数值变量 - 用药数量
    num_meds = st.number_input("Number of treatments/medications", 0, 50, 0, key="m1_meds")
    
    # 分类变量（通用Yes/No映射）
    binary_map = {"No": 0, "Yes": 1}
    
    disability = binary_map[st.selectbox("Long-standing illness/disability", list(binary_map.keys()), key="m1_dis")]
    blood_clot_leg = binary_map[st.selectbox("Blood clot in leg", list(binary_map.keys()), key="m1_bclot_leg")]
    hayfever = binary_map[st.selectbox("Hayfever/allergic rhinitis/eczema", list(binary_map.keys()), key="m1_hayfever")]
    angina = binary_map[st.selectbox("Angina pectoris", list(binary_map.keys()), key="m1_ang")]
    hypertension = binary_map[st.selectbox("Hypertension", list(binary_map.keys()), key="m1_hyp")]
    diabete = binary_map[st.selectbox("Diabetes", list(binary_map.keys()), key="m1_diab")]
    heartfailer = binary_map[st.selectbox("Heart Failure", list(binary_map.keys()), key="m1_hf")]
    stroke = binary_map[st.selectbox("Stroke", list(binary_map.keys()), key="m1_stroke")]
    infarc = binary_map[st.selectbox("Myocardial infarction", list(binary_map.keys()), key="m1_inf")]

# --- 分组 4: 生化指标 ---
with st.sidebar.expander("🧪 Biomarkers & Lab Tests", expanded=False):
    # 分两列布局，保持界面整洁
    c1, c2 = st.columns(2) 
    with c1:
        Albumin = st.number_input("Albumin (g/L)", 10.0, 80.0, 40.0, key="m1_alb")
        Alkaline_phosphatase = st.number_input("Alkaline phosphatase (U/L)", 0.0, 800.0, 80.0, key="m1_alp")
        Apolipoprotein_B = st.number_input("Apo B (g/L)", 0.0, 5.0, 1.0, key="m1_apb")
        C_reactive_protein = st.number_input("C-reactive protein (mg/L)", 0.0, 100.0, 5.0, key="m1_crp")
        Cholesterol = st.number_input("Cholesterol (mmol/L)", 0.0, 15.0, 5.0, key="m1_chol")
        Creatinine = st.number_input("Creatinine (μmol/L)", 0.0, 500.0, 70.0, key="m1_cre")
        Cystatin_C = st.number_input("Cystatin C (mg/L)", 0.0, 8.0, 0.9, key="m1_cys")
        Gamma_glutamyltransferase = st.number_input("Gamma-glutamyltransferase (U/L)", 0.0, 500.0, 50.0, key="m1_ggt")
    with c2:
        Glucose = st.number_input("Glucose (mmol/L)", 2.0, 35.0, 5.0, key="m1_glu")
        Glycated_haemoglobin = st.number_input("Glycated haemoglobin (%)", 3.0, 20.0, 5.7, key="m1_hba1c")
        LDL_direct = st.number_input("LDL direct (mmol/L)", 0.0, 10.0, 2.5, key="m1_ldl")
        SHBG = st.number_input("SHBG (nmol/L)", 0.0, 500.0, 50.0, key="m1_shbg")
        Total_bilirubin = st.number_input("Total bilirubin (μmol/L)", 0.0, 200.0, 15.0, key="m1_bil")
        Urate = st.number_input("Urate (mmol/L)", 0.0, 10.0, 3.5, key="m1_urate")
        Urea = st.number_input("Urea (mmol/L)", 0.0, 50.0, 5.0, key="m1_ure")
        prs = st.number_input("AF PRS", -5.0, 5.0, 0.0, key="m1_prs")

# 组装主模型数据（完全匹配新gbmvar列表）
input_data_main = {
    'age': age, 'bmi': bmi, 'sbp': sbp, 'weight': weight, 'sleepapnoea': sleepapnoea,
    'numoftreatments_medications': num_meds, 'height': height, 'lossweight': lossweight,
    'fall': fall, 'disability': disability, 'Blood_clot_in_the_leg': blood_clot_leg,
    'Hayfever_allergic_rhinitis_eczema': hayfever, 'Angina': angina, 'hypertension': hypertension,
    'drinkcup': drinkcup, 'diabete': diabete, 'smoke': smoke, 'race': race, 'sex': sex,
    'chronotype': chronotype, 'insomnia': insomnia, 'Albumin': Albumin,
    'Alkaline_phosphatase': Alkaline_phosphatase, 'Apolipoprotein_B': Apolipoprotein_B,
    'C_reactive_protein': C_reactive_protein, 'Cholesterol': Cholesterol,
    'Creatinine': Creatinine, 'Cystatin_C': Cystatin_C, 'Gamma_glutamyltransferase': Gamma_glutamyltransferase,
    'Glucose': Glucose, 'Glycated_haemoglobin': Glycated_haemoglobin, 'LDL_direct': LDL_direct,
    'SHBG': SHBG, 'Total_bilirubin': Total_bilirubin, 'Urate': Urate, 'Urea': Urea,
    'prs': prs, 'heartfailer': heartfailer, 'stroke': stroke, 'infarc': infarc
}
df_main = pd.DataFrame([input_data_main])
# 强制重排序（完全复制新的gbmvar列表，保证和模型训练时一致）
train_cols_order =['age','bmi','sbp','weight','sleepapnoea','numoftreatments_medications',
      'height','lossweight','fall','disability','Blood_clot_in_the_leg','Hayfever_allergic_rhinitis_eczema',
      'Angina','hypertension','drinkcup','diabete','smoke','race','sex','chronotype',
      'insomnia','Albumin','Alkaline_phosphatase','Apolipoprotein_B','C_reactive_protein',
      'Cholesterol','Creatinine','Cystatin_C','Gamma_glutamyltransferase',
      'Glucose','Glycated_haemoglobin','LDL_direct','SHBG','Total_bilirubin',
      'Urate','Urea','prs','heartfailer','stroke','infarc']
try:
    df_main = df_main[train_cols_order]
except KeyError as e:
    st.warning(f"列名匹配警告: {e}")  # 新增警告，方便调试
    pass

# ==========================================
# 5. PART B: 主页面展示 (Main Results)
# ==========================================
st.title("🫁 AF Risk Prediction (Elastic Net)")
st.markdown("---")

col_main_graph, col_main_metrics = st.columns([3, 1])

# 运行主模型预测
fig_main, risk_main, msg_main = plot_survival_curve(models['main'], df_main, "Main")

if fig_main:
    # 动态颜色逻辑
    if risk_main > 30:
        main_level = "High Risk"
        main_color = '#d62728' # Red
        delta_col = "inverse"
    elif risk_main > 10:
        main_level = "Moderate Risk"
        main_color = '#ff7f0e' # Orange
        delta_col = "off"
    else:
        main_level = "Low Risk"
        main_color = '#2ca02c' # Green
        delta_col = "normal"

    fig_main.update_traces(line=dict(color=main_color), fillcolor=f'rgba{tuple(int(main_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}')

    with col_main_metrics:
        st.markdown("### 🔍 Analysis")
        st.metric("Cumulative AF Risk", f"{risk_main:.1f}%", main_level, delta_color=delta_col)
        st.metric("AF-Free Probability", f"{(100-risk_main)/100:.1%}")

    with col_main_graph:
        st.plotly_chart(fig_main, use_container_width=True)
else:
    st.error(f"Main Model Error: {msg_main}")

st.markdown("---")

# ==========================================
# 6. PART C: 下方两个模型 (Split Layout)
# ==========================================
col_left, col_right = st.columns(2)

# >>>>> HARMS2-AF (左下) <<<<<
with col_left:
    st.header("📉 HARMS2-AF Prediction")
    st.markdown("#### 1. Calculate Score")
    
    with st.container(border=True):
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            h_age = st.number_input("Age", 30, 100, 65, key="h_age")
            h_bmi = st.number_input("BMI", 10.0, 60.0, 25.0, key="h_bmi")
            h_drinks = st.number_input("Drinks/week (8g)", 0, 100, 5, key="h_drinks")
            h_sex = st.selectbox("Sex", ["Female", "Male"], key="h_sex")
        with h_col2:
            h_hyper = st.selectbox("Hypertension", ["No", "Yes"], key="h_hyper")
            h_sleep = st.selectbox("Sleep Apnoea", ["No", "Yes"], key="h_sleep")
            h_smoke = st.selectbox("Ever Smoke", ["No", "Yes"], key="h_smoke")
        
        # Values
        v_h_hyper = 1 if h_hyper == "Yes" else 0
        v_h_sex = 1 if h_sex == "Male" else 0
        v_h_sleep = 1 if h_sleep == "Yes" else 0
        v_h_smoke = 1 if h_smoke == "Yes" else 0
        
        # Score Calc
        h_score = 0
        if v_h_hyper: h_score += 4
        if h_age >= 65: h_score += 2
        elif h_age >= 60: h_score += 1
        if h_bmi >= 30: h_score += 1
        if v_h_sex: h_score += 2
        if v_h_sleep: h_score += 2
        if v_h_smoke: h_score += 1
        if h_drinks >= 15: h_score += 2
        elif h_drinks >= 7: h_score += 1
        
        # ★★★ 修复点：添加分割线，确保评分展示清晰可见 ★★★
        st.divider()
        st.metric("HARMS2-AF Score", h_score)

    st.markdown("#### 2. Predicted Survival Curve")
    if models['harms']:
        df_harms = pd.DataFrame([{
            'hypertension': v_h_hyper,
            'age': h_age,
            'bmi': h_bmi,
            'sex': v_h_sex,
            'sleepapnoea': v_h_sleep,
            'smoke': v_h_smoke,
            'drinkcup': h_drinks
        }])
        
        fig_h, risk_h, msg_h = plot_survival_curve(models['harms'], df_harms, "HARMS2", line_color='#1f77b4')
        
        if fig_h:
            st.plotly_chart(fig_h, use_container_width=True)
            st.info(f"HARMS2 Model - Cumulative Risk: **{risk_h:.1f}%**")
        else:
            st.error(f"HARMS Chart Error: {msg_h}")
    else:
        st.warning("⚠️ 'harms_cox.pkl' not found.")

# >>>>> CHARGE-AF (右下) <<<<<
with col_right:
    st.header("⚡ CHARGE-AF Prediction")
    st.markdown("#### 1. Calculate Score")
    
    with st.container(border=True):
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            c_age = st.number_input("Age", 30, 100, 65, key="c_age")
            c_height = st.number_input("Height (m)", 1.0, 2.5, 1.75, key="c_height")
            c_weight = st.number_input("Weight (kg)", 30.0, 200.0, 75.0, key="c_weight")
            c_race = st.selectbox("Race", ["Other", "White"], index=1, key="c_race")
            c_sbp = st.number_input("Systolic BP", 80, 250, 120, key="c_sbp")
            c_dbp = st.number_input("Diastolic BP", 40, 150, 80, key="c_dbp")
        with c_col2:
            c_smoke = st.selectbox("Smoking (Current)", ["No", "Yes"], key="c_smoke")
            c_hyper = st.selectbox("Hypertension", ["No", "Yes"], key="c_hyper")
            c_diab = st.selectbox("Diabetes", ["No", "Yes"], key="c_diab")
            c_hf = st.selectbox("Heart Failure", ["No", "Yes"], key="c_hf")
            c_mi = st.selectbox("Myocardial Infarction", ["No", "Yes"], key="c_mi")

        # Values
        v_c_race = 1 if c_race == "White" else 0
        v_c_smoke = 1 if c_smoke == "Yes" else 0
        v_c_hyper = 1 if c_hyper == "Yes" else 0
        v_c_diab = 1 if c_diab == "Yes" else 0
        v_c_hf = 1 if c_hf == "Yes" else 0
        v_c_mi = 1 if c_mi == "Yes" else 0

        # Score Calc
        c_score = (c_age/5*0.508) + (v_c_race*0.465) + (c_height*10*0.248) + \
                  (c_weight/15*0.115) + (c_sbp/20*0.197) - (c_dbp/10*0.101) + \
                  (v_c_smoke*0.359) + (v_c_hyper*0.349) + (v_c_diab*0.237) + \
                  (v_c_hf*0.701) + (v_c_mi*0.496)

        # ★★★ 修复点：添加分割线，确保评分展示清晰可见 ★★★
        st.divider()
        st.metric("CHARGE-AF Score", f"{c_score:.3f}")

    st.markdown("#### 2. Predicted Survival Curve")
    if models['charge']:
        df_charge = pd.DataFrame([{
            'age': c_age,
            'race': v_c_race,
            'height': c_height,
            'weight': c_weight,
            'sbp': c_sbp, 
            'dbp': c_dbp, 
            'smoke': v_c_smoke,
            'hypertension': v_c_hyper,
            'diabete': v_c_diab, 
            'heartfailer': v_c_hf, 
            'infarc': v_c_mi 
        }])
        
        fig_c, risk_c, msg_c = plot_survival_curve(models['charge'], df_charge, "CHARGE", line_color='#ff7f0e')
        
        if fig_c:
            st.plotly_chart(fig_c, use_container_width=True)
            st.info(f"CHARGE Model - Cumulative Risk: **{risk_c:.1f}%**")
        else:
            st.error(f"CHARGE Chart Error: {msg_c}")
    else:
        st.warning("⚠️ 'charge_cox.pkl' not found.")
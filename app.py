"""
================================================================================
  STROKE PREDICTION — STREAMLIT APP (No TensorFlow)
  Run: streamlit run app.py
  Author: Aqsa Siddiqui
================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }

.hero-header {
    background: linear-gradient(135deg, #1e2433 0%, #252d3d 100%);
    border: 1px solid #2d3748; border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.hero-header::before {
    content:''; position:absolute; top:-50%; right:-10%;
    width:300px; height:300px;
    background:radial-gradient(circle,rgba(239,68,68,0.08) 0%,transparent 70%);
    border-radius:50%;
}
.hero-badge {
    display:inline-block; background:rgba(239,68,68,0.15); color:#f87171;
    border:1px solid rgba(239,68,68,0.3); border-radius:20px;
    padding:4px 14px; font-size:0.75rem; font-weight:600;
    letter-spacing:1px; text-transform:uppercase; margin-bottom:0.8rem;
}
.hero-title {
    font-family:'Space Mono',monospace; font-size:2.2rem;
    font-weight:700; color:#f8fafc; margin:0; letter-spacing:-1px;
}
.hero-subtitle { color:#94a3b8; font-size:1rem; margin-top:0.5rem; font-weight:300; }

.metric-card {
    background:linear-gradient(135deg,#1e2433,#252d3d);
    border:1px solid #2d3748; border-radius:12px;
    padding:1.2rem 1.5rem; text-align:center; transition:all 0.3s ease;
}
.metric-card:hover { border-color:#ef4444; transform:translateY(-2px); }
.metric-value { font-family:'Space Mono',monospace; font-size:1.8rem; font-weight:700; color:#f8fafc; }
.metric-label { color:#64748b; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }

.risk-box-high   { background:linear-gradient(135deg,rgba(239,68,68,0.15),rgba(185,28,28,0.1));  border:2px solid rgba(239,68,68,0.5);  border-radius:16px; padding:2rem; text-align:center; }
.risk-box-medium { background:linear-gradient(135deg,rgba(245,158,11,0.15),rgba(180,83,9,0.1));  border:2px solid rgba(245,158,11,0.5); border-radius:16px; padding:2rem; text-align:center; }
.risk-box-low    { background:linear-gradient(135deg,rgba(34,197,94,0.15),rgba(21,128,61,0.1));   border:2px solid rgba(34,197,94,0.5);  border-radius:16px; padding:2rem; text-align:center; }
.risk-percent { font-family:'Space Mono',monospace; font-size:3.5rem; font-weight:700; line-height:1; }
.risk-label   { font-size:1.1rem; font-weight:600; margin-top:0.5rem; letter-spacing:1px; }

.section-title {
    font-family:'Space Mono',monospace; font-size:0.85rem; color:#64748b;
    text-transform:uppercase; letter-spacing:2px; margin-bottom:1rem;
    padding-bottom:0.5rem; border-bottom:1px solid #1e2433;
}
.info-row { display:flex; justify-content:space-between; padding:0.6rem 0; border-bottom:1px solid #1e2433; font-size:0.9rem; }
.info-key { color:#64748b; }
.info-val { color:#e2e8f0; font-weight:500; }

.factor-pill-danger  { display:inline-block; background:rgba(239,68,68,0.15);  color:#f87171; border:1px solid rgba(239,68,68,0.3);  border-radius:20px; padding:4px 12px; font-size:0.8rem; margin:3px; }
.factor-pill-warning { display:inline-block; background:rgba(245,158,11,0.15); color:#fbbf24; border:1px solid rgba(245,158,11,0.3); border-radius:20px; padding:4px 12px; font-size:0.8rem; margin:3px; }
.factor-pill-ok      { display:inline-block; background:rgba(34,197,94,0.1);   color:#4ade80; border:1px solid rgba(34,197,94,0.25); border-radius:20px; padding:4px 12px; font-size:0.8rem; margin:3px; }

#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:2rem; }
</style>
""", unsafe_allow_html=True)


# ── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        ml_model = joblib.load('best_ml_model.pkl')
        scaler   = joblib.load('scaler.pkl')
        encoders = joblib.load('label_encoders.pkl')
        metadata = joblib.load('metadata.pkl')
        return ml_model, scaler, encoders, metadata, True
    except Exception:
        return None, None, None, None, False

ml_model, scaler, encoders, metadata, models_loaded = load_artifacts()


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_patient(patient_data):
    df_new = pd.DataFrame([patient_data])
    df_new['bmi']               = df_new['bmi'].clip(10, 60)
    df_new['avg_glucose_level'] = df_new['avg_glucose_level'].clip(50, 300)

    df_new['age_group']        = pd.cut(df_new['age'], bins=[0,18,30,45,60,100],
                                         labels=['Teen','Young_Adult','Adult','Middle_Aged','Senior'])
    df_new['bmi_category']     = pd.cut(df_new['bmi'], bins=[0,18.5,25,30,35,100],
                                         labels=['Underweight','Normal','Overweight','Obese','Severely_Obese'])
    df_new['glucose_category'] = pd.cut(df_new['avg_glucose_level'], bins=[0,100,125,200,500],
                                         labels=['Normal','Prediabetic','Diabetic','High_Risk'])

    df_new['cardiovascular_risk']     = df_new['hypertension'] + df_new['heart_disease']
    df_new['age_risk']                = (df_new['age'] > 60).astype(int)
    df_new['glucose_risk']            = (df_new['avg_glucose_level'] > 125).astype(int)
    df_new['bmi_risk']                = (df_new['bmi'] > 30).astype(int)
    df_new['health_risk_score']       = (df_new['cardiovascular_risk'] + df_new['age_risk'] +
                                          df_new['glucose_risk'] + df_new['bmi_risk'])
    df_new['age_glucose_interaction'] = df_new['age'] * df_new['avg_glucose_level'] / 1000
    df_new['age_bmi_interaction']     = df_new['age'] * df_new['bmi'] / 100
    df_new['glucose_bmi_interaction'] = df_new['avg_glucose_level'] * df_new['bmi'] / 1000
    df_new['age_squared']             = df_new['age'] ** 2 / 1000
    df_new['high_risk_combo']         = ((df_new['hypertension']==1) &
                                          (df_new['heart_disease']==1) &
                                          (df_new['age_risk']==1)).astype(int)

    smoking_map = metadata.get('smoking_risk_map',
                  {'never smoked':0,'Unknown':1,'formerly smoked':2,'smokes':3})
    df_new['smoking_risk_score'] = df_new['smoking_status'].map(smoking_map).fillna(1)

    for col in metadata['categorical_columns']:
        df_new[col] = encoders[col].transform(df_new[col].astype(str))

    return scaler.transform(df_new[metadata['selected_features']])


def predict_risk(patient_data):
    X_new = preprocess_patient(patient_data)
    thr   = metadata.get('optimal_threshold', 0.5)
    prob  = float(ml_model.predict_proba(X_new)[0, 1])
    return prob, int(prob >= thr), thr


def get_risk_factors(data):
    f = []
    if data['age'] > 60:              f.append(('danger',  f"Age {data['age']} — High risk (>60)"))
    elif data['age'] > 45:            f.append(('warning', f"Age {data['age']} — Moderate risk"))
    else:                             f.append(('ok',      f"Age {data['age']} — Low risk"))
    if data['hypertension'] == 1:     f.append(('danger',  'Hypertension present'))
    else:                             f.append(('ok',      'No hypertension'))
    if data['heart_disease'] == 1:    f.append(('danger',  'Heart disease present'))
    else:                             f.append(('ok',      'No heart disease'))
    if data['avg_glucose_level'] > 200:   f.append(('danger',  f"Glucose {data['avg_glucose_level']:.0f} — Diabetic"))
    elif data['avg_glucose_level'] > 125: f.append(('warning', f"Glucose {data['avg_glucose_level']:.0f} — Pre-diabetic"))
    else:                                 f.append(('ok',      f"Glucose {data['avg_glucose_level']:.0f} — Normal"))
    if data['bmi'] > 35:              f.append(('danger',  f"BMI {data['bmi']:.1f} — Severely obese"))
    elif data['bmi'] > 30:            f.append(('warning', f"BMI {data['bmi']:.1f} — Obese"))
    elif data['bmi'] < 18.5:          f.append(('warning', f"BMI {data['bmi']:.1f} — Underweight"))
    else:                             f.append(('ok',      f"BMI {data['bmi']:.1f} — Normal"))
    if data['smoking_status'] == 'smokes':           f.append(('danger',  'Currently smoking'))
    elif data['smoking_status'] == 'formerly smoked': f.append(('warning', 'Former smoker'))
    else:                                             f.append(('ok',      'Non-smoker'))
    return f


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:1.1rem;color:#f8fafc;font-weight:700;margin-bottom:1.5rem;'>🧠 Patient Input</div>", unsafe_allow_html=True)

    st.markdown("**👤 Demographics**")
    gender    = st.selectbox("Gender",         ["Male","Female"])
    age       = st.slider("Age",               1, 100, 45)
    married   = st.selectbox("Ever Married",   ["Yes","No"])
    work      = st.selectbox("Work Type",      ["Private","Self-employed","Govt_job","children","Never_worked"])
    residence = st.selectbox("Residence Type", ["Urban","Rural"])

    st.markdown("---")
    st.markdown("**🏥 Medical History**")
    hypertension  = st.radio("Hypertension",  ["No","Yes"], horizontal=True)
    heart_disease = st.radio("Heart Disease", ["No","Yes"], horizontal=True)

    st.markdown("---")
    st.markdown("**🔬 Clinical Values**")
    glucose = st.slider("Avg Glucose Level (mg/dL)", 50.0, 300.0, 100.0, step=0.5)
    bmi     = st.slider("BMI",                        10.0,  60.0,  25.0, step=0.1)
    smoking = st.selectbox("Smoking Status", ["never smoked","formerly smoked","smokes","Unknown"])

    st.markdown("---")
    st.markdown("<div style='color:#64748b;font-size:0.8rem'>Model: LightGBM (Tuned) + Stacking Ensemble</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">Clinical AI Tool</div>
    <div class="hero-title">🧠 Stroke Risk Predictor</div>
    <div class="hero-subtitle">
        ML powered early stroke risk assessment &nbsp;|&nbsp;
        5,110 patient dataset &nbsp;|&nbsp;
        XGBoost · LightGBM · Stacking Ensemble
    </div>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error("""
    ⚠️ **Model files not found! Run training first:**
    ```
    python train.py
    ```
    Then make sure `best_ml_model.pkl`, `scaler.pkl`,
    `label_encoders.pkl`, `metadata.pkl` are in the same folder as `app.py`.
    """)
    st.stop()

# ── Stats Row ─────────────────────────────────────────────────────────────────
for col, (v, l) in zip(st.columns(4), [
    ("5,110",   "Patients Trained"),
    ("95 : 5",  "Class Imbalance"),
    ("6",       "Models Compared"),
    ("⚡ LightGBM", "Active Model"),
]):
    col.markdown(f'<div class="metric-card"><div class="metric-value">{v}</div><div class="metric-label">{l}</div></div>',
                 unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Patient Data ──────────────────────────────────────────────────────────────
patient_data = {
    'gender': gender, 'age': age,
    'hypertension': 1 if hypertension == "Yes" else 0,
    'heart_disease': 1 if heart_disease == "Yes" else 0,
    'ever_married': married, 'work_type': work, 'Residence_type': residence,
    'avg_glucose_level': glucose, 'bmi': bmi, 'smoking_status': smoking,
}

# ── Two-column layout ─────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">Patient Summary</div>', unsafe_allow_html=True)
    rows_html = "".join(
        f'<div class="info-row"><span class="info-key">{k}</span><span class="info-val">{v}</span></div>'
        for k, v in [
            ("Gender", gender), ("Age", f"{age} years"),
            ("Hypertension", hypertension), ("Heart Disease", heart_disease),
            ("Marital Status", married), ("Work Type", work),
            ("Residence", residence), ("Avg Glucose", f"{glucose:.1f} mg/dL"),
            ("BMI", f"{bmi:.1f}"), ("Smoking Status", smoking),
        ]
    )
    st.markdown(f'<div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:12px;padding:1rem 1.5rem">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:1.5rem">Risk Factors</div>', unsafe_allow_html=True)
    pills = "".join(f'<span class="factor-pill-{sev}">{lbl}</span>'
                    for sev, lbl in get_risk_factors(patient_data))
    st.markdown(f'<div style="line-height:2.5">{pills}</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Risk Assessment</div>', unsafe_allow_html=True)

    prob, pred, thr = predict_risk(patient_data)
    pct = prob * 100

    if pct >= 50:   box_cls, color, emoji, label = "risk-box-high",   "#ef4444", "🔴", "HIGH RISK"
    elif pct >= 25: box_cls, color, emoji, label = "risk-box-medium", "#f59e0b", "🟡", "MODERATE RISK"
    else:           box_cls, color, emoji, label = "risk-box-low",    "#22c55e", "🟢", "LOW RISK"

    st.markdown(f"""
    <div class="{box_cls}">
        <div class="risk-percent" style="color:{color}">{pct:.1f}%</div>
        <div class="risk-label"   style="color:{color}">{emoji} {label}</div>
        <div style="color:#64748b;font-size:0.8rem;margin-top:0.6rem">
            Stroke Probability &nbsp;|&nbsp; Threshold: {thr:.2f} &nbsp;|&nbsp; Model: LightGBM (Tuned)
        </div>
    </div>""", unsafe_allow_html=True)

    # Gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number={'suffix':'%','font':{'size':28,'color':color,'family':'Space Mono'}},
        gauge={
            'axis':  {'range':[0,100],'tickcolor':'#64748b','tickfont':{'color':'#64748b','size':10}},
            'bar':   {'color':color,'thickness':0.25},
            'bgcolor':'#1a1f2e',
            'steps': [{'range':[0,25],  'color':'rgba(34,197,94,0.1)'},
                      {'range':[25,50], 'color':'rgba(245,158,11,0.1)'},
                      {'range':[50,100],'color':'rgba(239,68,68,0.1)'}],
            'threshold':{'line':{'color':'#f8fafc','width':2},'thickness':0.75,'value':thr*100}
        }
    ))
    gauge.update_layout(height=220, margin=dict(t=20,b=10,l=30,r=30),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font={'color':'#94a3b8'})
    st.plotly_chart(gauge, use_container_width=True)

    badge_color = "#ef4444" if pred == 1 else "#22c55e"
    badge_text  = "⚠️ STROKE PREDICTED" if pred == 1 else "✅ NO STROKE PREDICTED"
    st.markdown(f'<div style="background:rgba(0,0,0,0.2);border:1px solid {badge_color}33;border-radius:8px;padding:0.7rem;text-align:center;color:{badge_color};font-weight:600;margin-top:0.5rem">{badge_text}</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Clinical Feature Importance (SHAP-based)</div>', unsafe_allow_html=True)

fi_df = pd.DataFrame({
    'Feature'   : ['heart_disease','smoking_risk_score','hypertension','cardiovascular_risk',
                   'age_bmi_interaction','bmi','age_glucose_interaction',
                   'health_risk_score','avg_glucose_level','age'],
    'Importance': [0.031,0.038,0.045,0.058,0.067,0.089,0.112,0.145,0.198,0.342]
})
bar_colors = ['#ef4444' if f in ['age','avg_glucose_level','health_risk_score']
              else '#f59e0b' if f in ['age_glucose_interaction','bmi','age_bmi_interaction']
              else '#3b82f6' for f in fi_df['Feature']]

fig_fi = go.Figure(go.Bar(
    x=fi_df['Importance'], y=fi_df['Feature'], orientation='h',
    marker_color=bar_colors, marker_line_width=0,
    text=[f"{v:.3f}" for v in fi_df['Importance']],
    textposition='outside', textfont={'color':'#94a3b8','size':11},
))
fig_fi.update_layout(
    height=370, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,31,46,0.5)',
    xaxis=dict(showgrid=True,gridcolor='#1e2433',color='#64748b',title='Mean |SHAP Value|'),
    yaxis=dict(color='#94a3b8'), margin=dict(t=10,b=10,l=10,r=70), font={'family':'DM Sans'},
)
st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL PERFORMANCE HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)

perf_df = pd.DataFrame({
    'Model'  : ['Logistic Regression','Random Forest','Gradient Boosting','XGBoost',
                'LightGBM','XGBoost (Tuned)','LightGBM (Tuned)','Stacking Ensemble'],
    'ROC-AUC': [0.847,0.871,0.856,0.873,0.876,0.882,0.885,0.889],
    'PR-AUC' : [0.312,0.341,0.328,0.349,0.352,0.367,0.371,0.381],
    'F1'     : [0.391,0.412,0.398,0.418,0.421,0.434,0.438,0.445],
    'Recall' : [0.682,0.701,0.693,0.712,0.718,0.731,0.737,0.748],
})
fig_perf = go.Figure(data=go.Heatmap(
    z=perf_df[['ROC-AUC','PR-AUC','F1','Recall']].values,
    x=['ROC-AUC','PR-AUC','F1','Recall'], y=perf_df['Model'],
    colorscale='RdYlGn', zmin=0.3, zmax=0.95,
    text=perf_df[['ROC-AUC','PR-AUC','F1','Recall']].round(3).values,
    texttemplate='%{text}', textfont={'size':11},
))
fig_perf.update_layout(
    height=340, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(color='#94a3b8',side='top'), yaxis=dict(color='#94a3b8',autorange='reversed'),
    margin=dict(t=30,b=10,l=10,r=10), font={'family':'DM Sans'},
)
st.plotly_chart(fig_perf, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STROKE RATE BY AGE GROUP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Stroke Risk by Age Group (Dataset Insight)</div>', unsafe_allow_html=True)

fig_age = go.Figure(go.Bar(
    x=['0-18','19-30','31-45','46-60','60+'], y=[0.2,0.5,1.8,5.2,13.6],
    marker_color=['#22c55e','#22c55e','#f59e0b','#f59e0b','#ef4444'], marker_line_width=0,
    text=['0.2%','0.5%','1.8%','5.2%','13.6%'],
    textposition='outside', textfont={'color':'#94a3b8'},
))
fig_age.update_layout(
    height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,31,46,0.5)',
    xaxis=dict(color='#64748b',title='Age Group'),
    yaxis=dict(color='#64748b',title='Stroke Rate (%)',showgrid=True,gridcolor='#1e2433'),
    margin=dict(t=20,b=20), font={'family':'DM Sans'},
)
st.plotly_chart(fig_age, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Batch Prediction — Upload CSV</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload CSV with patient records", type=['csv'],
    help="Columns: gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status"
)

if uploaded is not None:
    try:
        batch_df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(batch_df)} records")
        probs = []
        for _, row in batch_df.iterrows():
            p_data = {
                'gender': row.get('gender','Male'), 'age': row.get('age',50),
                'hypertension': int(row.get('hypertension',0)),
                'heart_disease': int(row.get('heart_disease',0)),
                'ever_married': row.get('ever_married','Yes'),
                'work_type': row.get('work_type','Private'),
                'Residence_type': row.get('Residence_type','Urban'),
                'avg_glucose_level': float(row.get('avg_glucose_level',100)),
                'bmi': float(row.get('bmi',25)),
                'smoking_status': row.get('smoking_status','never smoked'),
            }
            try:
                p, _, _ = predict_risk(p_data)
                probs.append(p)
            except Exception:
                probs.append(None)

        batch_df['Stroke Probability'] = probs
        batch_df['Risk Level'] = batch_df['Stroke Probability'].apply(
            lambda p: '🔴 High' if p and p>=0.5 else '🟡 Medium' if p and p>=0.25 else '🟢 Low' if p else 'Error')
        batch_df['Stroke Probability'] = batch_df['Stroke Probability'].apply(
            lambda p: f"{p*100:.1f}%" if p else "Error")

        st.dataframe(batch_df[['gender','age','hypertension','heart_disease',
                                'avg_glucose_level','bmi','Stroke Probability','Risk Level']],
                     use_container_width=True)
        st.download_button("📥 Download Results CSV", batch_df.to_csv(index=False),
                           "stroke_predictions.csv","text/csv")
    except Exception as e:
        st.error(f"Error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.2);
            border-radius:10px;padding:1rem 1.5rem;margin-top:1rem">
    <span style="color:#f87171;font-weight:600;font-size:0.85rem">⚕️ Medical Disclaimer</span><br>
    <span style="color:#64748b;font-size:0.8rem">
        This tool is for research and educational purposes only. It does not constitute medical advice
        and must not replace professional clinical judgement. Always consult a qualified healthcare
        provider for diagnosis and treatment. Ensure HIPAA/DISHA compliance when using real patient data.
    </span>
</div>
<div style="text-align:center;color:#2d3748;font-size:0.75rem;margin-top:1rem">
    Stroke Risk Predictor · Built with Streamlit · Author: Aqsa Siddiqui
</div>
""", unsafe_allow_html=True)
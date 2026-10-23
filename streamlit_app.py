import streamlit as st
import requests
import pandas as pd
import os

st.set_page_config(
    page_title="Titanic AI | MLOps",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --bg: #07111f;
  --panel: #0d1b2d;
  --line: #20334a;
  --teal: #36e0c0;
  --muted: #92a7bd;
}
.stApp {
  background: radial-gradient(ellipse at top right,#102a40 0%,#07111f 48%);
  color: #f1f6fb;
  font-family: 'DM Sans', sans-serif;
}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif !important; color:#f4f8fc !important;}
[data-testid="stSidebar"] {
  background: #091624;
  border-right: 1px solid #20334a;
}
[data-testid="stSidebar"] * {color:#d8e5f2;}
.block-container {padding-top:2rem; max-width:1450px;}
.hero {
  background: linear-gradient(120deg,#10253a,#102f3d 75%,#123c43);
  border:1px solid #285064; border-radius:22px;
  padding:30px 34px; margin:10px 0 24px 0;
}
.eyebrow {color:#36e0c0; letter-spacing:3px; font-size:11px; font-weight:700;}
.hero-title {font-family:'Space Grotesk';font-size:37px;font-weight:700;line-height:1.12;margin:10px 0;}
.hero-copy {color:#b2c5d7;font-size:14px;line-height:1.7;max-width:700px;}
.metric {
  background:linear-gradient(145deg,#102238,#0b1929);
  border:1px solid #20364c;border-radius:16px;padding:19px 20px;
  min-height:110px;
}
.metric-label {font-size:11px;color:#91a8bf;text-transform:uppercase;letter-spacing:1.4px;}
.metric-value {font-family:'Space Grotesk';font-size:25px;font-weight:700;margin-top:10px;color:#f3f8fc;}
.metric-note {font-size:11px;color:#36e0c0;margin-top:4px;}
.section-card {
  background:#0b1929;border:1px solid #20364c;
  border-radius:18px;padding:22px 24px;margin-bottom:16px;
}
.small-label {font-size:11px;letter-spacing:1.5px;color:#36e0c0;font-weight:700;}
.result-survive {
  background:linear-gradient(135deg,#103a38,#102b35);
  border:1px solid #2d8c7e;border-radius:18px;padding:24px;
}
.result-not {
  background:linear-gradient(135deg,#3a222b,#251b2a);
  border:1px solid #815064;border-radius:18px;padding:24px;
}
div.stButton > button {
  background:#36e0c0;color:#061521;border:0;border-radius:10px;
  min-height:48px;font-weight:700;width:100%;
}
div.stButton > button:hover {background:#70f2d9;color:#061521;border:0;}
.stCaption, [data-testid="stCaptionContainer"] {color:#91a8bf !important;}
hr {border-color:#20334a;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class='eyebrow'>MLOPS • PROJECT 04</div>", unsafe_allow_html=True)
    st.markdown("## ⚓ TITANIC AI")
    st.caption("Model deployment & inference platform")
    st.divider()
    page = st.radio("NAVIGATION", [
        "Prediction Studio", "Batch Results", "System Overview"
    ])
    st.divider()
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        online = health.ok
    except Exception:
        online = False
    st.markdown("### API STATUS")
    if online:
        st.success("● Backend connected")
    else:
        st.error("● Backend unavailable")
        st.caption("Start the FastAPI server on port 8000.")
    st.caption("Random Forest · Scikit-learn")
    st.caption("Academic demonstration only")

st.markdown("""
<div class="hero">
  <div class="eyebrow">INTELLIGENCE • INFERENCE • DEPLOYMENT</div>
  <div class="hero-title">Titanic Survival<br>Prediction Studio</div>
  <div class="hero-copy">
    An end-to-end machine learning deployment demonstration.
    Explore passenger attributes, submit real-time inference requests,
    and inspect the model-serving workflow through a dedicated interface.
  </div>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
for col, label, value, note in [
    (c1,"MODEL TYPE","Random Forest","Classification pipeline"),
    (c2,"INFERENCE","Real-time","FastAPI REST endpoint"),
    (c3,"VALIDATION","Pydantic","Typed request validation"),
    (c4,"DATA SOURCE","Kaggle","Titanic train.csv")
]:
    with col:
        st.markdown(f"""<div class="metric">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div></div>""",unsafe_allow_html=True)

st.write("")

if page == "Prediction Studio":
    left,right = st.columns([1.05,0.95],gap="large")
    with left:
        st.markdown("<div class='section-card'>",unsafe_allow_html=True)
        st.markdown("### Passenger profile")
        st.caption("Enter the passenger attributes to request a prediction.")
        with st.form("prediction_form"):
            a,b = st.columns(2)
            with a:
                pclass = st.selectbox("Passenger class", [1,2,3], index=2)
                sex = st.selectbox("Sex", ["female","male"])
                age = st.number_input("Age", min_value=0.0,max_value=100.0,value=22.0,step=1.0)
                sibsp = st.number_input("Siblings / spouses aboard",0,20,1)
            with b:
                parch = st.number_input("Parents / children aboard",0,20,0)
                fare = st.number_input("Ticket fare",min_value=0.0,value=7.25,step=1.0)
                embarked = st.selectbox("Port of embarkation",["S","C","Q"])
                st.markdown("<br>",unsafe_allow_html=True)
            submitted = st.form_submit_button("✦  RUN SURVIVAL INFERENCE")
        st.markdown("</div>",unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-card'>",unsafe_allow_html=True)
        st.markdown("### Prediction output")
        st.caption("Results from the saved model served by FastAPI.")
        if submitted:
            payload = {
                "Pclass":pclass,"Sex":sex,"Age":age,"SibSp":sibsp,
                "Parch":parch,"Fare":fare,"Embarked":embarked
            }
            try:
                response = requests.post(f"{API_URL}/predict",json=payload,timeout=15)
                if response.ok:
                    result = response.json()
                    survived = result["predicted_survived"] == 1
                    cls = "result-survive" if survived else "result-not"
                    label = "SURVIVAL PREDICTED" if survived else "NON-SURVIVAL PREDICTED"
                    symbol = "✦" if survived else "◈"
                    st.markdown(f"""<div class="{cls}">
                    <div class="small-label">MODEL INFERENCE COMPLETE</div>
                    <div style="font-size:32px;margin:14px 0 8px 0">{symbol} {label}</div>
                    <div style="color:#c0d5e3">Predicted class:
                    <b>{result['predicted_survived']}</b></div>
                    </div>""",unsafe_allow_html=True)
                    st.write("")
                    st.markdown("**Request summary**")
                    st.dataframe(pd.DataFrame([payload]),use_container_width=True,hide_index=True)
                else:
                    st.error(f"API returned HTTP {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Could not contact FastAPI: {e}")
        else:
            st.info("Your prediction will appear here after you submit a passenger profile.")
            st.markdown("""
            **Inference pipeline**
            
            1. Validate passenger fields  
            2. Send a REST request to FastAPI  
            3. Apply the saved preprocessing pipeline  
            4. Return the predicted class
            """)
        st.markdown("</div>",unsafe_allow_html=True)

elif page == "Batch Results":
    st.markdown("## Batch inference explorer")
    st.caption("Predictions generated by inference.py and saved to batch_predictions.csv.")
    if os.path.exists("batch_predictions.csv"):
        batch = pd.read_csv("batch_predictions.csv")
        x,y,z = st.columns(3)
        x.metric("Records processed",len(batch))
        if "Predicted_Survived" in batch:
            y.metric("Predicted survived",int((batch["Predicted_Survived"]==1).sum()))
            z.metric("Predicted not survived",int((batch["Predicted_Survived"]==0).sum()))
        st.dataframe(batch,use_container_width=True,hide_index=True)
        st.download_button("Download batch predictions CSV",
            batch.to_csv(index=False).encode("utf-8"),
            "batch_predictions.csv","text/csv")
    else:
        st.warning("batch_predictions.csv not found. Run inference.py first.")

elif page == "System Overview":
    st.markdown("## Deployment architecture")
    st.markdown("""
    <div class="section-card">
      <div class="small-label">END-TO-END WORKFLOW</div>
      <h3>From dataset to served inference</h3>
      <p>01 &nbsp; Kaggle Titanic dataset</p>
      <p>↓</p>
      <p>02 &nbsp; Preprocessing + model training</p>
      <p>↓</p>
      <p>03 &nbsp; Serialized pipeline: <code>model.pkl</code></p>
      <p>↓</p>
      <p>04 &nbsp; FastAPI endpoints: <code>/health</code> and <code>/predict</code></p>
      <p>↓</p>
      <p>05 &nbsp; Streamlit user interface</p>
    </div>
    """,unsafe_allow_html=True)
    st.markdown("### Project capabilities")
    st.write("- Holdout accuracy evaluation during training")
    st.write("- Online, batch and simulated streaming inference")
    st.write("- Reusable serialized preprocessing and classification pipeline")
    st.write("- Pydantic request validation and REST API serving")
    st.info("This project is an academic demonstration, not a reliable real-world predictor of individual outcomes.")

st.divider()
st.caption("MLOps CLA 4  •  Titanic Survival Prediction  •  Streamlit + FastAPI + Scikit-learn")

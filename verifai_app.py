import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Load model and vectorizer
@st.cache_resource
def load_model():
    vectorizer = joblib.load("./models/verifai_vectorizer.pkl")
    model = joblib.load("./models/verifai_voting_model.pkl")
    return vectorizer, model

vectorizer, model = load_model()

# Sidebar
st.sidebar.title("ℹ️ About VerifAI")
st.sidebar.markdown(
    """
    **VerifAI** is a Fake News Detection system powered by Machine Learning and NLP.

    🧠 **Model Used**: Voting Classifier (Logistic Regression + SVM + Random Forest)  
    🔤 **Feature Extraction**: TF-IDF Vectorizer  
    📄 **Explainability**: LIME (Local Interpretable Model-agnostic Explanations)  
    📊 **Evaluation**: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC

    ---
    **Instructions:**
    1. Enter a **news title**.
    2. Optionally provide a **context/paragraph**.
    3. Click **✅ Check Prediction** to classify.
    4. Use **🧪 Explain Prediction** to see LIME-based feature impact.

    ---
    🔐 *No data is stored. All processing is done locally.*
    """
)

# GitHub link
st.sidebar.markdown("---")
st.sidebar.markdown("🔗 [View on GitHub](https://github.com/Radical-Ghost/VerifAI---Fake-New-Detection)")

# App UI
st.title("🧠 VerifAI - Fake News Detector")
st.markdown("Enter the **news title** and (optionally) a **context/paragraph** to check if it's **Real or Fake**.")

# Input fields
title_input = st.text_input("📰 Enter news title/headline:")
context_input = st.text_area("📝 (Optional) Enter article context/paragraph:", height=150)

combined_input = title_input + " " + context_input if context_input else title_input

def explain_prediction(text, vectorizer, model):
    import lime
    from lime.lime_text import LimeTextExplainer

    class_names = ['Fake', 'Real']
    explainer = LimeTextExplainer(class_names=class_names)

    def predictor(texts):
        X = vectorizer.transform(texts)
        return model.predict_proba(X)

    exp = explainer.explain_instance(text, predictor, num_features=10)
    html = exp.as_html()

    # 🧼 Inject CSS fix
    style_fix = """
    <style>
        body {
            background-color: white !important;
            color: black !important;
            font-family: Arial, sans-serif;
        }
        .lime {
            background-color: #fff !important;
            color: #000 !important;
        }
        .highlight {
            background-color: #f5f5a0 !important;
            color: #000 !important;
        }
        .table {
            color: #000 !important;
        }
    </style>
    """
    return style_fix + html

# Initialize prediction toggle
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False
    st.session_state.pred_result = None
    st.session_state.pred_proba = None

# Toggle behavior
if st.button("✅ Check Prediction"):
    if combined_input.strip() == "":
        st.warning("Please enter the news title.")
    else:
        st.session_state.show_prediction = not st.session_state.show_prediction
        if st.session_state.show_prediction:
            X_input = vectorizer.transform([combined_input])
            st.session_state.pred_result = model.predict(X_input)[0]
            st.session_state.pred_proba = model.predict_proba(X_input)[0]

# Display prediction results if toggled
if st.session_state.show_prediction and st.session_state.pred_result is not None:
    prediction = st.session_state.pred_result
    proba = st.session_state.pred_proba

    label = "🟢 Real" if prediction == 1 else "🔴 Fake"
    confidence = f"{np.max(proba) * 100:.2f}%"

    st.subheader("Prediction:")
    st.markdown(f"### {label}")
    st.markdown(f"**Confidence:** {confidence}")

    st.progress(proba[1], text="Real")
    st.progress(proba[0], text="Fake")

# Initialize the toggle state if not already present
if "show_explanation" not in st.session_state:
    st.session_state.show_explanation = False

# Explanation toggle button (only show if text is provided)
if combined_input.strip():
    if st.button("🧪 Explain Prediction (LIME)"):
        st.session_state.show_explanation = not st.session_state.show_explanation

    if st.session_state.show_explanation:
        try:
            explanation_html = explain_prediction(combined_input, vectorizer, model)
            components.html(explanation_html, height=600, scrolling=True)
        except Exception as e:
            st.error(f"Failed to generate explanation: {e}")

st.markdown("---")
with st.expander("📊 Model Evaluation Report"):
    st.subheader("🧮 Classification Metrics")
    st.markdown("""
    - **Accuracy**: `0.9918`
    - **Precision**: `0.9899`
    - **Recall**: `0.9950`
    - **F1 Score**: `0.9925`
    """)

    st.markdown("### 🔍 Classification Report")
    st.code("""
              precision    recall  f1-score   support

    Fake       0.99      0.99      0.99      3581
    Real       0.99      1.00      0.99      4239

accuracy                           0.99      7820
macro avg       0.99      0.99      0.99      7820
weighted avg    0.99      0.99      0.99      7820
    """, language='text')

    st.subheader("📉 Confusion Matrix")
    st.image("./assets/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)


    st.subheader("📈 ROC Curve")
    st.image("./assets/roc_curve.png", caption="ROC Curve (AUC ≈ 1.00)", use_container_width=True)
    

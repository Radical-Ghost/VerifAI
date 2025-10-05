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

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .prediction-result {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .fake-result {
        background-color: #fee;
        border-left: 4px solid #f56565;
    }
    
    .real-result {
        background-color: #efe;
        border-left: 4px solid #48bb78;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

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

    # Enhanced CSS styling with proper scrolling
    style_fix = """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background-color: #fafafa !important;
            color: #262730 !important;
            font-family: "Source Sans Pro", sans-serif !important;
            line-height: 1.6;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }
        
        html {
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }
        
        .lime {
            background-color: #fafafa !important;
            color: #262730 !important;
            border-radius: 8px;
            padding: 10px 15px;
            margin: 0;
            max-height: 100vh;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }
        
        .lime h2 {
            color: #262730 !important;
            font-size: 1.5rem !important;
            margin-bottom: 10px !important;
            font-weight: 600 !important;
        }
        
        .lime h3 {
            color: #262730 !important;
            font-size: 1.2rem !important;
            margin: 10px 0 8px 0 !important;
            font-weight: 500 !important;
        }
        
        .lime table {
            width: 100% !important;
            border-collapse: collapse !important;
            margin: 10px 0 !important;
            background-color: white !important;
            border-radius: 6px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        }
        
        .lime th, .lime td {
            padding: 12px 16px !important;
            text-align: left !important;
            border-bottom: 1px solid #e6e6e6 !important;
            color: #262730 !important;
            font-size: 0.9rem !important;
        }
        
        .lime th {
            background-color: #f8f9fa !important;
            font-weight: 600 !important;
            color: #1f1f1f !important;
        }
        
        .lime tr:hover {
            background-color: #f8f9fa !important;
        }
        
        .lime .highlight {
            padding: 2px 4px !important;
            border-radius: 3px !important;
            font-weight: 500 !important;
        }
        
        /* Custom scrollbar styles (thin and styled) */
        ::-webkit-scrollbar {
            width: 8px !important;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1 !important;
            border-radius: 4px !important;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #c1c1c1 !important;
            border-radius: 4px !important;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8 !important;
        }
        
        /* Firefox scrollbar */
        html {
            scrollbar-width: thin !important;
            scrollbar-color: #c1c1c1 #f1f1f1 !important;
        }
        
        /* Ensure text wrapping */
        .lime p, .lime div {
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
        }
        
        /* Make content scrollable but contained */
        .lime > div {
            max-height: none !important;
            overflow: visible !important;
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
    
    # Create a more visually appealing result display
    result_class = "real-result" if prediction == 1 else "fake-result"
    
    st.markdown(f"""
    <div class="prediction-result {result_class}">
        <h2>{label}</h2>
        <p><strong>Confidence: {confidence}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Display probability bars with better styling
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🟢 Real", f"{proba[1]*100:.1f}%")
        st.progress(proba[1])
    
    with col2:
        st.metric("🔴 Fake", f"{proba[0]*100:.1f}%")
        st.progress(proba[0])

# Initialize the toggle state if not already present
if "show_explanation" not in st.session_state:
    st.session_state.show_explanation = False

# Explanation toggle button (only show if text is provided)
if combined_input.strip():
    if st.button("🧪 Explain Prediction (LIME)"):
        st.session_state.show_explanation = not st.session_state.show_explanation

    if st.session_state.show_explanation:
        try:
            with st.container():
                st.markdown("### 🔍 Feature Importance Analysis")
                st.markdown("The highlighted words show how much each word contributed to the prediction:")
                
                explanation_html = explain_prediction(combined_input, vectorizer, model)
                
                # Calculate dynamic height based on text length
                text_length = len(combined_input)
                base_height = 400
                dynamic_height = min(base_height + (text_length // 50) * 30, 800)  # Max 800px
                
                components.html(
                    explanation_html, 
                    height=dynamic_height, 
                    scrolling=True
                )
                
                st.info("💡 **How to read this:** Blue highlights indicate words that support the 'Real' prediction, while orange highlights show words that support the 'Fake' prediction. The intensity of the color represents the strength of influence.")
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
    

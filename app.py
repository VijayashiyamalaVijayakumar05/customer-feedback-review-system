
import streamlit as st
import pickle, re, os
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Feedback Intelligence",
    page_icon="💬",
    layout="centered"
)
 
# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp { background: #0F1117; }

            
.block-container {
    max-width: 95% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
 
.main-title {
    font-size: 4rem; font-weight: 800; color: #FFFFFF;
    text-align: center; margin: 5rem 0 0.3rem; letter-spacing: -0.5px;
}
.sub-title {
    text-align: center; color: #6B7280;
    font-size: 1.5rem; margin-bottom: 2rem;                  
}
            

.stTextArea textarea {
    background: #1C1F2B !important; border: 1.5px solid #2D3143 !important;min-height: 90px !important;
max-height: 90px !important;
    color: #E5E7EB !important; border-radius: 10px !important; font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #1D9E75 !important; box-shadow: 0 0 0 2px rgba(29,158,117,0.15) !important;min-height: 90px !important;
max-height: 90px !important;
}
div.stButton > button {
    background: #1D9E75 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    padding: 42px 2rem !important; font-size: 0.95rem !important;
    font-weight: 600 !important; width: 100% !important;
}
div.stButton > button:hover { background: #158A62 !important; }
 
.result-label { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.2rem; }
.result-sub { color: #9CA3AF; font-size: 0.85rem; }
 
.stat-row { display: flex; gap: 12px; margin: 1.2rem 0; }
.stat-box { flex: 1; background: #1C1F2B; border-radius: 10px; padding: 0.9rem 1rem; text-align: center; border: 1px solid #2D3143; }
.stat-val { font-size: 1.4rem; font-weight: 700; color: #FFFFFF; }
.stat-lbl { font-size: 0.75rem; color: #6B7280; margin-top: 2px; }
            

 
.section-head { font-size: 0.75rem; font-weight: 600; color: #6B7280; text-transform: uppercase; letter-spacing: 1px; margin: 1.6rem 0 0.6rem; }
hr.custom { border: none; border-top: 1px solid #2D3143; margin: 1.5rem 0; }
.preview-box { background: #1C1F2B; border: 1px solid #2D3143; border-radius: 8px; padding: 1rem 1rem; font-family: monospace; font-size: 0.82rem; color: #9CA3AF; word-break: break-word; }
</style>
""", unsafe_allow_html=True)
 
 
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f: model = pickle.load(f)
    with open('tfidf.pkl', 'rb') as f: tfidf = pickle.load(f)
    return model, tfidf
 
model, tfidf = load_model()
stop_words   = set(stopwords.words('english'))
 
 
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([w for w in text.split() if w not in stop_words])
 
 
def predict(text):
    cleaned = clean_text(text)
    vec     = tfidf.transform([cleaned])
    pred    = model.predict(vec)[0]
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(vec)[0]
        exp_s  = np.exp(scores - scores.max())
        conf   = dict(zip(model.classes_, exp_s / exp_s.sum()))
    else:
        conf = {c: 0.33 for c in model.classes_}
    return pred, conf, cleaned
 
 
# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Customer Feedback Review</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Paste a review — get instant sentiment analysis</div>', unsafe_allow_html=True)
st.markdown('<hr class="custom">', unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────────────────────

 
# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Single Review", "CSV Upload"])

with tab1:
    review = st.text_area("review", placeholder="Type a customer review here...",
                          height=200, label_visibility="collapsed")
    if st.button("Analyse Sentiment"):
        if review.strip():
            pred, conf, cleaned = predict(review)
            colors = {
                'positive': ('#1D9E75', 'result-positive', '😊', 'Positive sentiment detected'),
                'neutral':  ('#EF9F27', 'result-neutral',  '😐', 'Neutral sentiment detected'),
                'negative': ('#D85A30', 'result-negative', '😠', 'Negative sentiment detected'),
            }
            col, cls, emoji, desc = colors[pred]
 
            st.markdown(f"""
            <div class="{cls}">
                <div class="result-label" style="color:{col};">{emoji} {pred.upper()}</div>
                <div class="result-sub">{desc}</div>
            </div>""", unsafe_allow_html=True)
 
            st.markdown('<div class="section-head">Confidence</div>', unsafe_allow_html=True)
            bar_c = {'positive':'#1D9E75','neutral':'#EF9F27','negative':'#D85A30'}
            for sentiment, score in sorted(conf.items(), key=lambda x: x[1], reverse=True):
                pct = round(score * 100, 1)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">'
                    f'<span style="width:70px;font-size:0.82rem;color:#9CA3AF;">{sentiment}</span>'
                    f'<div style="flex:1;background:#2D3143;border-radius:4px;height:8px;">'
                    f'<div style="width:{pct}%;background:{bar_c[sentiment]};height:8px;border-radius:4px;"></div></div>'
                    f'<span style="width:42px;text-align:right;font-size:0.82rem;color:#E5E7EB;font-weight:600;">{pct}%</span>'
                    f'</div>', unsafe_allow_html=True)
 
            with st.expander("What did the model actually read?"):
                st.markdown(f'<div class="preview-box">{cleaned}</div>', unsafe_allow_html=True)
                st.caption("Stopwords removed · lowercased · symbols stripped — this is what TF-IDF processed")
        else:
            st.warning("Please type a review first!")
 
with tab2:
    st.markdown("Upload a CSV file with a column named **`review_text`**")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        if 'review_text' not in df_up.columns:
            st.error("Column 'review_text' not found. Please rename your column to review_text")
        else:
            with st.spinner("Analysing reviews..."):
                df_up['sentiment'] = df_up['review_text'].apply(lambda x: predict(str(x))[0])
            pos_c = (df_up['sentiment']=='positive').sum()
            neu_c = (df_up['sentiment']=='neutral').sum()
            neg_c = (df_up['sentiment']=='negative').sum()
            st.success(f"Done! Analysed {len(df_up)} reviews.")
            st.markdown(f"""
            <div class="stat-row">
              <div class="stat-box"><div class="stat-val" style="color:#1D9E75;">{pos_c}</div><div class="stat-lbl">Positive</div></div>
              <div class="stat-box"><div class="stat-val" style="color:#EF9F27;">{neu_c}</div><div class="stat-lbl">Neutral</div></div>
              <div class="stat-box"><div class="stat-val" style="color:#D85A30;">{neg_c}</div><div class="stat-lbl">Negative</div></div>
            </div>""", unsafe_allow_html=True)
            st.dataframe(df_up[['review_text','sentiment']].head(20), use_container_width=True)
            st.download_button("Download Results as CSV",
                df_up.to_csv(index=False).encode('utf-8'),
                "results_with_sentiment.csv", "text/csv")
 
st.markdown('<hr class="custom">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#4B5563;font-size:0.78rem;padding-bottom:1rem;">
    Customer Feedback Intelligence &nbsp;·&nbsp; Linear SVM &nbsp;·&nbsp; TF-IDF &nbsp;·&nbsp; Streamlit<br>
    Xploro ML Internship &nbsp;·&nbsp; Shiyamala &nbsp;·&nbsp; CSE (AIML)  </div>""", unsafe_allow_html=True)
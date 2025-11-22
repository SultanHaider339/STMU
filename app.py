# main_app.py - Enhanced Affective Assignment Analyzer

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
import torch
import pypdf
import docx
import re
from datetime import datetime
from typing import List, Dict, Any, Union
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
import numpy as np

# ============================================================
#                0. CONFIG + CONSTANTS
# ============================================================

EKMAN_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'shame', 'pride']

EMOTION_COLORS = {
    'anger': '#E74C3C', 'disgust': '#9B59B6', 'fear': '#8E44AD',
    'joy': '#F1C40F', 'sadness': '#3498DB', 'surprise': '#E67E22',
    'shame': '#95A5A6', 'pride': '#2ECC71', 'neutral': '#BDC3C7'
}

EMOTION_ICONS = {
    'anger': '😠', 'disgust': '🤢', 'fear': '😨', 'joy': '😊',
    'sadness': '😢', 'surprise': '😲', 'shame': '😳', 'pride': '😌', 'neutral': '😐'
}

GO_EMOTIONS_TO_EKMAN: Dict[str, str] = {
    'admiration': 'pride', 'amusement': 'joy', 'anger': 'anger', 'annoyance': 'anger',
    'approval': 'pride', 'caring': 'pride', 'confusion': 'surprise', 'curiosity': 'surprise',
    'desire': 'joy', 'disappointment': 'sadness', 'disapproval': 'anger', 'disgust': 'disgust',
    'embarrassment': 'shame', 'excitement': 'joy', 'fear': 'fear', 'gratitude': 'pride',
    'grief': 'sadness', 'joy': 'joy', 'love': 'joy', 'nervousness': 'fear',
    'optimism': 'pride', 'pride': 'pride', 'realization': 'surprise', 'relief': 'joy',
    'remorse': 'shame', 'sadness': 'sadness', 'surprise': 'surprise', 'neutral': 'neutral',
}

CRITICAL_THRESHOLD = 0.35
CRITICAL_EMOTIONS = ['anger', 'fear', 'sadness']

MOCK_EKMAN_LEXICON: Dict[str, Any] = {
    "anger": ["furious", "hate", "enraged", "mad", "frustrat", "irritat", "disgusting", "disapprov", "hostile", "resent"],
    "joy": ["happy", "delight", "wonderful", "great", "excit", "love", "cheer", "optimis", "pleased", "thrilled"],
    "sadness": ["sad", "depress", "grief", "remorse", "miserable", "sorrow", "disappoint", "hopeless", "lonely"],
    "fear": ["scared", "afraid", "terrified", "nervous", "anxiety", "worried", "panic", "dread", "uneasy"],
    "disgust": ["gross", "nauseating", "repulse", "sickening", "yuck", "contempt", "revolting", "appalling"],
    "surprise": ["shock", "astound", "confused", "curiosity", "realiz", "unexpected", "amazed", "startled"],
    "shame": ["ashamed", "embarrass", "guilty", "humiliat", "remorse", "regret", "mortified"],
    "pride": ["proud", "accomplish", "admire", "caring", "gratitude", "approval", "optimism", "confident", "achieve"],
    "amplifiers": ["very", "extremely", "so", "totally", "absolutely", "greatly", "much", "incredibly", "truly"],
    "negators": ["not", "never", "no", "don't", "didn't", "isn't", "couldn't", "wasn't", "hardly", "barely"]
}

# ============================================================
#           1. DATA EXTRACTION & PREPROCESSING
# ============================================================

def extract_text_from_file(file_path: Union[str, io.BytesIO], file_type: str) -> str:
    text = ""
    if file_type == 'pdf':
        try:
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            return f"ERROR_PDF_EXTRACTION: {e}"
    elif file_type == 'docx':
        try:
            document = docx.Document(file_path)
            for paragraph in document.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            return f"ERROR_DOCX_EXTRACTION: {e}"
    elif file_type == 'txt':
        try:
            if isinstance(file_path, str):
                text = open(file_path, 'r', encoding='utf-8').read()
            else:
                text = file_path.read().decode('utf-8')
        except Exception as e:
            return f"ERROR_TXT_EXTRACTION: {e}"
    else:
        return f"ERROR_UNSUPPORTED_TYPE: {file_type}"
    return " ".join(text.split()).strip()

def preprocess_text(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

def get_word_count(text: str) -> int:
    return len(text.split())

def get_readability_score(text: str) -> float:
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences or not words:
        return 0
    avg_sentence_len = len(words) / len(sentences)
    syllables = sum(count_syllables(w) for w in words)
    avg_syllables = syllables / len(words) if words else 0
    flesch = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables
    return max(0, min(100, flesch))

def count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith('e'):
        count -= 1
    return max(1, count)

# ============================================================
#            2. RULE-BASED EMOTION ENGINE
# ============================================================

class EkmanRuleScorer:
    def __init__(self, lexicon: Dict[str, Any]):
        self.lexicon = lexicon
        self.amplifiers = [w.lower() for w in lexicon["amplifiers"]]
        self.negators = [w.lower() for w in lexicon["negators"]]
        self.emotion_words = {e: [w.lower() for w in lexicon[e]] for e in EKMAN_EMOTIONS}

    def _score_token(self, word: str, context: List[str]):
        scores = {e: 0.0 for e in EKMAN_EMOTIONS}
        triggered = []
        for emotion, keywords in self.emotion_words.items():
            if any(word.startswith(k) for k in keywords):
                score = 1.0
                triggered.append(emotion)
                if any(a in context for a in self.amplifiers):
                    score *= 1.5
                if any(n in context for n in self.negators):
                    score *= -0.8
                scores[emotion] += score
        return scores, triggered

    def score_sentence(self, sentence: str):
        words = sentence.lower().split()
        total = {e: 0.0 for e in EKMAN_EMOTIONS}
        triggered_words = set()
        for i, w in enumerate(words):
            ctx = words[max(0, i - 3):i]
            s, trig = self._score_token(w, ctx)
            if trig:
                triggered_words.add(w)
            for k in total:
                total[k] += s[k]
        mag = sum(abs(v) for v in total.values())
        if mag == 0:
            return {e: 0.0 for e in EKMAN_EMOTIONS}, triggered_words
        return {e: abs(total[e]) / mag for e in total}, triggered_words

    def analyze_text(self, sentences: List[str]):
        output = []
        for s in sentences:
            sc, trig = self.score_sentence(s)
            primary = max(sc, key=sc.get) if sc else "neutral"
            output.append({
                "sentence": s, "system": "Rule-Based", "primary_emotion": primary,
                "scores": sc, "explanation": f"Keywords: {', '.join(trig)}" if trig else "No emotional cues."
            })
        return output

# ============================================================
#          3. TRANSFORMER MODEL (GO EMOTIONS)
# ============================================================

MODEL_NAME = "SamLowe/roberta-base-go_emotions"
DEVICE = 0 if torch.cuda.is_available() else -1

class TransformerEmotionModel:
    def __init__(self):
        self.classifier = self._load_model()
        self.id2label = self.classifier.model.config.id2label if self.classifier else {}

    @st.cache_resource
    def _load_model(_self):
        try:
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            classifier = pipeline("text-classification", model=model, tokenizer=tokenizer,
                                  device=DEVICE, return_all_scores=True, function_to_apply="sigmoid")
            return classifier
        except Exception as e:
            st.error(f"Model load failed: {e}")
            return None

    def batch_analyze(self, sentences: List[str]):
        if not self.classifier:
            return [{"sentence": s, "primary_emotion": "neutral", "scores": {}, "explanation": "Model not loaded"} for s in sentences]
        preds = self.classifier(sentences, batch_size=16)
        out = []
        for i, row in enumerate(preds):
            sc = {item["label"]: item["score"] for item in row}
            primary = max(sc, key=sc.get)
            out.append({"sentence": sentences[i], "system": "Transformer", "primary_emotion": primary,
                        "scores": sc, "explanation": f"Confidence: {sc[primary]:.2%}"})
        return out

# ============================================================
#            4. SCORE FUSION ENGINE
# ============================================================

class AnalysisEngine:
    def __init__(self):
        self.rule_scorer = EkmanRuleScorer(MOCK_EKMAN_LEXICON)
        self.transformer_model = TransformerEmotionModel()

    def fuse_scores(self, model_scores, rule_scores):
        mapped = {e: 0 for e in EKMAN_EMOTIONS}
        for go, val in model_scores.items():
            ek = GO_EMOTIONS_TO_EKMAN.get(go)
            if ek in mapped:
                mapped[ek] += val
        total = sum(mapped.values())
        if total:
            mapped = {k: v / total for k, v in mapped.items()}
        fused = {e: mapped.get(e, 0) * 0.7 + rule_scores.get(e, 0) * 0.3 for e in EKMAN_EMOTIONS}
        s = sum(fused.values())
        if s:
            fused = {k: v / s for k, v in fused.items()}
        return fused

    def analyze_submission(self, sents, filename, student_id):
        m = self.transformer_model.batch_analyze(sents)
        r = self.rule_scorer.analyze_text(sents)
        out = []
        for idx, (mm, rr) in enumerate(zip(m, r)):
            fused = self.fuse_scores(mm["scores"], rr["scores"])
            primary_fused = max(fused, key=fused.get)
            is_critical = any(fused[x] > CRITICAL_THRESHOLD for x in CRITICAL_EMOTIONS)
            sentiment = "positive" if primary_fused in ['joy', 'pride', 'surprise'] else "negative" if primary_fused in ['anger', 'fear', 'sadness', 'disgust', 'shame'] else "neutral"
            out.append({
                "index": idx + 1, "student_id": student_id, "filename": filename, "sentence": mm["sentence"],
                "model_primary": mm["primary_emotion"], "rule_primary": rr["primary_emotion"],
                "fused_primary": primary_fused, "fused_max_score": fused[primary_fused],
                "sentiment": sentiment, "is_critical": is_critical,
                "model_scores_28": mm["scores"], "rule_scores_8": rr["scores"], "fused_scores_8": fused,
                "model_explanation": mm["explanation"], "rule_explanation": rr["explanation"],
            })
        return out

# ============================================================
#             5. VISUALIZATION FUNCTIONS
# ============================================================

def create_emotion_radar(df, title):
    if df.empty:
        return go.Figure()
    agg = pd.DataFrame(df["fused_scores_8"].tolist()).mean()
    fig = go.Figure(data=go.Scatterpolar(
        r=[agg[e] for e in EKMAN_EMOTIONS], theta=EKMAN_EMOTIONS, fill='toself',
        line_color='#667eea', fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      title=title, height=400, template="plotly_dark")
    return fig

def create_emotion_heatmap(df, title):
    if df.empty:
        return go.Figure()
    scores_df = pd.DataFrame(df["fused_scores_8"].tolist())
    scores_df.index = [f"S{i+1}" for i in range(len(scores_df))]
    fig = px.imshow(scores_df.T, labels=dict(x="Sentence", y="Emotion", color="Score"),
                    color_continuous_scale="Viridis", title=title, height=400)
    fig.update_layout(template="plotly_dark")
    return fig

def create_emotion_timeline(df, title):
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    for emotion in EKMAN_EMOTIONS:
        scores = [row[emotion] for row in df["fused_scores_8"]]
        fig.add_trace(go.Scatter(x=list(range(1, len(scores)+1)), y=scores,
                                 mode='lines+markers', name=emotion,
                                 line=dict(color=EMOTION_COLORS[emotion], width=2)))
    fig.update_layout(title=title, xaxis_title="Sentence #", yaxis_title="Score",
                      height=400, template="plotly_dark", hovermode='x unified')
    return fig

def create_sentiment_gauge(positive_pct, negative_pct, neutral_pct):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=positive_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Positivity Score", 'font': {'size': 20}},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#2ECC71"},
               'steps': [{'range': [0, 33], 'color': "#E74C3C"},
                        {'range': [33, 66], 'color': "#F39C12"},
                        {'range': [66, 100], 'color': "#2ECC71"}],
               'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': positive_pct}}))
    fig.update_layout(height=300, template="plotly_dark")
    return fig

def create_emotion_bar_chart(df, title):
    if df.empty:
        return px.bar(title="No data")
    agg = pd.DataFrame(df["fused_scores_8"].tolist()).mean().reset_index()
    agg.columns = ["Emotion", "Score"]
    agg["Color"] = agg["Emotion"].map(EMOTION_COLORS)
    fig = px.bar(agg, x="Emotion", y="Score", title=title, color="Emotion",
                 color_discrete_map=EMOTION_COLORS, height=400)
    fig.update_layout(xaxis={'categoryorder': 'total descending'}, template="plotly_dark", showlegend=False)
    return fig

def create_primary_distribution_chart(df, title):
    if df.empty:
        return px.pie(names=["None"], values=[1], title="Empty")
    c = df["fused_primary"].value_counts().reset_index()
    c.columns = ["Emotion", "Count"]
    c["Color"] = c["Emotion"].map(EMOTION_COLORS)
    fig = px.pie(c, names="Emotion", values="Count", title=title, hole=.4,
                 color="Emotion", color_discrete_map=EMOTION_COLORS)
    fig.update_layout(template="plotly_dark")
    return fig

def create_word_cloud_data(df):
    all_text = " ".join(df["sentence"].tolist())
    words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
    stopwords = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'they', 'their', 'what', 'when', 'where', 'which', 'would', 'could', 'should', 'about', 'there', 'being', 'some', 'into', 'just', 'very', 'more', 'other'}
    words = [w for w in words if w not in stopwords]
    return Counter(words).most_common(20)

# ============================================================
#                    MAIN STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(layout="wide", page_title="Affective Analyzer Pro", page_icon="🧠")
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; color: white; text-align: center;}
    .metric-card {background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 1.5rem; border-radius: 12px; border: 1px solid #333; text-align: center;}
    .emotion-badge {display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: bold; margin: 0.2rem;}
    .critical-alert {background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {background-color: #1a1a2e; border-radius: 8px; padding: 10px 20px;}
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header"><h1>🧠 Affective Assignment Analyzer Pro</h1><p>AI-powered emotional intelligence for educational content</p></div>', unsafe_allow_html=True)
    
    engine = get_analysis_engine()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/brain.png", width=100)
        st.header("📁 Input Configuration")
        
        input_method = st.radio("Select Input Method", ["📄 Upload Files", "✏️ Paste Text", "🔗 URL (Coming Soon)"], horizontal=False)
        
        uploaded_files, pasted_text = None, ""
        if input_method == "📄 Upload Files":
            uploaded_files = st.file_uploader("Upload PDF/DOCX/TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        elif input_method == "✏️ Paste Text":
            pasted_text = st.text_area("Paste your text here", height=200, placeholder="Enter text to analyze...")
        
        st.divider()
        st.header("⚙️ Settings")
        student_id = st.text_input("Student/Document ID", value="DOC001")
        critical_threshold = st.slider("Critical Alert Threshold", 0.1, 0.8, CRITICAL_THRESHOLD, 0.05)
        show_advanced = st.checkbox("Show Advanced Analytics", value=True)
        
        st.divider()
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        st.divider()
        st.caption("v2.0 | Powered by RoBERTa + Rule Engine")

    if not run_btn:
        # Show welcome screen
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**📄 Step 1:** Upload documents or paste text")
        with col2:
            st.info("**⚙️ Step 2:** Configure analysis settings")
        with col3:
            st.info("**🚀 Step 3:** Click 'Run Analysis' to start")
        
        st.markdown("### 🎯 Features")
        feat_cols = st.columns(4)
        feat_cols[0].success("✅ Multi-format Support\n\nPDF, DOCX, TXT")
        feat_cols[1].success("✅ Hybrid Analysis\n\nAI + Rule-based")
        feat_cols[2].success("✅ Critical Alerts\n\nFlag concerning content")
        feat_cols[3].success("✅ Rich Visualizations\n\nInteractive charts")
        return

    # Process inputs
    all_results = []
    full_text = ""
    
    with st.spinner("🔄 Processing your content..."):
        if pasted_text.strip():
            sents = preprocess_text(pasted_text)
            all_results += engine.analyze_submission(sents, "Pasted_Text", student_id)
            full_text = pasted_text
        
        if uploaded_files:
            for f in uploaded_files:
                ft = f.name.split(".")[-1].lower()
                data = io.BytesIO(f.read())
                txt = extract_text_from_file(data, ft)
                if txt.startswith("ERROR"):
                    st.error(f"Error in {f.name}: {txt}")
                    continue
                sents = preprocess_text(txt)
                all_results += engine.analyze_submission(sents, f.name, student_id)
                full_text += " " + txt

    if not all_results:
        st.warning("⚠️ No analyzable sentences found. Please provide more content.")
        return

    df = pd.DataFrame(all_results)
    df["is_critical"] = df["is_critical"].astype(bool)
    
    # Calculate metrics
    total = len(df)
    critical = df["is_critical"].sum()
    avg_conf = df["fused_max_score"].mean() * 100
    mode_emo = df["fused_primary"].mode()[0] if not df.empty else "neutral"
    sentiment_counts = df["sentiment"].value_counts()
    pos_pct = sentiment_counts.get("positive", 0) / total * 100
    neg_pct = sentiment_counts.get("negative", 0) / total * 100
    neu_pct = sentiment_counts.get("neutral", 0) / total * 100

    # Summary Metrics
    st.header("📊 Analysis Dashboard")
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("📝 Sentences", total)
    m2.metric(f"{EMOTION_ICONS.get(mode_emo, '😐')} Primary Emotion", mode_emo.title())
    m3.metric("⚠️ Critical Flags", critical, delta=f"-{total-critical} safe" if critical < total else None)
    m4.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")
    m5.metric("📖 Word Count", get_word_count(full_text))
    m6.metric("📚 Readability", f"{get_readability_score(full_text):.0f}/100")

    # Critical Alert
    if critical > 0:
        st.markdown(f'<div class="critical-alert">⚠️ <strong>Alert:</strong> {critical} sentence(s) flagged with high levels of anger, fear, or sadness. Review recommended.</div>', unsafe_allow_html=True)
        with st.expander("🔍 View Flagged Sentences"):
            for _, row in df[df["is_critical"]].iterrows():
                st.warning(f"**[{row['fused_primary'].upper()}]** {row['sentence']}")

    st.divider()

    # Tabs for different views
    tabs = st.tabs(["📈 Overview", "🎨 Emotion Analysis", "📋 Detailed Data", "🔬 Advanced"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_emotion_bar_chart(df, "Average Emotion Scores"), use_container_width=True)
        with col2:
            st.plotly_chart(create_primary_distribution_chart(df, "Emotion Distribution"), use_container_width=True)
        
        st.plotly_chart(create_sentiment_gauge(pos_pct, neg_pct, neu_pct), use_container_width=True)

    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_emotion_radar(df, "Emotion Radar Profile"), use_container_width=True)
        with col2:
            st.plotly_chart(create_emotion_timeline(df, "Emotion Flow Timeline"), use_container_width=True)
        
        st.plotly_chart(create_emotion_heatmap(df, "Sentence-by-Sentence Emotion Heatmap"), use_container_width=True)

    with tabs[2]:
        st.subheader("📋 Per-Sentence Analysis")
        
        # Filters
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            emotion_filter = st.multiselect("Filter by Emotion", EKMAN_EMOTIONS, default=EKMAN_EMOTIONS)
        with filter_col2:
            sentiment_filter = st.multiselect("Filter by Sentiment", ["positive", "negative", "neutral"], default=["positive", "negative", "neutral"])
        with filter_col3:
            critical_filter = st.selectbox("Critical Status", ["All", "Critical Only", "Safe Only"])
        
        filtered_df = df[df["fused_primary"].isin(emotion_filter) & df["sentiment"].isin(sentiment_filter)]
        if critical_filter == "Critical Only":
            filtered_df = filtered_df[filtered_df["is_critical"]]
        elif critical_filter == "Safe Only":
            filtered_df = filtered_df[~filtered_df["is_critical"]]
        
        display_cols = ["index", "sentence", "fused_primary", "fused_max_score", "sentiment", "is_critical", "model_explanation"]
        st.dataframe(filtered_df[display_cols].rename(columns={
            "index": "#", "sentence": "Sentence", "fused_primary": "Emotion",
            "fused_max_score": "Confidence", "sentiment": "Sentiment",
            "is_critical": "Critical", "model_explanation": "Analysis"
        }), use_container_width=True, height=400)

    with tabs[3]:
        if show_advanced:
            st.subheader("🔬 Advanced Analytics")
            
            adv1, adv2 = st.columns(2)
            with adv1:
                st.markdown("**📊 Top Keywords Detected**")
                word_freq = create_word_cloud_data(df)
                if word_freq:
                    wf_df = pd.DataFrame(word_freq, columns=["Word", "Frequency"])
                    fig = px.bar(wf_df.head(15), x="Frequency", y="Word", orientation='h',
                                 color="Frequency", color_continuous_scale="Viridis")
                    fig.update_layout(template="plotly_dark", height=400, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
            
            with adv2:
                st.markdown("**🔄 Model vs Rule Agreement**")
                df["agreement"] = df["model_primary"].apply(lambda x: GO_EMOTIONS_TO_EKMAN.get(x, x)) == df["rule_primary"]
                agreement_pct = df["agreement"].mean() * 100
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=agreement_pct,
                    title={'text': "Model-Rule Agreement %"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#667eea"}}
                ))
                fig.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("**📈 Emotion Correlation Matrix**")
            scores_df = pd.DataFrame(df["fused_scores_8"].tolist())
            corr = scores_df.corr()
            fig = px.imshow(corr, text_auto='.2f', color_continuous_scale="RdBu_r",
                           title="How Emotions Correlate")
            fig.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)

    # Export Section
    st.divider()
    st.header("📥 Export Results")
    
    exp1, exp2, exp3, exp4 = st.columns(4)
    
    with exp1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📄 Download CSV", csv, f"emotion_report_{student_id}.csv", "text/csv", use_container_width=True)
    
    with exp2:
        json_data = df.to_json(orient="records", indent=2)
        st.download_button("📋 Download JSON", json_data, f"emotion_report_{student_id}.json", "application/json", use_container_width=True)
    
    with exp3:
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "document_id": student_id,
            "total_sentences": total,
            "primary_emotion": mode_emo,
            "critical_flags": int(critical),
            "avg_confidence": round(avg_conf, 2),
            "sentiment_breakdown": {"positive": round(pos_pct, 1), "negative": round(neg_pct, 1), "neutral": round(neu_pct, 1)},
            "emotion_averages": pd.DataFrame(df["fused_scores_8"].tolist()).mean().to_dict()
        }
        st.download_button("📊 Summary Report", json.dumps(summary, indent=2), f"summary_{student_id}.json", "application/json", use_container_width=True)
    
    with exp4:
        # Generate HTML report
        html_report = f"""
        <html><head><style>
        body {{font-family: Arial; padding: 20px; background: #1a1a2e; color: white;}}
        h1 {{color: #667eea;}} .metric {{background: #16213e; padding: 15px; border-radius: 10px; margin: 10px; display: inline-block;}}
        table {{width: 100%; border-collapse: collapse;}} th, td {{border: 1px solid #333; padding: 8px; text-align: left;}}
        th {{background: #667eea;}}
        </style></head><body>
        <h1>🧠 Emotion Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <div class="metric"><strong>Sentences:</strong> {total}</div>
        <div class="metric"><strong>Primary Emotion:</strong> {mode_emo}</div>
        <div class="metric"><strong>Critical Flags:</strong> {critical}</div>
        <div class="metric"><strong>Confidence:</strong> {avg_conf:.1f}%</div>
        <h2>Sentiment Breakdown</h2>
        <p>Positive: {pos_pct:.1f}% | Negative: {neg_pct:.1f}% | Neutral: {neu_pct:.1f}%</p>
        <h2>Detailed Analysis</h2>
        <table><tr><th>#</th><th>Sentence</th><th>Emotion</th><th>Confidence</th><th>Critical</th></tr>
        {''.join(f"<tr><td>{r['index']}</td><td>{r['sentence'][:100]}...</td><td>{r['fused_primary']}</td><td>{r['fused_max_score']:.2%}</td><td>{'⚠️' if r['is_critical'] else '✅'}</td></tr>" for _, r in df.iterrows())}
        </table></body></html>
        """
        st.download_button("🌐 HTML Report", html_report, f"report_{student_id}.html", "text/html", use_container_width=True)

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🧠 Affective Analyzer Pro v2.0 | Hybrid AI + Rule-Based Emotion Detection</p>
        <p>Powered by RoBERTa (GoEmotions) + Ekman Lexicon Engine</p>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_analysis_engine():
    return AnalysisEngine()


if __name__ == "__main__":
    main()

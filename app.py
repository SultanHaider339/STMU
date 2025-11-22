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
from typing import List, Dict, Any, Union, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64

# ============================================================
# 0. CONFIG + CONSTANTS
# ============================================================
EKMAN_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'shame', 'pride']
CRITICAL_EMOTIONS = ['anger', 'fear', 'sadness', 'disgust', 'shame']

GO_EMOTIONS_TO_EKMAN = {
    'admiration': 'pride', 'amusement': 'joy', 'anger': 'anger', 'annoyance': 'anger',
    'approval': 'pride', 'caring': 'pride', 'confusion': 'surprise', 'curiosity': 'surprise',
    'desire': 'joy', 'disappointment': 'sadness', 'disapproval': 'anger', 'disgust': 'disgust',
    'embarrassment': 'shame', 'excitement': 'joy', 'fear': 'fear', 'gratitude': 'pride',
    'grief': 'sadness', 'joy': 'joy', 'love': 'joy', 'nervousness': 'fear',
    'optimism': 'pride', 'pride': 'pride', 'realization': 'surprise', 'relief': 'joy',
    'remorse': 'shame', 'sadness': 'sadness', 'surprise': 'surprise', 'neutral': 'neutral',
}

# Enhanced lexicon with weights and better coverage
EKMAN_LEXICON = {
    "anger": {"furious":3, "hate":3, "enraged":3, "mad":2, "angry":2, "frustrated": f2, "irritated":2, "annoyed":1.5, "pissed":3, "rage":3},
    "joy": {"happy":2, "delighted":2.5, "joy":3, "excited":2, "wonderful":2.5, "great":1.5, "love":3, "amazing":2.5, "fantastic":2},
    "sadness": {"sad":2, "depressed":3, "grief":3, "heartbroken":3, "miserable":2.5, "devastated":3, "disappointed":2, "upset":2},
    "fear": {"scared":2.5, "afraid":2.5, "terrified":3, "anxious":2, "worried":2, "nervous":2, "panic":3},
    "disgust": {"disgusting":3, "gross":2.5, "repulsive":3, "sickening":2.5, "vile":3},
    "surprise": {"shocked":2.5, "amazed":2, "astonished":2.5, "surprised":2, "wow":1.5},
    "shame": {"ashamed":3, "embarrassed":2.5, "guilty":3, "humiliated":3, "regret":2.5},
    "pride": {"proud":3, "accomplished":2.5, "honored":2.5, "grateful":2, "thankful":2},
    "amplifiers": ["very", "extremely", "really", "so", "totally", "absolutely", "incredibly", "deeply"],
    "negators": ["not", "never", "no", "don't", "didn't", "isn't", "wasn't", "can't", "won't"],
}

# ============================================================
# 1. TEXT EXTRACTION
# ============================================================
def extract_text_from_file(file_bytes: bytes, file_type: str) -> str:
    text = ""
    try:
        if file_type == 'pdf':
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        elif file_type == 'docx':
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([p.text for p in doc.paragraphs])
        elif file_type in ['txt', 'text']:
            text = file_bytes.decode('utf-8')
    except Exception as e:
        return f"ERROR: {str(e)}"
    return " ".join(text.split())

def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

# ============================================================
# 2. RULE-BASED SCORER (Improved)
# ============================================================
class EkmanRuleScorer:
    def __init__(self):
        self.lex = EKMAN_LEXICON
        self.amplifiers = set(self.lex["amplifiers"])
        self.negators = set(self.lex["negators"])

    def score_sentence(self, sentence: str) -> Tuple[Dict[str, float], List[str]]:
        words = re.findall(r'\b\w+\b', sentence.lower())
        scores = {e: 0.0 for e in EKMAN_EMOTIONS}
        triggers = []

        for i, word in enumerate(words):
            context = words[max(0, i-4):i+4]
            neg = any(n in context for n in self.negators)
            amp = any(a in context for a in self.amplifiers)

            for emo, word_dict in [(e, self.lex[e]) for e in EKMAN_EMOTIONS if e in self.lex]:
                for keyword, weight in word_dict.items():
                    if keyword in word or word.startswith(keyword):
                        score = weight
                        if amp: score *= 1.6
                        if neg: score = -score * 0.7
                        scores[emo] += score
                        triggers.append(f"{word}→{emo}")
                        break
        total = sum(abs(v) for v in scores.values())
        if total > 0:
            scores = {k: max(0, v/total) for k,v in scores.items()}
        return scores, list(set(triggers))

# ============================================================
# 3. TRANSFORMER MODEL (Robust Loading)
# ============================================================
@st.cache_resource(show_spinner="Loading emotion AI model...")
def load_transformer_model():
    try:
        classifier = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1,
            function_to_apply="sigmoid"
        )
        return classifier
    except Exception as e:
        st.error(f"Failed to load model: {e}. Using rule-based only.")
        return None

# ============================================================
# 4. MAIN ANALYSIS ENGINE
# ============================================================
class AffectiveAnalyzer:
    def __init__(self):
        self.rule_scorer = EkmanRuleScorer()
        self.transformer = load_transformer_model()

    def analyze(self, sentences: List[str], student_id: str, filename: str):
        results = []
        rule_outputs = [self.rule_scorer.score_sentence(s) for s in sentences]

        # Transformer batch prediction
        transformer_scores = []
        if self.transformer and sentences:
            with st.spinner(f"Analyzing {len(sentences)} sentences with AI..."):
                preds = self.transformer(sentences, batch_size=32)
                for pred in preds:
                    scores = {item["label"]: item["score"] for item in pred}
                    transformer_scores.append(scores)
        else:
            transformer_scores = [{} for _ in sentences]

        for i, sent in enumerate(sentences):
            t_scores = transformer_scores[i] if i < len(transformer_scores) else {}
            r_scores, r_triggers = rule_outputs[i]

            # Map GO to Ekman
            ekman_from_t = {e: 0.0 for e in EKMAN_EMOTIONS}
            for go_label, score in t_scores.items():
                ek = GO_EMOTIONS_TO_EKMAN.get(go_label, "neutral")
                if ek in ekman_from_t:
                    ekman_from_t[ek] += score

            # Fuse: 65% transformer + 35% rule (if transformer available)
            weight_t = 0.65 if self.transformer else 0.0
            weight_r = 1.0 - weight_t
            fused = {}
            for e in EKMAN_EMOTIONS:
                fused[e] = (ekman_from_t.get(e, 0) * weight_t) + (r_scores.get(e, 0) * weight_r)
            fused_sum = sum(fused.values())
            if fused_sum > 0:
                fused = {k: v / fused_sum for k, v in fused.items()}

            primary = max(fused, key=fused.get) if fused_sum > 0 else "neutral"
            max_score = fused.get(primary, 0)
            is_critical = any(fused.get(e, 0) > st.session_state.critical_threshold for e in CRITICAL_EMOTIONS)

            results.append({
                "student_id": student_id,
                "filename": filename,
                "sentence": sent,
                "primary_emotion": primary,
                "confidence": max_score,
                "is_critical": is_critical,
                "fused_scores": fused,
                "rule_triggers": ", ".join(r_triggers) if r_triggers else "None",
                "anger": fused.get("anger", 0),
                "fear": fused.get("fear", 0),
                "sadness": fused.get("sadness", 0),
                "joy": fused.get("joy", 0),
            })
        return results

# ============================================================
# 5. VISUALIZATIONS
# ============================================================
def plot_emotion_timeline(df):
    if df.empty: return None
    df['idx'] = range(len(df))
    emotions = ['anger', 'fear', 'sadness', 'joy', 'pride']
    fig = go.Figure()
    for emo in emotions:
        fig.add_trace(go.Scatter(x=df['idx'], y=df[emo], mode='lines+markers', name=emo.capitalize()))
    fig.update_layout(title="Emotion Flow Over Sentences", xaxis_title="Sentence #", yaxis_title="Intensity")
    return fig

def generate_wordcloud(df):
    text = " ".join(df[df['is_critical']]['sentence'])
    if not text:
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

# ============================================================
# 6. MAIN APP
# ============================================================
def main():
    st.set_page_config(page_title="Affective Analyzer Pro", layout="wide")
    st.title("🧠 Affective Assignment Analyzer Pro")
    st.markdown("Detect emotional tone in student submissions • Flag concerning content • Support mental health awareness")

    # Initialize
    if "critical_threshold" not in st.session_state:
        st.session_state.critical_threshold = 0.38

    analyzer = AffectiveAnalyzer()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.session_state.critical_threshold = st.slider("Critical Emotion Threshold", 0.2, 0.7, st.session_state.critical_threshold, 0.02)
        st.markdown(f"**Current:** {st.session_state.critical_threshold}")

        st.header("📤 Upload Files")
        uploaded_files = st.file_uploader(
            "Upload student assignments",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="You can assign different student IDs to each file"
        )

        st.header("✍️ Or Paste Text")
        pasted_text = st.text_area("Paste assignment text", height=150)
        default_id = "S001"
        if uploaded_files:
            student_id = st.text_input("Default Student ID", value=default_id)
        else:
            student_id = st.text_input("Student ID", value=default_id)

        run = st.button("🚀 Analyze Submissions", type="primary", use_container_width=True)

    if not run:
        st.info("Upload files or paste text → Click **Analyze Submissions**")
        return

    all_results = []
    progress_bar = st.progress(0)

    # Process pasted text
    if pasted_text.strip():
        sents = split_into_sentences(pasted_text)
        if sents:
            results = analyzer.analyze(sents, student_id, "Pasted Text")
            all_results += results
        progress_bar.progress(0.3)

    # Process uploaded files
    if uploaded_files:
        for idx, file in enumerate(uploaded_files):
            file_id = st.text_input(f"Student ID for {file.name}", value=student_id, key=f"id_{idx}")
            bytes_data = file.read()
            ext = file.name.split(".")[-1].lower()
            text = extract_text_from_file(bytes_data, ext)
            if text.startswith("ERROR"):
                st.error(f"Failed to read {file.name}: {text}")
                continue
            sents = split_into_sentences(text)
            if not sents:
                st.warning(f"No sentences found in {file.name}")
                continue
            results = analyzer.analyze(sents, file_id, file.name)
            all_results += results
            progress_bar.progress((idx + 1) / len(uploaded_files))

    if not all_results:
        st.error("No valid text found.")
        return

    df = pd.DataFrame(all_results)
    progress_bar.empty()

    # Student filter
    students = df["student_id"].unique()
    selected_student = st.selectbox("Filter by Student", ["All Students"] + list(students))

    if selected_student != "All Students":
        df = df[df["student_id"] == selected_student]

    # Summary
    st.success(f"Analysis Complete • {len(df)} sentences • {df['is_critical'].sum()} critical flags")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Sentences", len(df))
    with col2: st.metric("Dominant Emotion", df["primary_emotion"].mode()[0])
    with col3: st.metric("Critical Flags", df["is_critical"].sum())
    with col4: st.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")

    # Charts
    tab1, tab2, tab3, tab4 = st_tabs(["Overview", "Emotion Flow", "Critical Sentences", "Export"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            avg_scores = pd.DataFrame(df["fused_scores"].tolist()).mean().sort_values(ascending=False)
            fig = px.bar(x=avg_scores.index, y=avg_scores.values, title="Average Emotion Intensity", color=avg_scores.values)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pie = px.pie(df, names="primary_emotion", title="Primary Emotion Distribution")
            st.plotly_chart(pie, use_container_width=True)

        st.plotly_chart(plot_emotion_timeline(df), use_container_width=True)

    with tab2:
        critical = df[df["is_critical"]]
        if not critical.empty:
            st.warning(f"Found {len(critical)} potentially concerning sentences")
            for _, row in critical.iterrows():
                st.error(f"**{row['primary_emotion'].upper()}** ({row['confidence']:.1%}): {row['sentence'][:200]}...")
        else:
            st.success("No critical emotional content detected.")

        wc = generate_wordcloud(df)
        if wc:
            st.pyplot(wc)

    with tab3:
        st.dataframe(df.drop(columns=["fused_scores"]), use_container_width=True, height=600)

    with tab4:
        csv = df.to_csv(index=False)
        excel = io.BytesIO()
        df.to_excel(excel, index=False, engine='openpyxl')
        excel.seek(0)

        st.download_button("📄 Download CSV", csv, "emotion_analysis.csv", "text/csv")
        st.download_button("📊 Download Excel", excel.read(), "emotion_analysis.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()

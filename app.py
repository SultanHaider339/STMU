# main_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import io
import json
import torch
import pypdf
import docx
import re
from typing import List, Dict, Any, Union
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
#                0. CONFIG + CONSTANTS
# ============================================================

EKMAN_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'shame', 'pride']

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
    "anger": ["furious", "hate", "enraged", "mad", "frustrat", "irritat", "disgusting", "disapprov"],
    "joy": ["happy", "delight", "wonderful", "great", "excit", "love", "cheer", "optimis"],
    "sadness": ["sad", "depress", "grief", "remorse", "miserable", "sorrow", "disappoint"],
    "fear": ["scared", "afraid", "terrified", "nervous", "anxiety", "worried"],
    "disgust": ["gross", "nauseating", "repulse", "sickening", "yuck", "contempt"],
    "surprise": ["shock", "astound", "confused", "curiosity", "realiz"],
    "shame": ["ashamed", "embarrass", "guilty", "humiliat", "remorse"],
    "pride": ["proud", "accomplish", "admire", "caring", "gratitude", "approval", "optimism"],
    "amplifiers": ["very", "extremely", "so", "totally", "absolutely", "greatly", "much"],
    "negators": ["not", "never", "no", "don't", "didn't", "isn't", "couldn't", "wasn't"]
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
    return [s.strip() for s in sentences if s.strip()]

# ============================================================
#            2. RULE-BASED EMOTION ENGINE
# ============================================================

class EkmanRuleScorer:
    def __init__(self, lexicon: Dict[str, Any]):
        self.lexicon = lexicon
        self.amplifiers = [w.lower() for w in lexicon["amplifiers"]]
        self.negators = [w.lower() for w in lexicon["negators"]]
        self.emotion_words = {
            e: [w.lower() for w in lexicon[e]] for e in EKMAN_EMOTIONS
        }

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
            if sc:
                primary = max(sc, key=sc.get)
            else:
                primary = "neutral"

            output.append({
                "sentence": s,
                "system": "Rule-Based",
                "primary_emotion": primary,
                "scores": sc,
                "explanation": f"Keywords: {', '.join(trig)}" if trig else "No emotional cues detected."
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

            classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=DEVICE,
                return_all_scores=True,
                function_to_apply="sigmoid",
            )
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

            out.append({
                "sentence": sentences[i],
                "system": "Transformer",
                "primary_emotion": primary,
                "scores": sc,
                "explanation": f"Highest prob: {sc[primary]:.4f}"
            })
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

        fused = {}
        for e in EKMAN_EMOTIONS:
            fused[e] = mapped.get(e, 0) * 0.7 + rule_scores.get(e, 0) * 0.3

        s = sum(fused.values())
        if s:
            fused = {k: v / s for k, v in fused.items()}

        return fused

    def analyze_submission(self, sents, filename, student_id):
        m = self.transformer_model.batch_analyze(sents)
        r = self.rule_scorer.analyze_text(sents)

        out = []
        for mm, rr in zip(m, r):

            fused = self.fuse_scores(mm["scores"], rr["scores"])
            primary_fused = max(fused, key=fused.get)

            is_critical = any(
                fused[x] > CRITICAL_THRESHOLD for x in CRITICAL_EMOTIONS
            )

            out.append({
                "student_id": student_id,
                "filename": filename,
                "sentence": mm["sentence"],

                "model_primary": mm["primary_emotion"],
                "rule_primary": rr["primary_emotion"],

                "fused_primary": primary_fused,
                "fused_max_score": fused[primary_fused],

                "is_critical": is_critical,
                "model_scores_28": mm["scores"],
                "rule_scores_8": rr["scores"],
                "fused_scores_8": fused,

                "model_explanation": mm["explanation"],
                "rule_explanation": rr["explanation"],
            })
        return out

# ============================================================
#                 5. STREAMLIT USER INTERFACE
# ============================================================

@st.cache_resource
def get_analysis_engine():
    return AnalysisEngine()

def generate_sentence_df(records):
    df = pd.DataFrame(records)
    if not df.empty:
        df["is_critical"] = df["is_critical"].astype(bool)
        df["fused_max_score"] = df["fused_max_score"].astype(float)
    return df

def create_emotion_bar_chart(df, title):
    if df.empty:
        return px.bar(title="No data")

    agg = pd.DataFrame(df["fused_scores_8"].tolist()).mean().reset_index()
    agg.columns = ["Emotion", "Average Score"]

    fig = px.bar(agg, x="Emotion", y="Average Score", title=title,
                 color="Average Score", height=400)
    fig.update_layout(xaxis={'categoryorder': 'total descending'})
    return fig

def create_primary_distribution_chart(df, title):
    if df.empty:
        return px.pie(names=["None"], values=[1], title="Empty")

    c = df["fused_primary"].value_counts().reset_index()
    c.columns = ["Emotion", "Count"]

    fig = px.pie(c, names="Emotion", values="Count", title=title, hole=.3)
    return fig

# ============================================================
#                    MAIN STREAMLIT APP
# ============================================================

def main():

    st.set_page_config(layout="wide", page_title="Affective Analyzer")
    st.title("🧠 Affective Assignment Analyzer")

    engine = get_analysis_engine()

    with st.sidebar:
        st.header("Input Method")
        uploaded_files = st.file_uploader("Upload PDF/DOCX/TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        pasted_text = st.text_area("Or Paste Text Here", height=250)
        student_id = st.text_input("Student ID", value="S001")
        run_btn = st.button("🚀 Run Analysis")

    if not run_btn:
        return

    all_results = []

    # Pasted Text
    if pasted_text.strip():
        sents = preprocess_text(pasted_text)
        all_results += engine.analyze_submission(sents, "Pasted_Text", student_id)

    # Uploaded Files
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

    if not all_results:
        st.warning("No sentences found.")
        return

    df = generate_sentence_df(all_results)

    # --------------------------------------------------------
    #             SUMMARY METRICS
    # --------------------------------------------------------
    st.header("📊 Summary Report")

    total = len(df)
    critical = df["is_critical"].sum()
    avg_conf = df["fused_max_score"].mean() * 100
    mode_emo = df["fused_primary"].mode()[0] if not df.empty else "Neutral"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sentences", total)
    col2.metric("Primary Emotion", mode_emo)
    col3.metric("Critical Flags", critical)
    col4.metric("Avg Confidence", f"{avg_conf:.1f}%")

    st.markdown("---")

    # --------------------------------------------------------
    #             VISUAL GRAPHS
    # --------------------------------------------------------
    st.subheader("Emotion Score Visualizations")

    c1, c2 = st.columns(2)
    c1.plotly_chart(
        create_emotion_bar_chart(df, "Average Fused Emotion Scores"),
        use_container_width=True
    )
    c2.plotly_chart(
        create_primary_distribution_chart(df, "Primary Emotion Distribution"),
        use_container_width=True
    )

    # --------------------------------------------------------
    #             FULL TABLE
    # --------------------------------------------------------
    st.subheader("Detailed Per-Sentence Analysis")
    st.dataframe(df, use_container_width=True)

    # --------------------------------------------------------
    #             EXPORT
    # --------------------------------------------------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, "emotion_report.csv", "text/csv")


if __name__ == "__main__":
    main()

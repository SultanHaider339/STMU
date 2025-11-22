# main_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import io
import time
import json
import torch
import pypdf
import docx
import re
from typing import List, Dict, Any, Union
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# --- 0. Configuration and Utilities ---

EKMAN_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'shame', 'pride']

# Mapping the 28 GoEmotions labels to the 8 Ekman-inspired emotions (Depth/Breadth)
GO_EMOTIONS_TO_EKMAN: Dict[str, str] = {
    'admiration': 'pride', 'amusement': 'joy', 'anger': 'anger', 'annoyance': 'anger',
    'approval': 'pride', 'caring': 'pride', 'confusion': 'surprise', 'curiosity': 'surprise',
    'desire': 'joy', 'disappointment': 'sadness', 'disapproval': 'anger', 'disgust': 'disgust',
    'embarrassment': 'shame', 'excitement': 'joy', 'fear': 'fear', 'gratitude': 'pride',
    'grief': 'sadness', 'joy': 'joy', 'love': 'joy', 'nervousness': 'fear',
    'optimism': 'pride', 'pride': 'pride', 'realization': 'surprise', 'relief': 'joy',
    'remorse': 'shame', 'sadness': 'sadness', 'surprise': 'surprise', 'neutral': 'neutral',
}

# Critical Emotion Threshold (Relevance/Significance)
CRITICAL_THRESHOLD = 0.35 
CRITICAL_EMOTIONS = ['anger', 'fear', 'sadness']

# In-Script Mock Lexicon (Rule-Based Logic/Explanation)
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


# --- 1. Text Extraction and Preprocessing ---

def extract_text_from_file(file_path: Union[str, io.BytesIO], file_type: str) -> str:
    """Extracts text from a given file stream based on its type."""
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
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                text = file_path.read().decode('utf-8')
        except Exception as e:
            return f"ERROR_TXT_EXTRACTION: {e}"
    else:
        return f"ERROR_UNSUPPORTED_TYPE: {file_type}"

    text = " ".join(text.split()).strip()
    return text

def preprocess_text(text: str) -> List[str]:
    """Tokenizes text into sentences."""
    # Split text by common sentence-ending punctuation followed by a space
    sentences = re.split(r'(?<=[.?!])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# --- 2. Rule-Based Ekman Scorer (Logic/Explanation) ---

class EkmanRuleScorer:
    """Implements the rule-based system for 8 Ekman-inspired emotions."""
    def __init__(self, lexicon: Dict[str, Any]):
        self.lexicon = lexicon
        self.amplifiers = [w.lower() for w in self.lexicon.get('amplifiers', [])]
        self.negators = [w.lower() for w in self.lexicon.get('negators', [])]
        self.emotion_words = {e: [w.lower() for w in self.lexicon.get(e, [])] for e in EKMAN_EMOTIONS}

    def _score_token(self, word: str, surrounding_words: List[str]) -> Dict[str, float]:
        scores = {e: 0.0 for e in EKMAN_EMOTIONS}
        triggered_emotions = []

        for emotion, keywords in self.emotion_words.items():
            # Check if the token is an emotional keyword
            if any(word.startswith(k) for k in keywords):
                score = 1.0
                triggered_emotions.append(emotion)
                
                # Apply Amplifier boost
                if any(amp in surrounding_words for amp in self.amplifiers):
                    score *= 1.5
                
                # Apply Negator reduction/inversion
                if any(neg in surrounding_words for neg in self.negators):
                    score *= -0.8
                    
                scores[emotion] += score
                
        return scores, triggered_emotions

    def score_sentence(self, sentence: str) -> Dict[str, float]:
        words = sentence.lower().split()
        cumulative_scores = {e: 0.0 for e in EKMAN_EMOTIONS}
        all_triggered_words = set()
        
        for i, word in enumerate(words):
            # Context window for checking amplifiers/negators (3 words before)
            context_window = words[max(0, i-3):i]
            token_scores, triggered_emotions = self._score_token(word, context_window)
            
            if triggered_emotions:
                all_triggered_words.add(word)
            
            for emotion, score in token_scores.items():
                cumulative_scores[emotion] += score

        # Normalize scores to sum to 1.0 (or 0 if all are zero)
        total_score_magnitude = sum(abs(s) for s in cumulative_scores.values())
        if total_score_magnitude > 0:
            normalized_scores = {e: max(0.0, abs(s) / total_score_magnitude) for e, s in cumulative_scores.items()}
            # Return normalized scores and the words that triggered them (for Logic/Explanation)
            return normalized_scores, all_triggered_words
        
        return {e: 0.0 for e in EKMAN_EMOTIONS}, all_triggered_words

    def analyze_text(self, sentences: List[str]) -> List[Dict[str, Any]]:
        results = []
        
        for sent in sentences:
            sent_scores, triggered_words = self.score_sentence(sent)
            
            # Find the primary emotion
            primary = 'neutral'
            max_score = 0.0
            if sent_scores:
                primary, max_score = max(sent_scores.items(), key=lambda item: item[1])
                if max_score == 0.0:
                    primary = 'neutral'

            results.append({
                'sentence': sent,
                'system': 'Rule-Based',
                'primary_emotion': primary,
                'scores': sent_scores,
                # Rule Explanation provides Logic and Evidence
                'explanation': f"Keywords: {', '.join(set(triggered_words))}. Max Score: {max_score:.4f}" if triggered_words else 'No explicit emotional cues found.'
            })
        return results

# --- 3. Transformer Model (Accuracy/Depth) ---

MODEL_NAME = "SamLowe/roberta-base-go_emotions"
DEVICE = 0 if torch.cuda.is_available() else -1

class TransformerEmotionModel:
    """Handles loading and inference for the Hugging Face transformer model."""
    
    def __init__(self):
        # Cache model loading to optimize performance
        self.classifier = self._load_model()
        self.id2label = self.classifier.model.config.id2label if self.classifier else {}

    @st.cache_resource
    def _load_model(self):
        try:
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=DEVICE,
                return_all_scores=True,
                function_to_apply='sigmoid' # Use sigmoid for multi-label prediction (GoEmotions)
            )
            return classifier
        except Exception as e:
            st.error(f"Error loading Transformer model. This is critical for Accuracy and Depth. {e}")
            return None

    def batch_analyze(self, sentences: List[str]) -> List[Dict[str, Any]]:
        if not self.classifier or not sentences:
            return [{
                'sentence': s, 'system': 'Transformer', 'primary_emotion': 'neutral',
                'scores': {label: 0.0 for label in self.id2label.values()},
                'explanation': 'Model not loaded.'
            } for s in sentences]
            
        raw_predictions = self.classifier(sentences, batch_size=16)
        
        results = []
        for i, pred_list in enumerate(raw_predictions):
            scores = {item['label']: item['score'] for item in pred_list}
            # Find the primary emotion (highest probability)
            primary_emotion = max(scores, key=scores.get) if scores else 'neutral'
            
            results.append({
                'sentence': sentences[i], 'system': 'Transformer',
                'primary_emotion': primary_emotion, 'scores': scores,
                'explanation': f"Highest probability: {scores.get(primary_emotion, 0.0):.4f}"
            })
            
        return results


# --- 4. Analysis Engine and Fusion (Fairness/Logic) ---

class AnalysisEngine:
    """Integrates both emotion detection systems and performs score fusion."""
    def __init__(self):
        self.rule_scorer = EkmanRuleScorer(MOCK_EKMAN_LEXICON) 
        self.transformer_model = TransformerEmotionModel()
        self.ekman_labels = EKMAN_EMOTIONS

    def fuse_scores(self, model_scores: Dict[str, float], rule_scores: Dict[str, float]) -> Dict[str, float]:
        """Fuses the two score vectors using a 70/30 weighted average (Logic/Fairness)."""
        fused_scores = {e: 0.0 for e in self.ekman_labels}
        transformer_weight = 0.7 # Giving higher weight to the data-driven model (Fairness)
        rule_weight = 0.3

        # 1. Map Transformer scores (28 labels) to 8 Ekman labels (Depth)
        mapped_model_scores = {e: 0.0 for e in self.ekman_labels}
        for go_label, score in model_scores.items():
            ekman_label = GO_EMOTIONS_TO_EKMAN.get(go_label)
            if ekman_label in mapped_model_scores:
                mapped_model_scores[ekman_label] += score

        # Re-normalize mapped model scores
        total_mapped = sum(mapped_model_scores.values())
        if total_mapped > 0:
            mapped_model_scores = {k: v / total_mapped for k, v in mapped_model_scores.items()}
        
        # 2. Weighted Average Fusion
        for ekman_label in self.ekman_labels:
            fused_scores[ekman_label] = (
                mapped_model_scores.get(ekman_label, 0.0) * transformer_weight +
                rule_scores.get(ekman_label, 0.0) * rule_weight
            )
            
        # 3. Final normalization (Accuracy)
        total_fused = sum(fused_scores.values())
        if total_fused > 0:
            fused_scores = {k: v / total_fused for k, v in fused_scores.items()}
        
        return fused_scores

    def analyze_submission(self, sentences: List[str], filename: str, student_id: str) -> List[Dict[str, Any]]:
        """Runs the end-to-end analysis."""
        model_results = self.transformer_model.batch_analyze(sentences)
        rule_results = self.rule_scorer.analyze_text(sentences)

        final_analysis = []
        min_len = min(len(model_results), len(rule_results))
        
        for m_res, r_res in zip(model_results[:min_len], rule_results[:min_len]):
            fused_scores = self.fuse_scores(m_res['scores'], r_res['scores'])
            
            # Identify Fused Primary (Precision)
            primary_fused = max(fused_scores, key=fused_scores.get)
            fused_max_score = fused_scores.get(primary_fused, 0.0)
            
            # Check for critical anomaly (Relevance/Significance)
            is_critical = False
            for emotion in CRITICAL_EMOTIONS:
                 if fused_scores.get(emotion, 0.0) > CRITICAL_THRESHOLD:
                    is_critical = True
                    break

            final_analysis.append({
                'student_id': student_id, 'filename': filename, 'sentence': m_res['sentence'],
                'model_primary': m_res['primary_emotion'], 'rule_primary': r_res['primary_emotion'],
                'fused_primary': primary_fused, 'fused_max_score': fused_max_score,
                'is_critical': is_critical,
                'model_scores_28': m_res['scores'],
                'rule_scores_8': r_res['scores'], 'fused_scores_8': fused_scores,
                'model_explanation': m_res['explanation'], 'rule_explanation': r_res['explanation'],
            })
            
        return final_analysis

    def export_to_csv(self, data: List[Dict[str, Any]], path: Any):
        """Converts the analysis results into a clean DataFrame and exports to CSV stream (Accuracy)."""
        records = []
        for item in data:
            record = {
                'student_id': item['student_id'], 'filename': item['filename'], 'sentence': item['sentence'],
                'fused_primary': item['fused_primary'],
                'fused_max_score': f"{item['fused_max_score']:.4f}",
                'model_primary': item['model_primary'], 
                'rule_primary': item['rule_primary'],
                'is_critical_flag': 'YES' if item['is_critical'] else 'NO',
            }
            # Add all 8 fused scores
            for emotion, score in item['fused_scores_8'].items():
                record[f'fused_{emotion}_score'] = f'{score:.4f}'
            
            # Add explanations (Logic)
            record['rule_explanation'] = item['rule_explanation']
            record['model_explanation'] = item['model_explanation']
            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(path, index=False)


# --- 5. Streamlit Application (Clarity, Significance, Breadth) ---

@st.cache_resource
def get_analysis_engine():
    """Load the analysis engine (and the transformer model) once."""
    return AnalysisEngine()

def generate_sentence_df(analysis_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Creates a DataFrame for internal use."""
    # Ensure this DF contains all data for charts/tables
    return pd.DataFrame(analysis_results)

def create_emotion_bar_chart(df: pd.DataFrame, title: str):
    """Creates a comparison bar chart of average fused scores (Significance)."""
    if df.empty:
        return px.bar(title="No data to display.")
        
    fused_scores_list = [res for res in df['fused_scores_8']]
    aggregated_scores = pd.DataFrame(fused_scores_list).mean().reset_index()
    aggregated_scores.columns = ['Emotion', 'Average Score']
    
    fig = px.bar(
        aggregated_scores, x='Emotion', y='Average Score', 
        title=f'⚖️ {title}', color='Average Score',
        color_continuous_scale=px.colors.sequential.Plasma, height=400
    )
    # Order by score descending (Significance)
    fig.update_layout(xaxis={'categoryorder':'total descending'}, coloraxis_showscale=False) 
    return fig

def main():
    st.set_page_config(layout="wide", page_title="Affective Assignment Analyzer", initial_sidebar_state="expanded")
    st.title("🧠 Affective Assignment Analyzer")
    st.subheader("Applying Paul's Standards of Critical Thinking to Emotional Analysis")

    try:
        engine = get_analysis_engine()
    except Exception as e:
        st.error(f"Failed to initialize the analysis engine: {e}")
        return

    # Sidebar for Input
    with st.sidebar:
        st.header("Input & Parameters")
        st.info("The fusion engine uses 70% Transformer (Depth/Accuracy) and 30% Rule-Based (Logic/Explanation).")
        uploaded_files = st.file_uploader(
            "Upload Assignment Files (PDF, DOCX, TXT)", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )
        pasted_text = st.text_area("Or Paste Text Directly Here", height=300)
        student_id_input = st.text_input("Student/Submission ID", value="S001")
        
        process_button = st.button("🚀 Run Critical Analysis", type="primary")

    if process_button:
        all_results = []
        
        # --- Processing Logic ---
        if pasted_text.strip():
            with st.spinner("Analyzing pasted text..."):
                sentences = preprocess_text(pasted_text)
                results = engine.analyze_submission(sentences, "Pasted_Text", student_id_input)
                all_results.extend(results)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_type = uploaded_file.name.split('.')[-1]
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    file_stream = io.BytesIO(uploaded_file.read())
                    text = extract_text_from_file(file_stream, file_type)
                    
                    if text.startswith("ERROR"):
                        st.error(f"Error for {uploaded_file.name}: {text}")
                        continue
                    
                    sentences = preprocess_text(text)
                    results = engine.analyze_submission(sentences, uploaded_file.name, student_id_input)
                    all_results.extend(results)

        # --- Display Results based on Critical Thinking Standards ---
        if all_results:
            st.success("Analysis Complete: Results adhere to intellectual standards.")
            final_df = generate_sentence_df(all_results)
            
            # 1. Overall Dashboard (Significance, Relevance, Clarity)
            st.header("📊 Submission Summary (Significance & Relevance)")
            
            # Calculate key metrics
            fused_mode = final_df['fused_primary'].mode()
            top_fused = fused_mode[0].title() if not fused_mode.empty else 'Neutral'
            critical_count = final_df['is_critical'].sum()
            total_sentences = len(final_df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Primary Emotional Theme (Fused)", 
                    value=top_fused,
                    help="The most frequently detected primary emotion across all sentences."
                )
            with col2:
                # Relevance Flag
                flag_emoji = "🚨" if critical_count > 0 else "🟢"
                st.metric(
                    label="Critical Anomaly Sentences", 
                    value=f"{flag_emoji} {critical_count} / {total_sentences}",
                    help=f"Sentences where Anger, Fear, or Sadness fused score exceeds {CRITICAL_THRESHOLD:.2f} (Relevance/Significance flag)."
                )
            with col3:
                 # Accuracy & Precision Metric
                 avg_confidence = final_df['fused_max_score'].mean() * 100
                 st.metric(
                    label="Avg. Primary Confidence (Precision)", 
                    value=f"{avg_confidence:.1f}%",
                    help="Average confidence score (probability) of the Primary Fused Emotion across the document."
                )

            st.plotly_chart(create_emotion_bar_chart(final_df, "Average Fused Emotion Scores (Ordered by Significance)"), use_container_width=True)
            
            # 2. Sentence-by-Sentence Breakdown (Breadth, Logic, Precision, Accuracy)
            st.header("📜 Sentence-Level Analysis (Breadth & Logic)")
            
            for index, row in final_df.iterrows():
                
                # Breadth Check: Check for disagreement between systems mapped to Ekman
                model_mapped = GO_EMOTIONS_TO_EKMAN.get(row['model_primary'], 'neutral')
                rule_primary = row['rule_primary']
                disagreement_flag = "❗️ DIVERGENCE" if model_mapped != rule_primary else ""
                
                # Clarity & Significance for the Expander title
                title_emoji = "🚨" if row['is_critical'] else ""
                expander_title = (
                    f"**{index+1}. {row['fused_primary'].title()}** ({row['fused_max_score']:.2f}) | "
                    f"{title_emoji} {disagreement_flag} | {row['sentence'][:90]}..."
                )
                
                with st.expander(expander_title):
                    st.markdown(f"**Sentence:** *{row['sentence']}*")
                    
                    # Columns for Dual-System Comparison (Breadth)
                    col_m, col_r, col_f = st.columns(3)
                    
                    # Model System (Depth/Accuracy)
                    with col_m:
                        st.caption("🔍 **System 1: Transformer Model** (Depth/Accuracy)")
                        st.markdown(f"**Primary (28-class):** `{row['model_primary'].title()}`")
                        st.markdown(f"**Mapped Ekman:** `{model_mapped.title()}`")
                        # Display scores (Precision)
                        scores_model = {k: f"{v:.4f}" for k, v in sorted(row['model_scores_28'].items(), key=lambda item: item[1], reverse=True)[:5]}
                        st.json(scores_model)

                    # Rule-Based System (Logic/Explanation)
                    with col_r:
                        st.caption("⚖️ **System 2: Rule-Based Lexicon** (Logic/Explanation)")
                        st.markdown(f"**Primary (8-class):** `{rule_primary.title()}`")
                        st.markdown(f"**Logic/Evidence:** {row['rule_explanation']}")
                        # Display scores (Precision)
                        scores_rule = {k: f"{v:.4f}" for k, v in sorted(row['rule_scores_8'].items(), key=lambda item: item[1], reverse=True)[:3]}
                        st.json(scores_rule)
                    
                    # Fused Result (Clarity/Fairness)
                    with col_f:
                        st.caption("🤝 **Final Fused Score** (Clarity/Fairness)")
                        st.markdown(f"**Fused Primary:** **`{row['fused_primary'].title()}`**")
                        st.markdown(f"**Confidence:** **`{row['fused_max_score']:.4f}`**")
                        # Display scores (Precision)
                        scores_fused = {k: f"{v:.4f}" for k, v in sorted(row['fused_scores_8'].items(), key=lambda item: item[1], reverse=True)}
                        st.json(scores_fused)

            # 3. Export Functionality
            csv_buffer = io.StringIO()
            engine.export_to_csv(all_results, csv_buffer)
            st.download_button(
                label="⬇️ Download Full Analysis (CSV for Accuracy Review)",
                data=csv_buffer.getvalue().encode('utf-8'),
                file_name=f"critical_emotion_analysis_{student_id_input}_{int(time.time())}.csv",
                mime="text/csv",
                help="Exports all primary classifications and the 8 fused scores for external review."
            )
            
        elif process_button:
            st.warning("No valid text was entered or extracted. Please provide text or upload a file.")
            
    if not process_button:
        st.subheader("How This App Applies Critical Thinking Standards")
        st.markdown("""
        The analysis process is intentionally dual-layered to meet intellectual standards:
        * **Breadth & Depth:** We use two different systems: a complex 28-class Transformer (AI) and a simpler 8-class Rule-Based (Lexicon). This provides multiple viewpoints.
        * **Logic & Explanation:** The Rule-Based system provides explicit keywords and context (our logic/evidence) for its classification, making the result traceable.
        * **Clarity & Precision:** All final scores are normalized and displayed with four decimal places for exactness.
        * **Significance & Relevance:** We highlight "Critical Anomaly Sentences" (Anger, Fear, Sadness) to guide teacher attention to the most important areas of student welfare.
        """)

if __name__ == "__main__":
    main()

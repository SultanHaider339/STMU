# main_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import io
import time
import json
from typing import List, Dict, Any, Union
import torch
import pypdf
import docx
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# --- 0. Configuration and Utilities ---

EKMAN_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'shame', 'pride']

# Mapping the 28 GoEmotions labels to the 8 Ekman-inspired emotions
GO_EMOTIONS_TO_EKMAN: Dict[str, str] = {
    'admiration': 'pride', 'amusement': 'joy', 'anger': 'anger', 'annoyance': 'anger',
    'approval': 'pride', 'caring': 'pride', 'confusion': 'surprise', 'curiosity': 'surprise',
    'desire': 'joy', 'disappointment': 'sadness', 'disapproval': 'anger', 'disgust': 'disgust',
    'embarrassment': 'shame', 'excitement': 'joy', 'fear': 'fear', 'gratitude': 'pride',
    'grief': 'sadness', 'joy': 'joy', 'love': 'joy', 'nervousness': 'fear',
    'optimism': 'pride', 'pride': 'pride', 'realization': 'surprise', 'relief': 'joy',
    'remorse': 'shame', 'sadness': 'sadness', 'surprise': 'surprise', 'neutral': 'neutral',
}

# In-Script Mock Lexicon (REPLACING the external JSON file)
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
    sentences = re.split(r'(?<=[.?!])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# --- 2. Rule-Based Ekman Scorer ---

class EkmanRuleScorer:
    """Implements the rule-based system for 8 Ekman-inspired emotions."""
    def __init__(self, lexicon: Dict[str, Any]):
        self.lexicon = lexicon
        self.amplifiers = [w.lower() for w in self.lexicon.get('amplifiers', [])]
        self.negators = [w.lower() for w in self.lexicon.get('negators', [])]
        self.emotion_words = {e: [w.lower() for w in self.lexicon.get(e, [])] for e in EKMAN_EMOTIONS}

    def _score_token(self, word: str, surrounding_words: List[str]) -> Dict[str, float]:
        scores = {e: 0.0 for e in EKMAN_EMOTIONS}
        
        for emotion, keywords in self.emotion_words.items():
            if any(word.startswith(k) for k in keywords):
                score = 1.0
                
                if any(amp in surrounding_words for amp in self.amplifiers):
                    score *= 1.5
                
                if any(neg in surrounding_words for neg in self.negators):
                    score *= -0.8
                    
                scores[emotion] += score
                
        return scores

    def score_sentence(self, sentence: str) -> Dict[str, float]:
        words = sentence.lower().split()
        cumulative_scores = {e: 0.0 for e in EKMAN_EMOTIONS}
        
        for i, word in enumerate(words):
            context_window = words[max(0, i-3):i] # 3 words before
            token_scores = self._score_token(word, context_window)
            
            for emotion, score in token_scores.items():
                cumulative_scores[emotion] += score

        total_score_magnitude = sum(abs(s) for s in cumulative_scores.values())
        if total_score_magnitude > 0:
            return {e: max(0.0, abs(s) / total_score_magnitude) for e, s in cumulative_scores.items()}
        
        return {e: 0.0 for e in EKMAN_EMOTIONS}

    def analyze_text(self, sentences: List[str]) -> List[Dict[str, Any]]:
        results = []
        all_keywords = set(k for k in sum(self.emotion_words.values(), []))
        
        for sent in sentences:
            sent_scores = self.score_sentence(sent)
            sorted_scores = sorted(sent_scores.items(), key=lambda item: item[1], reverse=True)
            primary = sorted_scores[0][0] if sorted_scores and sorted_scores[0][1] > 0 else 'neutral'
            
            triggered_words = [w for w in sent.lower().split() if any(w.startswith(k) for k in all_keywords)]
            
            results.append({
                'sentence': sent,
                'system': 'Rule-Based',
                'primary_emotion': primary,
                'scores': sent_scores,
                'explanation': f"Triggered by: {', '.join(set(triggered_words))}" if triggered_words else 'No explicit emotional cues found.'
            })
        return results

# --- 3. Transformer Model ---

MODEL_NAME = "SamLowe/roberta-base-go_emotions"
DEVICE = 0 if torch.cuda.is_available() else -1

class TransformerEmotionModel:
    """Handles loading and inference for the Hugging Face transformer model."""
    
    def __init__(self):
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.id2label = self.model.config.id2label
            
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=DEVICE,
                return_all_scores=True,
                function_to_apply='sigmoid'
            )
        except Exception as e:
            st.error(f"Error loading transformer model. Check internet connection and VRAM. {e}")
            self.classifier = None
            self.id2label = {i: str(i) for i in range(28)}

    def batch_analyze(self, sentences: List[str]) -> List[Dict[str, Any]]:
        if not self.classifier or not sentences:
            return [{
                'sentence': s, 'system': 'Transformer', 'primary_emotion': 'error',
                'scores': {label: 0.0 for label in self.id2label.values()},
                'explanation': 'Model not loaded or input empty.'
            } for s in sentences]
            
        raw_predictions = self.classifier(sentences, batch_size=16)
        
        results = []
        for i, pred_list in enumerate(raw_predictions):
            scores = {item['label']: item['score'] for item in pred_list}
            primary_emotion = max(scores, key=scores.get) if scores else 'neutral'
            
            results.append({
                'sentence': sentences[i], 'system': 'Transformer',
                'primary_emotion': primary_emotion, 'scores': scores,
                'explanation': 'Model-based classification.'
            })
            
        return results


# --- 4. Analysis Engine and Fusion ---

class AnalysisEngine:
    """Integrates both emotion detection systems and performs score fusion."""
    def __init__(self):
        # Pass the mock lexicon directly to the scorer
        self.rule_scorer = EkmanRuleScorer(MOCK_EKMAN_LEXICON) 
        self.transformer_model = TransformerEmotionModel()
        self.ekman_labels = EKMAN_EMOTIONS

    def fuse_scores(self, model_scores: Dict[str, float], rule_scores: Dict[str, float]) -> Dict[str, float]:
        """Fuses the two score vectors using a 70/30 weighted average."""
        fused_scores = {e: 0.0 for e in self.ekman_labels}
        transformer_weight = 0.7
        rule_weight = 0.3

        # Map Transformer scores (28 labels) to 8 Ekman labels
        mapped_model_scores = {e: 0.0 for e in self.ekman_labels}
        for go_label, score in model_scores.items():
            ekman_label = GO_EMOTIONS_TO_EKMAN.get(go_label)
            if ekman_label in mapped_model_scores:
                mapped_model_scores[ekman_label] += score

        total_mapped = sum(mapped_model_scores.values())
        if total_mapped > 0:
            mapped_model_scores = {k: v / total_mapped for k, v in mapped_model_scores.items()}
        
        # Weighted Average Fusion
        for ekman_label in self.ekman_labels:
            fused_scores[ekman_label] = (
                mapped_model_scores.get(ekman_label, 0.0) * transformer_weight +
                rule_scores.get(ekman_label, 0.0) * rule_weight
            )
            
        # Final normalization
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
            primary_fused = max(fused_scores, key=fused_scores.get)

            final_analysis.append({
                'student_id': student_id, 'filename': filename, 'sentence': m_res['sentence'],
                'model_primary': m_res['primary_emotion'], 'rule_primary': r_res['primary_emotion'],
                'fused_primary': primary_fused, 'model_scores_28': m_res['scores'],
                'rule_scores_8': r_res['scores'], 'fused_scores_8': fused_scores,
                'model_explanation': m_res['explanation'], 'rule_explanation': r_res['explanation'],
            })
            
        return final_analysis

    def export_to_csv(self, data: List[Dict[str, Any]], path: Any):
        """Converts the analysis results into a clean DataFrame and exports to CSV stream."""
        records = []
        for item in data:
            record = {
                'student_id': item['student_id'], 'filename': item['filename'], 'sentence': item['sentence'],
                'model_primary': item['model_primary'], 'rule_primary': item['rule_primary'],
                'fused_primary': item['fused_primary'],
            }
            for emotion, score in item['fused_scores_8'].items():
                record[f'fused_{emotion}_score'] = f'{score:.4f}'
            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(path, index=False)


# --- 5. Streamlit Application ---

@st.cache_resource
def get_analysis_engine():
    """Load the analysis engine (and the transformer model) once."""
    return AnalysisEngine()

def generate_sentence_df(analysis_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Creates a DataFrame for sentence-level display."""
    return pd.DataFrame([
        {
            'Sentence': res['sentence'], 'Model Primary': res['model_primary'].title(),
            'Rule Primary': res['rule_primary'].title(), 'Fused Primary': res['fused_primary'].title(),
            'Model Scores (28)': res['model_scores_28'], 'Rule Scores (8)': res['rule_scores_8'],
            'Fused Scores (8)': res['fused_scores_8'], 'Model Explanation': res['model_explanation'],
            'Rule Explanation': res['rule_explanation'],
        } for res in analysis_results
    ])

def create_emotion_bar_chart(df: pd.DataFrame, title: str):
    """Creates a comparison bar chart of average fused scores."""
    if df.empty:
        return px.bar(title="No data to display.")
        
    fused_scores_list = [res['Fused Scores (8)'] for res in df.to_dict('records')]
    aggregated_scores = pd.DataFrame(fused_scores_list).mean().reset_index()
    aggregated_scores.columns = ['Emotion', 'Average Score']
    
    fig = px.bar(
        aggregated_scores, x='Emotion', y='Average Score', 
        title=f'⚖️ {title}', color='Average Score',
        color_continuous_scale=px.colors.sequential.Plasma, height=400
    )
    fig.update_layout(xaxis={'categoryorder':'total descending'}, coloraxis_showscale=False)
    return fig


def main():
    st.set_page_config(layout="wide", page_title="Affective Assignment Analyzer")
    st.title("🧠 Affective Assignment Analyzer")
    st.markdown("Dual-system emotion detection for student submissions using **Transformer** and **Rule-Based Ekman** approaches.")

    try:
        engine = get_analysis_engine()
    except Exception as e:
        st.error(f"Failed to initialize the analysis engine. Check dependencies. Error: {e}")
        return

    # Sidebar for Input
    with st.sidebar:
        st.header("Upload/Input")
        uploaded_files = st.file_uploader(
            "Upload Assignment Files (PDF, DOCX, TXT)", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )
        pasted_text = st.text_area("Or Paste Text Directly Here", height=300)
        student_id_input = st.text_input("Student/Submission ID", value="S001")
        
        process_button = st.button("🚀 Run Analysis", type="primary")

    if process_button:
        all_results = []
        
        # 1. Process Pasted Text
        if pasted_text.strip():
            with st.spinner("Analyzing pasted text..."):
                sentences = preprocess_text(pasted_text)
                results = engine.analyze_submission(sentences, "Pasted_Text", student_id_input)
                all_results.extend(results)

        # 2. Process Uploaded Files
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

        # 3. Display Results
        if all_results:
            st.success("Analysis Complete!")
            final_df = generate_sentence_df(all_results)
            
            # Overall Dashboard
            st.header("📊 Overall Submission Dashboard")
            col1, col2 = st.columns([1, 1])
            with col1:
                primary_mode = final_df['Fused Primary'].mode()
                st.metric(label="Primary Fused Emotion", value=primary_mode[0] if not primary_mode.empty else 'Neutral')
            with col2:
                model_mode = final_df['Model Primary'].mode()
                st.metric(label="Top Model Emotion (28-class)", value=model_mode[0] if not model_mode.empty else 'Neutral')
                
            st.plotly_chart(create_emotion_bar_chart(final_df, "Average Fused Emotion Scores (8 Ekman)"), use_container_width=True)
            
            # Sentence-by-Sentence Breakdown
            st.header("📜 Sentence-by-Sentence Breakdown (Interpretability)")
            for index, row in final_df.iterrows():
                with st.expander(f"Sentence {index+1}: **{row['Fused Primary']}** | {row['Sentence'][:80]}..."):
                    st.markdown(f"**Sentence:** *{row['Sentence']}*")
                    col_m, col_r, col_f = st.columns(3)
                    with col_m:
                        st.caption(f"**Transformer Primary:** **{row['Model Primary']}**")
                        st.json({k: f"{v:.3f}" for k, v in row['Model Scores (28)'].items()})
                    with col_r:
                        st.caption(f"**Rule-Based Primary:** **{row['Rule Primary']}**")
                        st.json({k: f"{v:.3f}" for k, v in row['Rule Scores (8)'].items()})
                    with col_f:
                        st.caption(f"**Fused Primary:** **{row['Fused Primary']}**")
                        st.json({k: f"{v:.3f}" for k, v in row['Fused Scores (8)'].items()})

            # Export Functionality
            csv_buffer = io.StringIO()
            engine.export_to_csv(all_results, csv_buffer)
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_buffer.getvalue().encode('utf-8'),
                file_name=f"emotion_analysis_{student_id_input}_{int(time.time())}.csv",
                mime="text/csv",
            )
        elif process_button:
            st.warning("No valid text was entered or extracted from the uploaded files.")
            
    if not process_button:
        st.subheader("Instructions")
        st.markdown("""
        1. Upload files or Paste text in the sidebar.
        2. Click **Run Analysis** to execute the two parallel systems (Transformer and Rule-Based).
        3. Review the **Overall Dashboard** and the **Sentence-by-Sentence Breakdown** for interpretability.
        """)

if __name__ == "__main__":
    main()

# critical_thinking_analyzer.py
# Analyzes text using Paul's Standards of Critical Thinking

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
import numpy as np

# ============================================================
#     PAUL'S STANDARDS OF CRITICAL THINKING - DEFINITIONS
# ============================================================

PAUL_STANDARDS = {
    "clarity": {
        "name": "Clarity",
        "color": "#3498DB",
        "icon": "🔍",
        "description": "Is the statement clear and understandable?",
        "question": "Could you elaborate? Could you illustrate? Could you give an example?",
        "indicators_positive": ["specifically", "for example", "in other words", "to illustrate", "meaning", "that is", "namely", "such as", "defined as", "to clarify"],
        "indicators_negative": ["somehow", "something", "stuff", "things", "whatever", "kind of", "sort of", "like", "you know", "etc"],
        "weight": 1.0
    },
    "accuracy": {
        "name": "Accuracy",
        "color": "#2ECC71",
        "icon": "✓",
        "description": "Is the statement true and free from errors?",
        "question": "How could we verify this? How could we find out if this is true?",
        "indicators_positive": ["according to", "research shows", "data indicates", "evidence suggests", "studies confirm", "verified", "documented", "proven", "factually", "statistics show"],
        "indicators_negative": ["everyone knows", "obviously", "clearly", "always", "never", "all", "none", "definitely", "absolutely certain", "no doubt"],
        "weight": 1.0
    },
    "precision": {
        "name": "Precision",
        "color": "#9B59B6",
        "icon": "🎯",
        "description": "Is the statement specific and detailed enough?",
        "question": "Could you be more specific? Could you give more details?",
        "indicators_positive": ["exactly", "precisely", "approximately", "measured", "calculated", "percent", "ratio", "specifically", "in particular", "detailed"],
        "indicators_negative": ["a lot", "many", "few", "some", "often", "sometimes", "rarely", "big", "small", "good", "bad", "nice", "very"],
        "weight": 1.0
    },
    "relevance": {
        "name": "Relevance",
        "color": "#E67E22",
        "icon": "🔗",
        "description": "Does the statement relate to the issue at hand?",
        "question": "How does this relate to the problem? How does this help with the issue?",
        "indicators_positive": ["therefore", "consequently", "as a result", "this relates to", "connected to", "relevant because", "pertinent", "applicable", "bearing on", "in relation to"],
        "indicators_negative": ["by the way", "incidentally", "speaking of", "anyway", "besides", "also", "moreover", "furthermore", "in addition"],
        "weight": 1.0
    },
    "depth": {
        "name": "Depth",
        "color": "#E74C3C",
        "icon": "📊",
        "description": "Does the statement address the complexity of the issue?",
        "question": "What factors make this difficult? What are the complexities?",
        "indicators_positive": ["underlying", "fundamental", "root cause", "complexity", "nuanced", "multifaceted", "layers", "deeper", "systematic", "comprehensive", "thorough"],
        "indicators_negative": ["simple", "easy", "just", "only", "merely", "basic", "straightforward", "obvious solution"],
        "weight": 1.0
    },
    "breadth": {
        "name": "Breadth",
        "color": "#1ABC9C",
        "icon": "🌐",
        "description": "Does the statement consider other viewpoints?",
        "question": "Is there another way to look at this? What would this look like from another perspective?",
        "indicators_positive": ["alternatively", "on the other hand", "from another perspective", "considering also", "however", "conversely", "different view", "opposing argument", "some argue", "others believe"],
        "indicators_negative": ["the only way", "must be", "has to be", "no other", "single solution", "one answer"],
        "weight": 1.0
    },
    "logic": {
        "name": "Logic",
        "color": "#F1C40F",
        "icon": "⚙️",
        "description": "Does the statement make sense and follow logically?",
        "question": "Does this follow from the evidence? Does this really make sense together?",
        "indicators_positive": ["because", "therefore", "thus", "hence", "consequently", "it follows that", "logically", "reasoning", "if then", "implies", "leads to"],
        "indicators_negative": ["but", "although", "despite", "regardless", "anyway", "still"],
        "weight": 1.0
    },
    "significance": {
        "name": "Significance",
        "color": "#8E44AD",
        "icon": "⭐",
        "description": "Is this the most important issue to focus on?",
        "question": "Is this the most important problem to consider? Which of these facts is most important?",
        "indicators_positive": ["importantly", "significantly", "crucially", "essentially", "fundamentally", "key point", "primary", "central", "critical", "vital", "paramount"],
        "indicators_negative": ["trivial", "minor", "insignificant", "unimportant", "negligible"],
        "weight": 1.0
    },
    "fairness": {
        "name": "Fairness",
        "color": "#16A085",
        "icon": "⚖️",
        "description": "Is the statement free from bias and self-interest?",
        "question": "Is my thinking justifiable? Am I considering others' viewpoints sympathetically?",
        "indicators_positive": ["objectively", "impartially", "fairly", "balanced", "unbiased", "neutral", "considering all", "without prejudice", "equitably", "justly"],
        "indicators_negative": ["obviously wrong", "stupid", "idiotic", "ridiculous", "absurd", "they always", "those people", "typical"],
        "weight": 1.0
    }
}

SCORE_LEVELS = {
    "excellent": {"min": 0.75, "color": "#2ECC71", "label": "Excellent", "icon": "🌟"},
    "good": {"min": 0.55, "color": "#3498DB", "label": "Good", "icon": "✅"},
    "adequate": {"min": 0.35, "color": "#F1C40F", "label": "Adequate", "icon": "⚠️"},
    "needs_work": {"min": 0.0, "color": "#E74C3C", "label": "Needs Improvement", "icon": "❌"}
}

# ============================================================
#           DATA EXTRACTION & PREPROCESSING
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
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

# ============================================================
#        PAUL'S CRITICAL THINKING ANALYZER ENGINE
# ============================================================

class CriticalThinkingAnalyzer:
    def __init__(self):
        self.standards = PAUL_STANDARDS
    
    def analyze_standard(self, sentence: str, standard_key: str) -> Dict[str, Any]:
        """Analyze a sentence against a specific standard"""
        standard = self.standards[standard_key]
        sentence_lower = sentence.lower()
        words = sentence_lower.split()
        
        # Count positive and negative indicators
        positive_count = 0
        negative_count = 0
        found_positive = []
        found_negative = []
        
        for indicator in standard["indicators_positive"]:
            if indicator.lower() in sentence_lower:
                positive_count += 1
                found_positive.append(indicator)
        
        for indicator in standard["indicators_negative"]:
            if indicator.lower() in sentence_lower:
                negative_count += 1
                found_negative.append(indicator)
        
        # Calculate base score
        base_score = 0.5  # Start neutral
        
        # Adjust for positive indicators
        base_score += min(positive_count * 0.15, 0.4)
        
        # Penalize for negative indicators
        base_score -= min(negative_count * 0.12, 0.35)
        
        # Additional heuristics per standard
        score_adjustment = self._apply_heuristics(sentence, standard_key, words)
        base_score += score_adjustment
        
        # Clamp score between 0 and 1
        final_score = max(0.0, min(1.0, base_score))
        
        # Determine level
        level = self._get_score_level(final_score)
        
        # Generate feedback
        feedback = self._generate_feedback(standard_key, final_score, found_positive, found_negative)
        
        return {
            "standard": standard_key,
            "standard_name": standard["name"],
            "score": final_score,
            "level": level,
            "color": standard["color"],
            "icon": standard["icon"],
            "positive_indicators": found_positive,
            "negative_indicators": found_negative,
            "feedback": feedback,
            "question": standard["question"]
        }
    
    def _apply_heuristics(self, sentence: str, standard_key: str, words: List[str]) -> float:
        """Apply additional heuristics based on sentence structure"""
        adjustment = 0.0
        sentence_lower = sentence.lower()
        
        if standard_key == "clarity":
            # Longer sentences with proper structure tend to be clearer
            if len(words) > 8 and len(words) < 30:
                adjustment += 0.05
            # Questions often seek clarity
            if "?" in sentence:
                adjustment += 0.05
            # Very short sentences may lack clarity
            if len(words) < 5:
                adjustment -= 0.1
                
        elif standard_key == "accuracy":
            # Numbers and statistics suggest accuracy
            if any(char.isdigit() for char in sentence):
                adjustment += 0.1
            # Quotes suggest citation
            if '"' in sentence or "'" in sentence:
                adjustment += 0.05
                
        elif standard_key == "precision":
            # Numbers indicate precision
            digit_count = sum(1 for c in sentence if c.isdigit())
            adjustment += min(digit_count * 0.03, 0.15)
            # Percentages are precise
            if "%" in sentence or "percent" in sentence_lower:
                adjustment += 0.1
                
        elif standard_key == "relevance":
            # Connecting words show relevance
            connectors = ["this", "that", "which", "these", "those"]
            if any(c in words for c in connectors):
                adjustment += 0.05
                
        elif standard_key == "depth":
            # Longer, complex sentences often show depth
            if len(words) > 15:
                adjustment += 0.08
            # Multiple clauses suggest depth
            if sentence.count(",") >= 2:
                adjustment += 0.05
                
        elif standard_key == "breadth":
            # Comparative words show breadth
            comparatives = ["while", "whereas", "compared", "contrast", "both", "either"]
            if any(c in sentence_lower for c in comparatives):
                adjustment += 0.1
                
        elif standard_key == "logic":
            # Causal language shows logic
            causal = ["cause", "effect", "result", "lead", "due to", "since"]
            if any(c in sentence_lower for c in causal):
                adjustment += 0.1
                
        elif standard_key == "significance":
            # Emphasis words show significance awareness
            emphasis = ["must", "need", "essential", "require", "necessary"]
            if any(e in sentence_lower for e in emphasis):
                adjustment += 0.08
                
        elif standard_key == "fairness":
            # First person plural suggests inclusivity
            if "we" in words or "our" in words:
                adjustment += 0.05
            # Absolute language reduces fairness
            absolutes = ["always", "never", "everyone", "no one", "all", "none"]
            if any(a in words for a in absolutes):
                adjustment -= 0.1
        
        return adjustment
    
    def _get_score_level(self, score: float) -> Dict[str, Any]:
        """Determine the performance level based on score"""
        for level_key, level_data in SCORE_LEVELS.items():
            if score >= level_data["min"]:
                return {"key": level_key, **level_data}
        return {"key": "needs_work", **SCORE_LEVELS["needs_work"]}
    
    def _generate_feedback(self, standard_key: str, score: float, 
                          found_positive: List[str], found_negative: List[str]) -> str:
        """Generate constructive feedback for the standard"""
        standard = self.standards[standard_key]
        
        if score >= 0.75:
            base = f"Excellent {standard['name'].lower()}! "
            if found_positive:
                base += f"Good use of: {', '.join(found_positive[:3])}."
        elif score >= 0.55:
            base = f"Good {standard['name'].lower()}. "
            if found_negative:
                base += f"Consider replacing: {', '.join(found_negative[:2])}."
            else:
                base += f"Could strengthen with more specific language."
        elif score >= 0.35:
            base = f"Adequate {standard['name'].lower()}, but needs improvement. "
            base += f"Ask yourself: {standard['question']}"
        else:
            base = f"{standard['name']} needs significant improvement. "
            if found_negative:
                base += f"Avoid vague terms like: {', '.join(found_negative[:2])}. "
            base += f"Consider: {standard['question']}"
        
        return base
    
    def analyze_sentence(self, sentence: str, index: int) -> Dict[str, Any]:
        """Analyze a sentence against all Paul's Standards"""
        results = {
            "index": index,
            "sentence": sentence,
            "word_count": len(sentence.split()),
            "standards": {},
            "overall_score": 0.0,
            "overall_level": None,
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        total_score = 0.0
        all_analyses = []
        
        for standard_key in self.standards:
            analysis = self.analyze_standard(sentence, standard_key)
            results["standards"][standard_key] = analysis
            total_score += analysis["score"]
            all_analyses.append((standard_key, analysis["score"]))
        
        # Calculate overall score
        results["overall_score"] = total_score / len(self.standards)
        results["overall_level"] = self._get_score_level(results["overall_score"])
        
        # Identify strengths and weaknesses
        sorted_analyses = sorted(all_analyses, key=lambda x: x[1], reverse=True)
        results["strengths"] = [self.standards[s[0]]["name"] for s in sorted_analyses[:3] if s[1] >= 0.55]
        results["weaknesses"] = [self.standards[s[0]]["name"] for s in sorted_analyses[-3:] if s[1] < 0.55]
        
        # Generate recommendations
        for standard_key, score in sorted_analyses[-2:]:
            if score < 0.55:
                results["recommendations"].append(self.standards[standard_key]["question"])
        
        return results
    
    def analyze_document(self, sentences: List[str], doc_name: str, doc_id: str) -> Dict[str, Any]:
        """Analyze an entire document"""
        sentence_results = []
        standard_totals = {k: 0.0 for k in self.standards}
        
        for i, sentence in enumerate(sentences):
            result = self.analyze_sentence(sentence, i + 1)
            result["document_name"] = doc_name
            result["document_id"] = doc_id
            sentence_results.append(result)
            
            for standard_key in self.standards:
                standard_totals[standard_key] += result["standards"][standard_key]["score"]
        
        # Calculate document-level statistics
        num_sentences = len(sentences)
        standard_averages = {k: v / num_sentences for k, v in standard_totals.items()}
        overall_avg = sum(standard_averages.values()) / len(standard_averages)
        
        return {
            "document_name": doc_name,
            "document_id": doc_id,
            "total_sentences": num_sentences,
            "sentence_results": sentence_results,
            "standard_averages": standard_averages,
            "overall_average": overall_avg,
            "overall_level": self._get_score_level(overall_avg)
        }

# ============================================================
#              VISUALIZATION FUNCTIONS
# ============================================================

def create_standard_radar(averages: Dict[str, float], title: str):
    """Create radar chart for standards overview"""
    categories = [PAUL_STANDARDS[k]["name"] for k in averages.keys()]
    values = list(averages.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=2),
        name='Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1.0]),
            angularaxis=dict(tickfont=dict(size=11))
        ),
        title=title,
        height=450,
        template="plotly_dark"
    )
    return fig

def create_standards_bar_chart(averages: Dict[str, float], title: str):
    """Create bar chart with Paul's Standards colors"""
    data = []
    for key, score in averages.items():
        data.append({
            "Standard": PAUL_STANDARDS[key]["name"],
            "Score": score,
            "Color": PAUL_STANDARDS[key]["color"],
            "Icon": PAUL_STANDARDS[key]["icon"]
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values("Score", ascending=True)
    
    fig = px.bar(df, x="Score", y="Standard", orientation='h',
                 color="Standard", color_discrete_map={
                     PAUL_STANDARDS[k]["name"]: PAUL_STANDARDS[k]["color"] 
                     for k in PAUL_STANDARDS
                 })
    
    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, 1], title="Score"),
        yaxis=dict(title=""),
        height=400,
        template="plotly_dark",
        showlegend=False
    )
    return fig

def create_sentence_heatmap(sentence_results: List[Dict], title: str):
    """Create heatmap of all sentences vs standards"""
    if not sentence_results:
        return go.Figure()
    
    # Build matrix
    standards_list = list(PAUL_STANDARDS.keys())
    matrix = []
    labels = []
    
    for result in sentence_results[:30]:  # Limit for readability
        row = [result["standards"][s]["score"] for s in standards_list]
        matrix.append(row)
        labels.append(f"S{result['index']}")
    
    fig = px.imshow(
        matrix,
        x=[PAUL_STANDARDS[s]["name"] for s in standards_list],
        y=labels,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title=title
    )
    
    fig.update_layout(
        height=max(400, len(labels) * 25),
        template="plotly_dark",
        xaxis=dict(tickangle=45)
    )
    return fig

def create_score_distribution(sentence_results: List[Dict], title: str):
    """Create distribution of overall scores"""
    scores = [r["overall_score"] for r in sentence_results]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        marker_color='#667eea',
        opacity=0.8
    ))
    
    # Add threshold lines
    for level_key, level_data in SCORE_LEVELS.items():
        if level_data["min"] > 0:
            fig.add_vline(x=level_data["min"], line_dash="dash", 
                         line_color=level_data["color"],
                         annotation_text=level_data["label"])
    
    fig.update_layout(
        title=title,
        xaxis_title="Overall Score",
        yaxis_title="Number of Sentences",
        height=350,
        template="plotly_dark"
    )
    return fig

def create_progress_chart(sentence_results: List[Dict], title: str):
    """Show how scores progress through the document"""
    if not sentence_results:
        return go.Figure()
    
    fig = go.Figure()
    
    # Overall score line
    fig.add_trace(go.Scatter(
        x=[r["index"] for r in sentence_results],
        y=[r["overall_score"] for r in sentence_results],
        mode='lines+markers',
        name='Overall Score',
        line=dict(color='#667eea', width=3)
    ))
    
    # Add trend line
    x = np.array([r["index"] for r in sentence_results])
    y = np.array([r["overall_score"] for r in sentence_results])
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    fig.add_trace(go.Scatter(
        x=x,
        y=p(x),
        mode='lines',
        name='Trend',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Sentence Number",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=350,
        template="plotly_dark"
    )
    return fig

# ============================================================
#           REPORT GENERATION FUNCTIONS
# ============================================================

def generate_sentence_report_html(result: Dict) -> str:
    """Generate HTML report for a single sentence"""
    html = f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 20px; border-radius: 15px; margin: 15px 0; 
                border-left: 5px solid {result['overall_level']['color']};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <h4 style="margin: 0; color: white;">📝 Sentence {result['index']}</h4>
            <span style="background: {result['overall_level']['color']}; padding: 5px 15px; 
                        border-radius: 20px; color: white; font-weight: bold;">
                {result['overall_level']['icon']} {result['overall_level']['label']} ({result['overall_score']:.0%})
            </span>
        </div>
        <p style="color: #ccc; font-style: italic; margin-bottom: 20px; 
                  padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
            "{result['sentence']}"
        </p>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
    """
    
    for standard_key, analysis in result["standards"].items():
        level_color = analysis["level"]["color"]
        html += f"""
            <div style="background: rgba(255,255,255,0.05); padding: 12px; border-radius: 10px;
                        border-top: 3px solid {analysis['color']};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: {analysis['color']}; font-weight: bold;">
                        {analysis['icon']} {analysis['standard_name']}
                    </span>
                    <span style="background: {level_color}; padding: 2px 8px; border-radius: 10px; 
                                font-size: 12px; color: white;">
                        {analysis['score']:.0%}
                    </span>
                </div>
                <p style="color: #999; font-size: 11px; margin: 8px 0 0 0;">
                    {analysis['feedback'][:100]}{'...' if len(analysis['feedback']) > 100 else ''}
                </p>
            </div>
        """
    
    html += "</div>"
    
    # Strengths and Weaknesses
    if result["strengths"] or result["weaknesses"]:
        html += '<div style="display: flex; gap: 20px; margin-top: 15px;">'
        if result["strengths"]:
            html += f"""
                <div style="flex: 1; background: rgba(46, 204, 113, 0.1); padding: 10px; border-radius: 8px;">
                    <strong style="color: #2ECC71;">💪 Strengths:</strong>
                    <span style="color: #ccc;"> {', '.join(result['strengths'])}</span>
                </div>
            """
        if result["weaknesses"]:
            html += f"""
                <div style="flex: 1; background: rgba(231, 76, 60, 0.1); padding: 10px; border-radius: 8px;">
                    <strong style="color: #E74C3C;">🎯 Areas to Improve:</strong>
                    <span style="color: #ccc;"> {', '.join(result['weaknesses'])}</span>
                </div>
            """
        html += "</div>"
    
    html += "</div>"
    return html

def generate_full_report_html(doc_result: Dict) -> str:
    """Generate complete HTML report for download"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Critical Thinking Analysis Report - {doc_result['document_name']}</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f23; color: #fff; padding: 40px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px; }}
            .metric {{ background: #1a1a2e; padding: 20px; border-radius: 10px; text-align: center; }}
            .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
            .metric-label {{ color: #888; font-size: 0.9em; }}
            .standard-card {{ background: #1a1a2e; padding: 15px; border-radius: 10px; margin: 10px 0; }}
            .sentence-report {{ background: #16213e; padding: 20px; border-radius: 15px; margin: 20px 0; }}
            .score-badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
            th {{ background: #667eea; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧠 Critical Thinking Analysis Report</h1>
            <p>Based on Paul's Universal Intellectual Standards</p>
            <p>Document: {doc_result['document_name']} | ID: {doc_result['document_id']}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="metric-grid">
            <div class="metric">
                <div class="metric-value">{doc_result['overall_level']['icon']}</div>
                <div class="metric-label">{doc_result['overall_level']['label']}</div>
            </div>
            <div class="metric">
                <div class="metric-value">{max(doc_result['standard_averages'], key=doc_result['standard_averages'].get).title()}</div>
                <div class="metric-label">Strongest Area</div>
            </div>
        </div>
        
        <h2>📊 Standards Overview</h2>
        <table>
            <tr>
                <th>Standard</th>
                <th>Score</th>
                <th>Level</th>
                <th>Description</th>
            </tr>
    """
    
    for key, score in doc_result['standard_averages'].items():
        std = PAUL_STANDARDS[key]
        level = "Excellent" if score >= 0.75 else "Good" if score >= 0.55 else "Adequate" if score >= 0.35 else "Needs Work"
        level_color = "#2ECC71" if score >= 0.75 else "#3498DB" if score >= 0.55 else "#F1C40F" if score >= 0.35 else "#E74C3C"
        html += f"""
            <tr>
                <td style="color: {std['color']}; font-weight: bold;">{std['icon']} {std['name']}</td>
                <td>{score:.0%}</td>
                <td><span style="background: {level_color}; padding: 3px 10px; border-radius: 10px;">{level}</span></td>
                <td style="color: #888;">{std['description']}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>📝 Detailed Sentence Analysis</h2>
    """
    
    for result in doc_result['sentence_results']:
        html += f"""
        <div class="sentence-report" style="border-left: 4px solid {result['overall_level']['color']};">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <strong>Sentence {result['index']}</strong>
                <span class="score-badge" style="background: {result['overall_level']['color']};">
                    {result['overall_level']['icon']} {result['overall_score']:.0%}
                </span>
            </div>
            <p style="color: #aaa; font-style: italic;">"{result['sentence']}"</p>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 15px;">
        """
        
        for std_key, analysis in result['standards'].items():
            html += f"""
                <div style="background: rgba(255,255,255,0.05); padding: 8px; border-radius: 8px; border-top: 2px solid {analysis['color']};">
                    <span style="color: {analysis['color']};">{analysis['icon']} {analysis['standard_name']}: </span>
                    <span style="color: {analysis['level']['color']};">{analysis['score']:.0%}</span>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    return html

# ============================================================
#                    MAIN STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(layout="wide", page_title="Paul's Critical Thinking Analyzer", page_icon="🧠")
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { color: white; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.8); margin: 10px 0 0 0; }
    .standard-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .metric-value-large {
        font-size: 3em;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 0.2em;
    }
    .metric-label-small {
        color: #888;
        font-size: 0.9em;
    }
    .sidebar-header {
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stCodeBlock {
        background-color: #1a1a2e !important;
        border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header"><h1>🧠 Paul\'s Critical Thinking Analyzer</h1><p>Assess text against Paul\'s Universal Intellectual Standards (Clarity, Accuracy, Precision, etc.)</p></div>', unsafe_allow_html=True)

    # --- Sidebar for Input ---
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><h2>Input Text</h2></div>', unsafe_allow_html=True)

        input_method = st.radio("Choose Input Method", ["Text Input", "File Upload"], index=0)

        uploaded_file = None
        user_text = ""
        doc_name = "Untitled Document"

        if input_method == "File Upload":
            uploaded_file = st.file_uploader("Upload Document (.txt, .pdf, .docx)", type=['txt', 'pdf', 'docx'])
            if uploaded_file:
                doc_name = uploaded_file.name
                file_type = uploaded_file.name.split('.')[-1].lower()
                text_result = extract_text_from_file(uploaded_file, file_type)
                
                if text_result.startswith("ERROR_"):
                    st.error(f"File extraction failed: {text_result}")
                else:
                    user_text = text_result
        else:
            user_text = st.text_area("Paste Text for Analysis", height=300, 
                                     value="The current economic policy is obviously flawed. It should be changed immediately because everyone knows a different system will clearly yield better results, but all the politicians are too corrupt to understand the simple solution.")
            doc_name = "User Input Text"
        
        # Add a section to display the raw text for verification
        st.subheader("Raw Text Preview")
        st.code(user_text[:500] + ('...' if len(user_text) > 500 else ''), language='text')

    # --- Main Content Area ---
    if user_text:
        # Preprocess and Analyze
        try:
            sentences = preprocess_text(user_text)
            if not sentences:
                st.warning("The extracted text is too short or could not be properly segmented into sentences (min 10 characters).")
                st.stop()

            analyzer = CriticalThinkingAnalyzer()
            doc_id = str(hash(user_text)) # Simple deterministic ID
            doc_result = analyzer.analyze_document(sentences, doc_name, doc_id)

            st.success(f"Analysis Complete: {doc_result['total_sentences']} sentences processed.")
            
            # 1. Overall Metrics
            st.header("📊 Document Overview")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value-large" style="color: {doc_result['overall_level']['color']};">{doc_result['overall_level']['icon']}</div>
                    <div class="metric-label-small">Overall Level: <strong>{doc_result['overall_level']['label']}</strong></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value-large">{doc_result['overall_average']:.1%}</div>
                    <div class="metric-label-small">Average Critical Thinking Score</div>
                </div>
                """, unsafe_allow_html=True)

            strongest = max(doc_result['standard_averages'], key=doc_result['standard_averages'].get)
            weakest = min(doc_result['standard_averages'], key=doc_result['standard_averages'].get)

            with col3:
                 st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value-large" style="color: {PAUL_STANDARDS[strongest]['color']};">{PAUL_STANDARDS[strongest]['icon']}</div>
                    <div class="metric-label-small">Strongest Standard: <strong>{PAUL_STANDARDS[strongest]['name']}</strong></div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value-large" style="color: {PAUL_STANDARDS[weakest]['color']};">{PAUL_STANDARDS[weakest]['icon']}</div>
                    <div class="metric-label-small">Weakest Standard: <strong>{PAUL_STANDARDS[weakest]['name']}</strong></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # 2. Visualization Charts
            st.header("📈 Visualization of Standards")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.plotly_chart(create_standard_radar(doc_result['standard_averages'], "Standards Radar Chart"), use_container_width=True)

            with chart_col2:
                st.plotly_chart(create_standards_bar_chart(doc_result['standard_averages'], "Average Score by Standard"), use_container_width=True)

            st.plotly_chart(create_progress_chart(doc_result['sentence_results'], "Critical Thinking Score Progression"), use_container_width=True)
            
            if len(doc_result['sentence_results']) > 1:
                st.plotly_chart(create_sentence_heatmap(doc_result['sentence_results'], "Sentence-by-Sentence Score Heatmap (First 30)"), use_container_width=True)

            st.markdown("---")

            # 3. Detailed Sentence Breakdown
            st.header("📝 Detailed Sentence Breakdown")
            
            # Dropdown/Selector for Sentence
            sentence_options = {r['index']: r['sentence'][:70] + '...' for r in doc_result['sentence_results']}
            selected_index = st.selectbox("Select a Sentence to View Details", options=list(sentence_options.keys()), format_func=lambda x: f"Sentence {x}: {sentence_options[x]}")

            selected_result = next((r for r in doc_result['sentence_results'] if r['index'] == selected_index), None)

            if selected_result:
                st.markdown(generate_sentence_report_html(selected_result), unsafe_allow_html=True)

            st.markdown("---")

            # 4. Download Report
            st.header("⬇️ Download Full Report")
            full_html_report = generate_full_report_html(doc_result)
            
            st.download_button(
                label="Download Full HTML Report",
                data=full_html_report,
                file_name=f"{doc_name.replace(' ', '_')}_critical_thinking_report.html",
                mime="text/html"
            )


        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.exception(e)

    else:
        st.info("Paste or upload a document in the sidebar to begin the Critical Thinking Analysis.")

if __name__ == '__main__':
    main()

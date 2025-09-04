import os
from dotenv import load_dotenv
import streamlit as st
import requests
import pandas as pd
import tempfile
import base64
from loguru import logger
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from pathlib import Path
import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re
import torch
from collections import Counter
from datetime import datetime, timedelta
import time

# Load environment variables
load_dotenv()
router_api_key = os.getenv("OPENROUTER_API_KEY")
guardian_api_key = os.getenv("GUARDIAN_API_KEY")
if not router_api_key:
    raise ValueError("Missing OPENROUTER_API_KEY in .env file.")
if not guardian_api_key:
    raise ValueError("Missing GUARDIAN_API_KEY in .env file.")

# OpenRouter client
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=router_api_key)

# Configure Streamlit
st.set_page_config(
    page_title="Investor Thematic Report Generator", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.title("📊 Investor Thematic Report Generator")
st.markdown("Generate polished, investor-grade PDF reports from financial news summaries.")

# GPU Status Display
if torch.cuda.is_available():
    st.success(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    device_id = 0
else:
    st.warning("⚠️ CUDA not available, running on CPU (slower performance)")
    device_id = -1

# Configuration Class
class ConfigManager:
    def __init__(self):
        self.config = {
            'min_articles': 10,
            'max_articles': 200,
            'min_content_length': 200,
            'max_content_length': 15000,
            'min_word_count': 50,
            'clustering_methods': ['kmeans', 'hierarchical'],
            'quality_thresholds': {
                'min_silhouette': 0.2,
                'max_davies_bouldin': 2.0,
                'min_articles_per_cluster': 2
            },
            'batch_size': 8,
            'max_summary_length': 150,
            'min_summary_length': 40
        }
    
    def update_from_sidebar(self):
        """Update config from sidebar inputs"""
        self.config['batch_size'] = st.sidebar.slider("Batch Size", 4, 16, 8)
        self.config['max_articles'] = st.sidebar.slider("Max Articles", 50, 1000, 200)
        self.config['min_silhouette'] = st.sidebar.slider("Min Silhouette Score", 0.1, 0.5, 0.2)

config_manager = ConfigManager()

# Text Processing Functions
def clean_text(text):
    """Advanced text cleaning for financial content"""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep financial symbols and punctuation
    text = re.sub(r'[^\w\s.,!?;:\-$%€£¥]', '', text)
    # Remove standalone numbers that might be page numbers or irrelevant
    text = re.sub(r'\b\d+\b(?=\s|$)', '', text)
    
    return text.strip()

def filter_quality_articles(df):
    """Filter articles by quality metrics"""
    config = config_manager.config
    
    # Clean article content
    df['article_content'] = df['article_content'].apply(clean_text)
    
    # Calculate quality metrics
    df['content_length'] = df['article_content'].str.len()
    df['word_count'] = df['article_content'].str.split().str.len()
    df['sentence_count'] = df['article_content'].str.count(r'[.!?]+')
    df['avg_word_length'] = df['article_content'].str.replace(r'[^\w\s]', '').str.split().str.len() / df['word_count']
    
    initial_count = len(df)
    
    # Apply filters
    df_filtered = df[
        (df['content_length'] >= config['min_content_length']) & 
        (df['content_length'] <= config['max_content_length']) &
        (df['word_count'] >= config['min_word_count']) &
        (df['sentence_count'] >= 3) &  # At least 3 sentences
        (df['avg_word_length'].fillna(0) > 3)  # Reasonable average word length
    ].copy()
    
    # Remove duplicates based on content similarity (simple approach)
    df_filtered = df_filtered.drop_duplicates(subset=['article_content'], keep='first')
    
    # Remove articles that are too similar (basic deduplication)
    if len(df_filtered) > 1:
        # Remove articles with >80% similar titles
        df_filtered = df_filtered.drop_duplicates(
            subset=[df_filtered['article_content'].str[:100]], 
            keep='first'
        )
    
    filtered_count = len(df_filtered)
    st.info(f"Article Quality Filter: {initial_count} → {filtered_count} articles ({((filtered_count/initial_count)*100):.1f}% retained)")
    
    return df_filtered

# Model Loading with Caching
@st.cache_resource
def load_models():
    """Load and cache models for reuse"""
    try:
        # Use a more efficient summarization model
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            model_kwargs={"low_cpu_mem_usage": True}
        )
        
        # Use a better embedding model for financial content
        embedder = SentenceTransformer(
            'all-MiniLM-L12-v2',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        st.success("✅ Models loaded successfully!")
        return summarizer, embedder
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        # Fallback to CPU
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1
        )
        embedder = SentenceTransformer('all-MiniLM-L12-v2', device='cpu')
        st.warning("⚠️ Loaded models on CPU as fallback")
        return summarizer, embedder

# Batch Processing Functions
def batch_summarize(texts, summarizer, batch_size=8):
    """Process summaries in batches for better efficiency"""
    summaries = []
    config = config_manager.config
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_idx = i // batch_size + 1
        status_text.text(f"Processing batch {batch_idx}/{total_batches}")
        
        batch = texts[i:i+batch_size]
        # Truncate to reasonable length for model
        truncated_batch = [text[:4000] for text in batch]
        
        try:
            batch_summaries = summarizer(
                truncated_batch,
                max_length=config['max_summary_length'],
                min_length=config['min_summary_length'],
                do_sample=False,
                batch_size=len(batch)
            )
            summaries.extend([s["summary_text"] for s in batch_summaries])
        except Exception as e:
            st.warning(f"Error in batch {batch_idx}: {str(e)}")
            # Fallback: process individually
            for text in truncated_batch:
                try:
                    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
                    summaries.append(summary[0]["summary_text"])
                except:
                    summaries.append("Summary unavailable due to processing error.")
        
        progress_bar.progress(batch_idx / total_batches)
    
    status_text.empty()
    progress_bar.empty()
    return summaries

# Advanced Clustering Functions
def find_optimal_clusters(embeddings, methods=['kmeans', 'hierarchical']):
    """Find optimal number of clusters using multiple methods and metrics"""
    
    # Dimensionality reduction for better clustering
    n_components = min(50, embeddings.shape[1], embeddings.shape[0]-1)
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_pca)
    
    results = []
    k_min = 2
    k_max = min(10, len(embeddings) // 3)  # Ensure minimum articles per cluster
    
    st.info(f"Testing clustering with k from {k_min} to {k_max} using {methods}")
    
    for method_name in methods:
        for k in range(k_min, k_max + 1):
            try:
                # Create clusterer based on method
                if method_name == 'kmeans':
                    clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
                elif method_name == 'hierarchical':
                    clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
                else:
                    continue
                
                labels = clusterer.fit_predict(embeddings_scaled)
                
                # Calculate multiple metrics
                sil_score = silhouette_score(embeddings_scaled, labels)
                db_score = davies_bouldin_score(embeddings_scaled, labels)
                ch_score = calinski_harabasz_score(embeddings_scaled, labels)
                
                # Check cluster size distribution
                unique_labels, counts = np.unique(labels, return_counts=True)
                min_cluster_size = np.min(counts)
                cluster_balance = np.std(counts) / np.mean(counts)  # Lower is better
                
                # Composite score (higher is better)
                composite_score = (sil_score * 0.4) + (ch_score / 1000 * 0.3) - (db_score * 0.2) - (cluster_balance * 0.1)
                
                results.append({
                    'method': method_name,
                    'k': k,
                    'silhouette': sil_score,
                    'davies_bouldin': db_score,
                    'calinski_harabasz': ch_score,
                    'min_cluster_size': min_cluster_size,
                    'cluster_balance': cluster_balance,
                    'composite_score': composite_score,
                    'labels': labels
                })
                
            except Exception as e:
                st.warning(f"Error with {method_name} k={k}: {str(e)}")
                continue
    
    if not results:
        st.error("No valid clustering results found")
        return None, embeddings_scaled, []
    
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'labels'} for r in results])
    
    # Find best configuration
    best_idx = results_df['composite_score'].idxmax()
    best_config = results[best_idx]
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    for method in methods:
        method_data = results_df[results_df['method'] == method]
        if len(method_data) > 0:
            ax1.plot(method_data['k'], method_data['silhouette'], marker='o', label=f'{method} - Silhouette')
            ax2.plot(method_data['k'], method_data['davies_bouldin'], marker='s', label=f'{method} - Davies-Bouldin')
            ax3.plot(method_data['k'], method_data['calinski_harabasz'], marker='^', label=f'{method} - Calinski-Harabasz')
            ax4.plot(method_data['k'], method_data['composite_score'], marker='d', label=f'{method} - Composite')
    
    ax1.set_title('Silhouette Score (Higher = Better)')
    ax1.set_xlabel('Number of Clusters')
    ax1.legend()
    
    ax2.set_title('Davies-Bouldin Index (Lower = Better)')
    ax2.set_xlabel('Number of Clusters')
    ax2.legend()
    
    ax3.set_title('Calinski-Harabasz Index (Higher = Better)')
    ax3.set_xlabel('Number of Clusters')
    ax3.legend()
    
    ax4.set_title('Composite Score (Higher = Better)')
    ax4.set_xlabel('Number of Clusters')
    ax4.legend()
    
    plt.tight_layout()
    
    # Mark best configuration
    best_k = best_config['k']
    best_method = best_config['method']
    ax4.axvline(x=best_k, color='red', linestyle='--', alpha=0.7)
    ax4.text(best_k, ax4.get_ylim()[1]*0.9, f'Best: {best_method}\nk={best_k}', 
             ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(plot_path.name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return best_config, embeddings_scaled, plot_path.name

def generate_theme_name_robust(summaries, tid, retries=3):
    """Generate theme names with consistency and error handling"""
    if not summaries:
        return f"Theme {tid + 1}"
    
    joined_summaries = "\n\n".join(summaries[:5])  # Limit to first 5 summaries
    
    candidates = []
    prompt = f"""
You are a financial analyst. Based on these article summaries, create a concise theme label (3-5 words max).

Requirements:
- Focus on the main financial/business topic
- Use professional terminology
- Be specific and descriptive
- Return ONLY the theme name, no quotes or formatting

Summaries:
{joined_summaries}

Theme name:"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1:free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=50,
                extra_headers={"HTTP-Referer": "https://openrouter.ai", "X-Title": "Theme Label"}
            )
            
            raw_output = response.choices[0].message.content.strip()
            # Clean output
            clean_output = re.sub(r'["\'\[\]{}()]', '', raw_output)
            clean_output = re.sub(r'Theme name:?', '', clean_output, flags=re.IGNORECASE).strip()
            
            if clean_output and len(clean_output.split()) <= 6:
                candidates.append(clean_output)
                
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed for theme {tid}: {str(e)}")
            continue
    
    if candidates:
        # Return most common candidate or first if all different
        most_common = Counter(candidates).most_common(1)
        return most_common[0][0] if most_common else candidates[0]
    else:
        return f"Financial Theme {tid + 1}"

# Comprehensive Evaluation Function
def comprehensive_evaluation(df, embeddings, labels, theme_names):
    """Multi-dimensional evaluation of clustering and content quality"""
    
    # Clustering quality metrics
    clustering_metrics = {
        'silhouette_score': silhouette_score(embeddings, labels),
        'davies_bouldin_index': davies_bouldin_score(embeddings, labels),
        'calinski_harabasz_index': calinski_harabasz_score(embeddings, labels)
    }
    
    # Content diversity within clusters
    diversity_scores = []
    cluster_sizes = []
    
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_texts = df[cluster_mask]['summary'].tolist()
        cluster_sizes.append(len(cluster_texts))
        
        if len(cluster_texts) > 1:
            cluster_embeddings = embeddings[cluster_mask]
            # Calculate average pairwise cosine similarity
            sim_matrix = cosine_similarity(cluster_embeddings)
            # Get upper triangle (excluding diagonal)
            upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
            if len(upper_tri_indices[0]) > 0:
                avg_similarity = np.mean(sim_matrix[upper_tri_indices])
                diversity_scores.append(1 - avg_similarity)  # Higher diversity = lower similarity
    
    # Cluster balance
    cluster_balance = {
        'mean_size': np.mean(cluster_sizes),
        'std_size': np.std(cluster_sizes),
        'min_size': np.min(cluster_sizes),
        'max_size': np.max(cluster_sizes),
        'balance_ratio': np.std(cluster_sizes) / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else float('inf')
    }
    
    return {
        'clustering_quality': clustering_metrics,
        'content_diversity': np.mean(diversity_scores) if diversity_scores else 0,
        'cluster_balance': cluster_balance,
        'n_clusters': len(np.unique(labels)),
        'total_articles': len(df)
    }

# Enhanced PDF Generation
def generate_enhanced_pdf(df, theme_names, evaluation_results):
    """Generate comprehensive PDF report with evaluation metrics"""
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp.name, pagesize=letter)
    width, height = letter
    
    # Title page
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 80, "Financial Market Thematic Analysis")
    
    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, height - 110, f"Generated on {datetime.now().strftime('%B %d, %Y')}")
    c.drawCentredString(width / 2, height - 130, f"Analysis of {evaluation_results['total_articles']} articles across {evaluation_results['n_clusters']} themes")
    
    # Add evaluation summary
    y = height - 180
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Analysis Quality Metrics")
    y -= 25
    
    c.setFont("Helvetica", 11)
    metrics = evaluation_results['clustering_quality']
    c.drawString(60, y, f"• Silhouette Score: {metrics['silhouette_score']:.3f} (Higher is better, >0.3 is good)")
    y -= 15
    c.drawString(60, y, f"• Davies-Bouldin Index: {metrics['davies_bouldin_index']:.3f} (Lower is better, <1.0 is good)")
    y -= 15
    c.drawString(60, y, f"• Content Diversity: {evaluation_results['content_diversity']:.3f} (Higher indicates diverse content within themes)")
    y -= 15
    c.drawString(60, y, f"• Average Articles per Theme: {evaluation_results['cluster_balance']['mean_size']:.1f}")
    
    c.showPage()
    
    # Generate content for each theme
    all_content = []
    y = height - 50
    
    for tid in sorted(df["theme_id"].unique()):
        cluster_articles = df[df["theme_id"] == tid]
        summaries = cluster_articles["summary"].tolist()
        
        # Generate insight using OpenRouter
        content = generate_insight_enhanced(summaries, tid, theme_names.get(tid, f"Theme {tid + 1}"))
        all_content.append(content)
        
        theme_title = theme_names.get(tid, f"Theme {tid + 1}")
        
        # Theme header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, y, f"{theme_title}")
        y -= 15
        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"({len(summaries)} articles)")
        y -= 25
        
        # Content
        c.setFont("Helvetica", 10)
        for line in content.split("\n"):
            for subline in textwrap.wrap(line, 100):
                if y < 50:
                    c.showPage()
                    y = height - 50
                c.drawString(40, y, subline)
                y -= 12
            y -= 6
        y -= 20
        
        if y < 200:  # Start new page if space is low
            c.showPage()
            y = height - 50
    
    c.save()
    return tmp.name, all_content

def generate_insight_enhanced(texts, tid, theme_name):
    """Generate enhanced insights with better prompting"""
    if not texts:
        return f"No content available for {theme_name}."
    
    joined_texts = "\n\n".join(texts[:8])  # Limit to prevent token overflow
    
    prompt = f"""As a senior financial analyst, write a comprehensive report section for the theme "{theme_name}".

Based on these article summaries:
{joined_texts}

Structure your analysis with these sections:

**Executive Summary** (2-3 sentences summarizing the key theme)

**Key Market Trends** (3-4 specific trends with evidence from the articles)

**Investment Implications** (3-4 actionable insights for investors, including potential risks and opportunities)

**Market Outlook** (Brief forward-looking perspective)

Keep each section focused and professional. Use specific details from the summaries when possible."""

    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
            extra_headers={"HTTP-Referer": "https://openrouter.ai", "X-Title": "Investment Insight"}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Error generating insight for {theme_name}: {str(e)}")
        return f"**{theme_name}**\n\nAnalysis temporarily unavailable due to processing constraints. Key articles in this theme focus on relevant financial developments that warrant investor attention."

# Enhanced LLM Judge Evaluation
def enhanced_llm_judge_evaluation(report_content, df, theme_names):
    """Comprehensive report evaluation using LLM"""
    
    # Calculate some basic statistics for context
    n_articles = len(df)
    n_themes = len(theme_names)
    avg_articles_per_theme = n_articles / n_themes if n_themes > 0 else 0
    
    context_info = f"""
Report Context:
- Total articles analyzed: {n_articles}
- Number of themes identified: {n_themes}
- Average articles per theme: {avg_articles_per_theme:.1f}
- Themes: {', '.join(theme_names.values())}
"""
    
    judge_prompt = f"""You are an expert financial report evaluator. Please evaluate this investor thematic report comprehensively.

{context_info}

Report Content:
{report_content[:6000]}  # Truncate to manage token limits

Please evaluate the report on these criteria (score 1-10 for each):

1. **Content Quality**: Are the insights meaningful and well-supported?
2. **Investment Relevance**: How valuable is this for actual investment decisions?
3. **Clarity & Structure**: Is the report well-organized and easy to understand?
4. **Market Understanding**: Does it demonstrate good grasp of financial markets?
5. **Actionability**: Are the recommendations specific and actionable?

For each criterion, provide:
- Score (1-10)
- Brief justification (1-2 sentences)

Format your response as:
Content Quality: [Score]/10 - [Justification]
Investment Relevance: [Score]/10 - [Justification]
[etc.]

Overall Assessment: [2-3 sentence summary]"""

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-image-preview:free",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.1,
            max_tokens=1000,
            extra_headers={"HTTP-Referer": "https://openrouter.ai", "X-Title": "Report Evaluation"}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Error in LLM judge evaluation: {str(e)}")
        return "Evaluation temporarily unavailable due to processing constraints."

# Progress Tracking Components
def create_progress_tracker(steps):
    """Create progress tracking components"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    return progress_bar, status_text, steps

def update_progress(progress_bar, status_text, step, total_steps, message):
    """Update progress bar and status"""
    progress = step / total_steps
    progress_bar.progress(progress)
    status_text.text(f"Step {step}/{total_steps}: {message}")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
config_manager.update_from_sidebar()

# Date inputs
from_date = st.sidebar.date_input("From Date", pd.to_datetime("2023-01-01"))
to_date = st.sidebar.date_input("To Date", pd.to_datetime("today"))

# Advanced settings
with st.sidebar.expander("🔧 Advanced Settings"):
    clustering_methods = st.multiselect(
        "Clustering Methods",
        ['kmeans', 'hierarchical'],
        default=['kmeans', 'hierarchical']
    )
    use_pca = st.checkbox("Use PCA for dimensionality reduction", value=True)
    enable_quality_filter = st.checkbox("Enable article quality filtering", value=True)

# Validation
if to_date <= from_date:
    st.error("❌ End date must be after start date")
    st.stop()

if (to_date - from_date).days > 365:
    st.warning("⚠️ Large date range may cause slower performance")

# Image Banner
image_path = Path(r"C:\Users\triki\Desktop\MLOps and GenAi\Ai_Financial_Insights_Agent\images\fin_insights.png")
if image_path.exists():
    st.image(str(image_path), caption="GenAI Financial Insights", width=500)

# =========================
# MAIN PIPELINE EXECUTION
# =========================

if st.button("🚀 Run Enhanced Pipeline", type='primary', use_container_width=True):
    start_time = time.time()
    
    # Initialize progress tracking
    total_steps = 7
    progress_bar, status_text, _ = create_progress_tracker(total_steps)
    
    try:
        # Step 1: Load Models
        update_progress(progress_bar, status_text, 1, total_steps, "Loading AI models...")
        summarizer, embedder = load_models()
        
        # Step 2: Scrape Articles
        update_progress(progress_bar, status_text, 2, total_steps, "Fetching financial news articles...")
        
        endpoint = "https://content.guardianapis.com/search"
        params = {
            "q": "financial news OR markets OR economy OR business",
            "section": "business",
            "from-date": from_date.isoformat(),
            "to-date": to_date.isoformat(),
            "show-fields": "bodyText,webPublicationDate",
            "page-size": min(config_manager.config['max_articles'], 200),
            "api-key": guardian_api_key
        }
        
        resp = requests.get(endpoint, params=params)
        resp.raise_for_status()
        articles = resp.json()["response"]["results"]
        
        if not articles:
            st.error("❌ No articles found for the specified date range")
            st.stop()
        
        # Create dataframe
        df = pd.DataFrame([
            {
                "publicationDate": pd.to_datetime(a["webPublicationDate"]).tz_localize(None),
                "article_content": a["fields"]["bodyText"]
            }
            for a in articles
            if a.get("fields", {}).get("bodyText")
        ]).sort_values("publicationDate", ascending=False)
        
        st.success(f"✅ Scraped {len(df)} articles")
        
        # Step 3: Quality Filtering
        if enable_quality_filter:
            update_progress(progress_bar, status_text, 3, total_steps, "Filtering article quality...")
            df = filter_quality_articles(df)
            
            if len(df) < 10:
                st.warning(f"⚠️ Only {len(df)} quality articles found. Consider expanding date range or lowering quality thresholds.")
                if len(df) < 5:
                    st.error("❌ Insufficient articles for meaningful analysis")
                    st.stop()
        
        # Step 4: Summarization
        update_progress(progress_bar, status_text, 4, total_steps, "Generating article summaries...")
        
        df["summary"] = batch_summarize(
            df["article_content"].tolist(), 
            summarizer, 
            batch_size=config_manager.config['batch_size']
        )
        
        st.success(f"✅ Generated summaries for {len(df)} articles")
        
        # Step 5: Advanced Clustering
        update_progress(progress_bar, status_text, 5, total_steps, "Performing advanced clustering analysis...")
        
        # Generate embeddings
        embeddings = embedder.encode(df["summary"].tolist(), convert_to_numpy=True, show_progress_bar=False)
        
        # Find optimal clustering
        best_config, processed_embeddings, plot_path = find_optimal_clusters(
            embeddings, 
            methods=clustering_methods if clustering_methods else ['kmeans']
        )
        
        if best_config is None:
            st.error("❌ Clustering analysis failed")
            st.stop()
        
        # Apply best clustering
        df["theme_id"] = best_config['labels']
        best_k = best_config['k']
        best_method = best_config['method']
        
        st.success(f"✅ Optimal clustering: {best_method} with {best_k} clusters")
        
        # Display clustering visualization
        st.subheader("📊 Clustering Analysis")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(plot_path, caption=f"Clustering Evaluation - Best: {best_method} (k={best_k})", use_container_width=True)
        
        with col2:
            st.metric("Silhouette Score", f"{best_config['silhouette']:.3f}")
            st.metric("Davies-Bouldin Index", f"{best_config['davies_bouldin']:.3f}")
            st.metric("Min Cluster Size", best_config['min_cluster_size'])
            st.metric("Cluster Balance", f"{best_config['cluster_balance']:.3f}")
        
        # Step 6: Generate Theme Names
        update_progress(progress_bar, status_text, 6, total_steps, "Generating meaningful theme labels...")
        
        theme_names = {}
        theme_progress = st.progress(0)
        
        for i, tid in enumerate(sorted(df["theme_id"].unique())):
            theme_summaries = df[df["theme_id"] == tid]["summary"].tolist()
            theme_name = generate_theme_name_robust(theme_summaries, tid)
            theme_names[tid] = theme_name
            theme_progress.progress((i + 1) / len(df["theme_id"].unique()))
        
        theme_progress.empty()
        
        # Display theme summary
        st.subheader("🧠 Identified Themes")
        theme_summary_data = []
        for tid in sorted(df["theme_id"].unique()):
            cluster_data = df[df["theme_id"] == tid]
            theme_summary_data.append({
                "Theme": theme_names[tid],
                "Articles": len(cluster_data),
                "Percentage": f"{(len(cluster_data)/len(df)*100):.1f}%",
                "Date Range": f"{cluster_data['publicationDate'].min().strftime('%Y-%m-%d')} to {cluster_data['publicationDate'].max().strftime('%Y-%m-%d')}"
            })
        
        theme_df = pd.DataFrame(theme_summary_data)
        st.dataframe(theme_df, use_container_width=True)
        
        # Step 7: Comprehensive Evaluation
        update_progress(progress_bar, status_text, 7, total_steps, "Performing comprehensive evaluation...")
        
        df["theme"] = df["theme_id"].map(theme_names)
        evaluation_results = comprehensive_evaluation(df, processed_embeddings, df["theme_id"].values, theme_names)
        
        # Display evaluation metrics
        st.subheader("📈 Analysis Quality Assessment")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Silhouette Score", f"{evaluation_results['clustering_quality']['silhouette_score']:.3f}")
        with col2:
            st.metric("Content Diversity", f"{evaluation_results['content_diversity']:.3f}")
        with col3:
            st.metric("Average Theme Size", f"{evaluation_results['cluster_balance']['mean_size']:.1f}")
        with col4:
            st.metric("Theme Balance", f"{evaluation_results['cluster_balance']['balance_ratio']:.2f}")
        
        # Quality interpretation
        sil_score = evaluation_results['clustering_quality']['silhouette_score']
        if sil_score > 0.5:
            quality_status = "🟢 Excellent clustering quality"
        elif sil_score > 0.3:
            quality_status = "🟡 Good clustering quality"
        elif sil_score > 0.2:
            quality_status = "🟠 Fair clustering quality"
        else:
            quality_status = "🔴 Poor clustering quality - consider different parameters"
        
        st.info(quality_status)
        
        # Generate Enhanced PDF Report
        st.subheader("📄 Generating Enhanced Report...")
        pdf_path, all_content = generate_enhanced_pdf(df, theme_names, evaluation_results)
        
        # PDF Download
        with open(pdf_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f'<a href="data:application/pdf;base64,{b64}" download="enhanced_financial_report.pdf">'
                '📄 ⬇️ Download Enhanced Report</a>', 
                unsafe_allow_html=True
            )
        
        # Enhanced LLM Judge Evaluation
        st.subheader("🔍 Expert Report Evaluation")
        report_content = "\n\n".join(all_content)
        evaluation_text = enhanced_llm_judge_evaluation(report_content, df, theme_names)
        
        st.text_area("📑 Expert Evaluation", evaluation_text, height=300)
        
        # Parse and structure evaluation
        eval_data = []
        lines = evaluation_text.split('\n')
        for line in lines:
            # Look for pattern: "Criterion: Score/10 - Justification"
            match = re.match(r'^(.*?):\s*([0-9]+)/10\s*-\s*(.+)', line.strip())
            if match:
                criterion = match.group(1).strip()
                score = match.group(2).strip()
                justification = match.group(3).strip()
                eval_data.append({
                    "Criterion": criterion,
                    "Score": f"{score}/10",
                    "Justification": justification
                })
        
        if eval_data:
            eval_df = pd.DataFrame(eval_data)
            st.subheader("📊 Evaluation Summary")
            st.dataframe(eval_df, use_container_width=True)
        
        # Export all data to Excel files
        st.subheader("💾 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        # Export 1: Clustering metrics
        with col1:
            metrics_data = {
                'metric': list(evaluation_results['clustering_quality'].keys()),
                'value': list(evaluation_results['clustering_quality'].values())
            }
            metrics_df = pd.DataFrame(metrics_data)
            metrics_excel = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
            metrics_df.to_excel(metrics_excel.name, index=False)
            
            with open(metrics_excel.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                st.markdown(
                    f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" '
                    'download="clustering_metrics.xlsx">📊 Clustering Metrics</a>',
                    unsafe_allow_html=True
                )
        
        # Export 2: Themed articles
        with col2:
            export_df = df[['publicationDate', 'article_content', 'summary', 'theme_id', 'theme']].copy()
            export_df['publicationDate'] = export_df['publicationDate'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            themed_excel = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
            export_df.to_excel(themed_excel.name, index=False)
            
            with open(themed_excel.name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                st.markdown(
                    f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" '
                    'download="themed_articles.xlsx">📰 Themed Articles</a>',
                    unsafe_allow_html=True
                )
        
        # Export 3: Evaluation results
        with col3:
            if eval_data:
                eval_excel = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                eval_df.to_excel(eval_excel.name, index=False)
                
                with open(eval_excel.name, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    st.markdown(
                        f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" '
                        'download="evaluation_results.xlsx">🔍 Evaluation Results</a>',
                        unsafe_allow_html=True
                    )
        
        # Performance Summary
        end_time = time.time()
        processing_time = end_time - start_time
        
        st.success(f"🎉 Pipeline completed successfully in {processing_time:.1f} seconds!")
        
        # Final Statistics
        st.subheader("📊 Final Statistics")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Total Articles", len(df))
        with stats_col2:
            st.metric("Themes Identified", len(theme_names))
        with stats_col3:
            st.metric("Processing Time", f"{processing_time:.1f}s")
        with stats_col4:
            st.metric("Articles/Second", f"{len(df)/processing_time:.1f}")
        
        # Detailed theme breakdown
        with st.expander("🔍 Detailed Theme Analysis"):
            for tid in sorted(df["theme_id"].unique()):
                theme_data = df[df["theme_id"] == tid]
                st.write(f"**{theme_names[tid]}** ({len(theme_data)} articles)")
                
                # Show sample summaries
                sample_summaries = theme_data["summary"].head(3).tolist()
                for i, summary in enumerate(sample_summaries, 1):
                    st.write(f"  {i}. {summary[:200]}...")
                
                st.write("---")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error: {str(e)}")
        st.info("Please check your internet connection and API keys.")
        
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        logger.error(f"Pipeline error: {e}", exc_info=True)
        st.info("Please try again or contact support if the issue persists.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>© 2025 Zeineb Moalla | Financial Data Science Master Student</p>
    <p>Enhanced with advanced AI models, multi-metric clustering, and comprehensive evaluation</p>
</div> """, unsafe_allow_html=True )
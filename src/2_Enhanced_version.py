
# import os
# from dotenv import load_dotenv
# import streamlit as st
# import requests
# import pandas as pd
# import tempfile
# import base64
# from loguru import logger
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# from sklearn.manifold import TSNE
# from kneed import KneeLocator
# import numpy as np
# import matplotlib.pyplot as plt
# from openai import OpenAI
# from pathlib import Path
# import textwrap
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# import re
# import torch
# from datasets import Dataset
# from datetime import datetime


# # Load environment variables
# load_dotenv()
# router_api_key = os.getenv("OPENROUTER_API_KEY")
# guardian_api_key = os.getenv("GUARDIAN_API_KEY")
# default_page_size = int(os.getenv("GUARDIAN_PAGE_SIZE", 5))


# if not router_api_key:
#     raise ValueError("Missing OPENROUTER_API_KEY in .env file.")
# if not guardian_api_key:
#     raise ValueError("Missing GUARDIAN_API_KEY in .env file.")


# # OpenRouter client
# client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=router_api_key)


# # ═══════════════════════════════════════════════════════════════════════
# # PAGE CONFIGURATION
# # ═══════════════════════════════════════════════════════════════════════
# st.set_page_config(
#     page_title="Financial Insights AI",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )


# # ═══════════════════════════════════════════════════════════════════════
# # SESSION STATE INITIALIZATION
# # ═══════════════════════════════════════════════════════════════════════
# if "pipeline_running" not in st.session_state:
#     st.session_state.pipeline_running = False
# if "results" not in st.session_state:
#     st.session_state.results = {}


# # ═══════════════════════════════════════════════════════════════════════
# # ENHANCED CSS STYLING
# # ═══════════════════════════════════════════════════════════════════════
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

# * {
#     font-family: 'Inter', sans-serif;
# }

# /* Main Background */
# .main {
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     padding: 2rem;
# }

# .stApp {
#     background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
# }

# /* Header Container */
# .app-header {
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     padding: 2.5rem;
#     border-radius: 20px;
#     margin-bottom: 2rem;
#     box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
#     text-align: center;
#     animation: slideDown 0.6s ease-out;
# }

# .app-title {
#     font-size: 3rem;
#     font-weight: 800;
#     color: white;
#     margin-bottom: 0.5rem;
#     text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
# }

# .app-subtitle {
#     color: rgba(255, 255, 255, 0.9);
#     font-size: 1.2rem;
#     font-weight: 400;
# }

# /* Cards */
# .info-card {
#     background: white;
#     border-radius: 15px;
#     padding: 1.5rem;
#     margin: 1rem 0;
#     box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
#     transition: all 0.3s ease;
#     border-left: 4px solid #667eea;
# }

# .info-card:hover {
#     transform: translateY(-5px);
#     box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
# }

# /* Step Cards */
# .step-card {
#     background: white;
#     border-radius: 15px;
#     padding: 2rem;
#     margin: 1.5rem 0;
#     box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
#     border-top: 5px solid #667eea;
#     animation: fadeInUp 0.5s ease-out;
# }

# .step-header {
#     font-size: 1.5rem;
#     font-weight: 700;
#     color: #667eea;
#     margin-bottom: 1rem;
#     display: flex;
#     align-items: center;
#     gap: 0.5rem;
# }

# .step-number {
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     color: white;
#     width: 40px;
#     height: 40px;
#     border-radius: 50%;
#     display: flex;
#     align-items: center;
#     justify-content: center;
#     font-weight: 700;
#     font-size: 1.2rem;
# }

# /* Buttons */
# .stButton button {
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
#     color: white !important;
#     border: none !important;
#     border-radius: 12px !important;
#     padding: 0.8rem 2.5rem !important;
#     font-weight: 600 !important;
#     font-size: 1.1rem !important;
#     transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
#     box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
#     cursor: pointer !important;
# }

# .stButton button:hover {
#     transform: translateY(-3px) !important;
#     box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6) !important;
# }

# .stButton button:active {
#     transform: translateY(-1px) !important;
# }

# /* Sidebar Styling */
# section[data-testid="stSidebar"] {
#     background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
#     padding: 2rem 1rem;
# }

# section[data-testid="stSidebar"] * {
#     color: white !important;
# }

# section[data-testid="stSidebar"] .stDateInput input {
#     color: #2d3748 !important;
#     background: white !important;
#     border-radius: 8px !important;
# }

# section[data-testid="stSidebar"] .stNumberInput input {
#     color: #2d3748 !important;
#     background: white !important;
#     border-radius: 8px !important;
# }

# /* Success/Error Messages */
# .stSuccess {
#     background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
#     color: white !important;
#     border-radius: 12px !important;
#     padding: 1rem !important;
#     border: none !important;
# }

# .stError {
#     background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
#     color: white !important;
#     border-radius: 12px !important;
#     padding: 1rem !important;
#     border: none !important;
# }

# .stInfo {
#     background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
#     color: white !important;
#     border-radius: 12px !important;
#     padding: 1rem !important;
#     border: none !important;
# }

# .stWarning {
#     background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
#     color: white !important;
#     border-radius: 12px !important;
#     padding: 1rem !important;
#     border: none !important;
# }

# /* Dataframe Styling */
# .stDataFrame {
#     border-radius: 12px !important;
#     overflow: hidden !important;
#     box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important;
# }

# /* Metrics */
# [data-testid="stMetricValue"] {
#     font-size: 2rem !important;
#     font-weight: 700 !important;
#     color: #667eea !important;
# }

# [data-testid="stMetricLabel"] {
#     font-weight: 600 !important;
#     color: #4b5563 !important;
# }

# /* Progress Bar */
# .stProgress > div > div {
#     background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
#     border-radius: 10px !important;
# }

# /* Expanders */
# .streamlit-expanderHeader {
#     background: white !important;
#     border-radius: 12px !important;
#     border: 1px solid #e2e8f0 !important;
#     transition: all 0.3s ease !important;
#     font-weight: 600 !important;
# }

# .streamlit-expanderHeader:hover {
#     background: #f8fafc !important;
#     border-color: #667eea !important;
# }

# /* Download Links */
# a {
#     color: #667eea !important;
#     text-decoration: none !important;
#     font-weight: 600 !important;
#     transition: all 0.3s ease !important;
# }

# a:hover {
#     color: #764ba2 !important;
#     text-decoration: underline !important;
# }

# /* Animations */
# @keyframes slideDown {
#     from {
#         opacity: 0;
#         transform: translateY(-30px);
#     }
#     to {
#         opacity: 1;
#         transform: translateY(0);
#     }
# }

# @keyframes fadeInUp {
#     from {
#         opacity: 0;
#         transform: translateY(20px);
#     }
#     to {
#         opacity: 1;
#         transform: translateY(0);
#     }
# }

# @keyframes pulse {
#     0%, 100% {
#         opacity: 1;
#     }
#     50% {
#         opacity: 0.7;
#     }
# }

# /* Spinner */
# .stSpinner > div {
#     border-top-color: #667eea !important;
# }

# /* Text Area */
# .stTextArea textarea {
#     border-radius: 12px !important;
#     border: 2px solid #e2e8f0 !important;
#     font-family: 'Inter', sans-serif !important;
# }

# .stTextArea textarea:focus {
#     border-color: #667eea !important;
#     box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
# }

# /* Images */
# img {
#     border-radius: 15px !important;
#     box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
# }

# /* Hide Streamlit Branding */
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# header {visibility: hidden;}
# </style>
# """, unsafe_allow_html=True)


# # ═══════════════════════════════════════════════════════════════════════
# # HEADER
# # ═══════════════════════════════════════════════════════════════════════
# st.markdown("""
# <div class="app-header">
#     <div class="app-title">📊 Financial Insights AI</div>
#     <div class="app-subtitle">Generate investor-grade thematic reports powered by AI</div>
# </div>
# """, unsafe_allow_html=True)


# # ═══════════════════════════════════════════════════════════════════════
# # SIDEBAR
# # ═══════════════════════════════════════════════════════════════════════
# with st.sidebar:
#     st.markdown("### ⚙️ Configuration")
#     st.markdown("---")
    
#     from_date = st.date_input(
#         "📅 From Date",
#         pd.to_datetime("2023-01-01"),
#         help="Start date for article scraping"
#     )
    
#     to_date = st.date_input(
#         "📅 To Date",
#         pd.to_datetime("today"),
#         help="End date for article scraping"
#     )
    
#     page_size = st.number_input(
#         "📰 Articles Per Fetch",
#         min_value=2,
#         max_value=500,
#         value=default_page_size if default_page_size >= 2 else 10,
#         step=10,
#         help="Number of articles to fetch (minimum 2 for clustering)"
#     )
    
#     st.markdown("---")
    
#     # System Status
#     st.markdown("### 📊 System Status")
    
#     cuda_status = "🟢 CUDA Available" if torch.cuda.is_available() else "🟡 CPU Mode"
#     device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
#     st.markdown(f"""
#     <div class="info-card">
#         <strong>Compute:</strong> {cuda_status}<br>
#         <strong>Device:</strong> {device_name}
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("---")
    
#     # Quick Info
#     st.markdown("### 💡 Pipeline Steps")
#     st.markdown("""
#     <div style='font-size: 0.9rem; line-height: 1.8;'>
#     1️⃣ Scrape financial news<br>
#     2️⃣ Summarize articles<br>
#     3️⃣ Cluster by themes<br>
#     4️⃣ Generate insights<br>
#     5️⃣ Create PDF report<br>
#     6️⃣ LLM evaluation
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("---")
#     st.caption("© 2025 Zeineb Moalla")
#     st.caption("Financial Data Science")


# # ═══════════════════════════════════════════════════════════════════════
# # BANNER IMAGE
# # ═══════════════════════════════════════════════════════════════════════
# image_path = Path(r'../images/fin_insights.png')
# if image_path.exists():
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         st.image(str(image_path), use_container_width=True)


# # ═══════════════════════════════════════════════════════════════════════
# # HELPER FUNCTIONS
# # ═══════════════════════════════════════════════════════════════════════
# def find_best_k(embeddings, k_min=2, k_max=None):
#     """
#     Hybrid method: Compute WCSS + Silhouette, detect elbow, choose best k
#     """
#     n = embeddings.shape[0]
    
#     if k_max is None:
#         k_max = min(12, max(4, int(np.sqrt(max(4, n)) * 2)))
    
#     k_max = min(k_max, n - 1)
    
#     if n <= k_min:
#         raise ValueError(f"Not enough samples ({n}) for clustering with k_min={k_min}")
    
#     k_values = list(range(k_min, k_max + 1))
#     wcss = []
#     silhouettes = []
    
#     for k in k_values:
#         km = KMeans(n_clusters=k, random_state=42, n_init=10)
#         km.fit(embeddings)
#         wcss.append(km.inertia_)
#         sil = silhouette_score(embeddings, km.labels_) if k > 1 else float("nan")
#         silhouettes.append(sil)
    
#     metrics_df = pd.DataFrame({
#         "k": k_values,
#         "wcss": wcss,
#         "silhouette": silhouettes
#     })
    
#     try:
#         kl = KneeLocator(k_values, wcss, curve="convex", direction="decreasing")
#         elbow_k = kl.elbow
#     except Exception:
#         elbow_k = None
    
#     silhouette_k = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])
    
#     if elbow_k is not None:
#         best_k = int(round((elbow_k + silhouette_k) / 2))
#     else:
#         best_k = silhouette_k
    
#     # Plot WCSS
#     fig1, ax1 = plt.subplots(figsize=(10, 6))
#     ax1.plot(k_values, wcss, marker='o', linewidth=2, markersize=8, color='#667eea')
#     ax1.set_xlabel("Number of Clusters (k)", fontsize=12, fontweight='bold')
#     ax1.set_ylabel("WCSS (Inertia)", fontsize=12, fontweight='bold')
#     ax1.set_title("Elbow Method - WCSS", fontsize=14, fontweight='bold')
#     ax1.grid(True, alpha=0.3)
#     if elbow_k is not None:
#         ax1.axvline(elbow_k, color='red', linestyle='--', linewidth=2, label=f'Elbow at k={elbow_k}')
#         ax1.legend(fontsize=10)
#     elbow_plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
#     fig1.savefig(elbow_plot_path.name, bbox_inches='tight', dpi=150)
#     plt.close(fig1)
    
#     # Plot Silhouette
#     fig2, ax2 = plt.subplots(figsize=(10, 6))
#     ax2.plot(k_values, silhouettes, marker='o', linewidth=2, markersize=8, color='#764ba2')
#     ax2.set_xlabel("Number of Clusters (k)", fontsize=12, fontweight='bold')
#     ax2.set_ylabel("Silhouette Score", fontsize=12, fontweight='bold')
#     ax2.set_title("Silhouette Scores", fontsize=14, fontweight='bold')
#     ax2.grid(True, alpha=0.3)
#     ax2.axvline(silhouette_k, color='green', linestyle='--', linewidth=2, label=f'Max at k={silhouette_k}')
#     ax2.legend(fontsize=10)
#     silhouette_plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
#     fig2.savefig(silhouette_plot_path.name, bbox_inches='tight', dpi=150)
#     plt.close(fig2)
    
#     return best_k, elbow_plot_path.name, silhouette_plot_path.name, metrics_df


# # ═══════════════════════════════════════════════════════════════════════
# # MAIN PIPELINE BUTTON
# # ═══════════════════════════════════════════════════════════════════════
# st.markdown("<br>", unsafe_allow_html=True)

# col1, col2, col3 = st.columns([1, 2, 1])
# with col2:
#     run_pipeline = st.button(
#         "🚀 Run Full AI Pipeline",
#         type='primary',
#         use_container_width=True
#     )

# if run_pipeline:
#     st.session_state.pipeline_running = True
    
#     # ═══════════════════════════════════════════════════════════════════
#     # STEP 1: SCRAPE ARTICLES
#     # ═══════════════════════════════════════════════════════════════════
#     st.markdown("""
#     <div class="step-card">
#         <div class="step-header">
#             <div class="step-number">1</div>
#             <span>Scraping Financial News</span>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     with st.spinner("📰 Fetching articles from The Guardian..."):
#         endpoint = "https://content.guardianapis.com/search"
#         params = {
#             "q": "financial news",
#             "section": "business",
#             "from-date": from_date.isoformat(),
#             "to-date": to_date.isoformat(),
#             "show-fields": "bodyText,webPublicationDate",
#             "page-size": page_size,
#             "api-key": guardian_api_key
#         }
#         resp = requests.get(endpoint, params=params)
#         resp.raise_for_status()
#         articles = resp.json()["response"]["results"]
#         df = pd.DataFrame([
#             {"publicationDate": pd.to_datetime(a["webPublicationDate"]).tz_localize(None),
#              "article_content": a["fields"]["bodyText"]}
#             for a in articles
#         ]).sort_values("publicationDate", ascending=False)
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("📰 Articles Scraped", len(df))
#     with col2:
#         st.metric("📅 Date Range", f"{from_date} to {to_date}")
#     with col3:
#         st.metric("🔍 Source", "The Guardian")
    
#     with st.expander("📋 View Scraped Articles"):
#         st.dataframe(df[["publicationDate"]].head(20), use_container_width=True)
    
#     # ═══════════════════════════════════════════════════════════════════
#     # STEP 2: SUMMARIZATION
#     # ═══════════════════════════════════════════════════════════════════
#     st.markdown("""
#     <div class="step-card">
#         <div class="step-header">
#             <div class="step-number">2</div>
#             <span>AI-Powered Summarization</span>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     with st.spinner("🤖 Summarizing articles with BART..."):
#         summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt", device=0 if torch.cuda.is_available() else -1)
#         dataset = Dataset.from_pandas(df)
        
#         def summarize_batch(batch):
#             texts = [t[:4000] for t in batch["article_content"]]
#             results = summarizer(texts, max_length=150, min_length=40, do_sample=False, batch_size=8)
#             batch["summary"] = [r["summary_text"] for r in results]
#             return batch
        
#         dataset = dataset.map(summarize_batch, batched=True, batch_size=8)
#         df = dataset.to_pandas()
    
#     st.success(f"✅ Successfully summarized {len(df)} articles")
    
#     with st.expander("📝 View Sample Summaries"):
#         for idx in range(min(3, len(df))):
#             st.markdown(f"**Summary {idx + 1}:**")
#             st.info(df.iloc[idx]["summary"])
    
#     # ═══════════════════════════════════════════════════════════════════
#     # STEP 3: CLUSTERING
#     # ═══════════════════════════════════════════════════════════════════
#     st.markdown("""
#     <div class="step-card">
#         <div class="step-header">
#             <div class="step-number">3</div>
#             <span>Intelligent Clustering & Theme Detection</span>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     with st.spinner("🧠 Generating embeddings and clustering..."):
#         embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#         embeddings = embedder.encode(df["summary"].tolist(), batch_size=64, convert_to_numpy=True, show_progress_bar=True)
#         best_k, elbow_plot_path, silhouette_plot_path, metrics_df = find_best_k(embeddings, k_min=2, k_max=10)
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(elbow_plot_path, caption=f"Elbow Method Analysis (Selected k={best_k})", use_container_width=True)
#     with col2:
#         st.image(silhouette_plot_path, caption="Silhouette Score Analysis", use_container_width=True)
    
#     st.success(f"✅ Optimal number of themes detected: **{best_k}**")
    
#     # Fit final KMeans
#     df["theme_id"] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(embeddings)
    
#     # Clustering metrics
#     silhouette = silhouette_score(embeddings, df["theme_id"])
#     davies = davies_bouldin_score(embeddings, df["theme_id"])
#     calinski = calinski_harabasz_score(embeddings, df["theme_id"])
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("📊 Silhouette Score", f"{silhouette:.3f}", help="Higher is better (0-1)")
#     with col2:
#         st.metric("📉 Davies-Bouldin", f"{davies:.3f}", help="Lower is better")
#     with col3:
#         st.metric("📈 Calinski-Harabasz", f"{calinski:.1f}", help="Higher is better")
    
#     # Save metrics
#     metrics = {
#         "silhouette_score": silhouette,
#         "davies_bouldin_index": davies,
#         "calinski_harabasz_index": calinski,
#         "best_k": int(best_k)
#     }
#     metrics_summary_df = pd.DataFrame([metrics])
#     metrics_excel_path = "clustering_evaluation_metrics.xlsx"
#     metrics_summary_df.to_excel(metrics_excel_path, index=False)
    
#     with open(metrics_excel_path, "rb") as f:
#         b64 = base64.b64encode(f.read()).decode()
#         st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="clustering_metrics.xlsx">⬇️ Download Clustering Metrics</a>', unsafe_allow_html=True)
    
#     # t-SNE Visualization
#     st.markdown("#### 🎨 t-SNE Cluster Visualization")
    
#     n_samples = embeddings.shape[0]
#     if n_samples > 2:
#         with st.spinner("Generating t-SNE visualization..."):
#             adjusted_perplexity = min(15, n_samples - 1)
#             tsne = TSNE(n_components=2, random_state=42, perplexity=adjusted_perplexity, init="random")
#             reduced = tsne.fit_transform(embeddings)
            
#             fig_tsne, ax_tsne = plt.subplots(figsize=(12, 8))
#             scatter = ax_tsne.scatter(
#                 reduced[:, 0],
#                 reduced[:, 1],
#                 c=df["theme_id"],
#                 cmap="tab10",
#                 s=100,
#                 alpha=0.7,
#                 edgecolors='white',
#                 linewidths=1
#             )
#             ax_tsne.set_title(f"t-SNE Projection of {best_k} Themes", fontsize=16, fontweight='bold')
#             ax_tsne.grid(True, alpha=0.3)
#             legend1 = ax_tsne.legend(*scatter.legend_elements(), title="Theme", loc="best")
#             ax_tsne.add_artist(legend1)
            
#             tsne_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
#             fig_tsne.savefig(tsne_path.name, bbox_inches="tight", dpi=150)
#             plt.close(fig_tsne)
            
#             st.image(tsne_path.name, use_container_width=True)
#     else:
#         st.warning("⚠️ Not enough articles for t-SNE visualization (need at least 3)")
    
#     # ═══════════════════════════════════════════════════════════════════
#     # STEP 4: THEME LABELING
#     # ═══════════════════════════════════════════════════════════════════
#     st.markdown("""
#     <div class="step-card">
#         <div class="step-header">
#             <div class="step-number">4</div>
#             <span>Generating Human-Readable Theme Names</span>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     theme_names = {}
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     for idx, tid in enumerate(sorted(df["theme_id"].unique())):
#         status_text.text(f"Labeling theme {idx + 1}/{best_k}...")
#         joined = "\n\n".join(df[df["theme_id"] == tid]["summary"].tolist())
#         prompt = f"""
#         You are a financial analyst. Given these summaries, suggest a concise (3–5 word) label capturing their theme.
#         Return only the theme name as plain text without quotes or formatting.
        
#         Summaries:
#         {joined}
#         """
#         res = client.chat.completions.create(
#             model="deepseek/deepseek-chat-v3.1:free",
#             messages=[{"role": "user", "content": prompt}],
#             extra_headers={"HTTP-Referer": "https://openrouter.ai", "X-Title": "Theme Label"}
#         )
#         raw_output = res.choices[0].message.content.strip()
#         clean_output = re.sub(r'\\boxed\{["\']?(.*?)["\']?\}', r'\1', raw_output)
#         theme_names[tid] = clean_output.strip()
#         progress_bar.progress((idx + 1) / best_k)
    
#     status_text.empty()
#     progress_bar.empty()
    
#     st.success("✅ Theme labels generated successfully!")
    
#     st.markdown("#### 🏷️ Detected Themes:")
#     for tid, label in theme_names.items():
#         article_count = len(df[df["theme_id"] == tid])
#         st.markdown(f"""
#         <div class="info-card">
#             <strong>Theme {tid + 1}:</strong> {label}<br>
#             <small>📰 {article_count} articles</small>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Save clustered data
#     df["theme"] = df["theme_id"].map(theme_names)
#     labeled_excel_path = "clustered_articles_with_themes.xlsx"
#     df[["article_content", "summary", "theme"]].to_excel(labeled_excel_path, index=False)
    
#     with open(labeled_excel_path, "rb") as f:
#         b64 = base64.b64encode(f.read()).decode()
#         st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="clustered_articles.xlsx">⬇️ Download Clustered Articles</a>', unsafe_allow_html=True)
    
#     # ═══════════════════════════════════════════════════════════════════
#     # STEP 5: GENERATE INSIGHTS & PDF
#     # ═══════════════════════════════════════════════════════════════════
#     st.markdown("""
#     <div class="step-card">
#         <div class="step-header">
#             <div class="step-number">5</div>
#             <span>Generating Investor Report & PDF</span>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     def generate_insight(texts, tid):
#         prompt = f"""
#         Generate a report section titled '{theme_names.get(tid, f"Theme {tid + 1}")}'.
        
#         Based on these summaries:
#         "{texts}"
        
#         Write a structured analysis with:
#         **Executive Summary**
#         **Key Trends**
#         **Implications for Investors**
#         """
#         r = client.chat.completions.create(
#             model="deepseek/deepseek-chat-v3.1:free",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return r.choices[0].message.content.strip()
    
#     with st.spinner("📄 Generating PDF report..."):
#         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
#         c = canvas.Canvas(tmp.name, pagesize=letter)
#         width, height = letter
#         c.setFont("Helvetica-Bold", 18)
#         c.drawCentredString(width / 2, height - 50, "Investor-Grade Thematic Report")
#         c.setFont("Helvetica", 12)
#         c.drawCentredString(width / 2, height - 70, f"Generated on {datetime.now().strftime('%B %d, %Y')}")
#         y = height - 100
        
#         all_content = []
#         for tid in sorted(df["theme_id"].unique()):
#             summaries = df[df["theme_id"] == tid]["summary"].tolist()
#             content = generate_insight("\n\n".join(summaries), tid)
#             all_content.append(content)
            
#             theme_title = theme_names.get(tid, f"Theme {tid + 1}")
#             c.setFont("Helvetica-Bold", 14)
#             c.drawString(40, y, theme_title)
#             y -= 20
#             c.setFont("Helvetica", 11)
#             for line in content.split("\n"):
#                 for subline in textwrap.wrap(line, 95):
#                     if y < 50:
#                         c.showPage()
#                         c.setFont("Helvetica", 11)
#                         y = height - 50
#                     c.drawString(40, y, subline)
#                     y -= 14
#                 y -= 8
#             y -= 16
        
#         c.save()
    
#     st.success("✅ PDF report generated successfully!")
    
#     with open(tmp.name, "rb") as f:
#         b64 = base64.b64encode(f.read()).decode()
#         st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="investor_report.pdf">📥 Download Full PDF Report</a>', unsafe_allow_html=True)
    
#     # ═══════════════════════════════════════════════════════════════════
#     # STEP 6: LLM EVALUATION
#     # ═══════════════════════════════════════════════════════════════════
#     st.markdown("""
#     <div class="step-card">
#         <div class="step-header">
#             <div class="step-number">6</div>
#             <span>AI Quality Evaluation (LLM-as-a-Judge)</span>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     with st.spinner("🎯 Evaluating report quality..."):
#         report_content = "\n\n".join(all_content)
        
#         judge_prompt = f"""
#         You are an expert financial report evaluator.
#         Evaluate this investor report on:
#         - Coherence (1-10)
#         - Accuracy of insights (1-10)
#         - Value to investors (1-10)
#         - Clarity of writing (1-10)
        
#         Provide scores and brief justifications.
        
#         Report:
#         {report_content}
#         """
        
#         judge_res = client.chat.completions.create(
#             model="google/gemma-3n-e2b-it:free",
#             messages=[{"role": "user", "content": judge_prompt}]
#         )
#         evaluation_text = judge_res.choices[0].message.content.strip()
    
#     st.markdown("#### 📊 Quality Evaluation Results:")
#     st.text_area("", evaluation_text, height=300)
    
#     # Parse and save evaluation
#     eval_data = []
#     for line in evaluation_text.split("\n"):
#         match = re.match(r"^(.*?):\s*([0-9]+)/?10?.*", line.strip(), re.IGNORECASE)
#         if match:
#             criterion = match.group(1).strip()
#             score = match.group(2).strip()
#             eval_data.append({"criterion": criterion, "score": score})
#     if not eval_data:
#         eval_data.append({"criterion": "Full Evaluation", "score": evaluation_text})
    
#     eval_df = pd.DataFrame(eval_data)
#     eval_excel_path = "llm_evaluation.xlsx"
#     eval_df.to_excel(eval_excel_path, index=False)
    
#     with open(eval_excel_path, "rb") as f:
#         b64 = base64.b64encode(f.read()).decode()
#         st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="quality_evaluation.xlsx">⬇️ Download Evaluation Report</a>', unsafe_allow_html=True)
    
#     st.session_state.pipeline_running = False
#     st.balloons()


# # ═══════════════════════════════════════════════════════════════════════
# # FOOTER
# # ═══════════════════════════════════════════════════════════════════════
# st.markdown("<br><br>", unsafe_allow_html=True)
# st.markdown("""
# <div style='text-align: center; padding: 2rem; border-top: 2px solid #e2e8f0;'>
#     <p style='color: #667eea; font-weight: 600; font-size: 1.1rem;'>© 2025 Zeineb Moalla | Financial Data Science Master Student</p>
#     <p style='color: #94a3b8; font-size: 0.9rem;'>Powered by AI • Built with ❤️</p>
# </div>
# """, unsafe_allow_html=True)


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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from kneed import KneeLocator
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from pathlib import Path
import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re
import torch
from datasets import Dataset
from datetime import datetime


# Load environment variables
load_dotenv()
router_api_key = os.getenv("OPENROUTER_API_KEY")
guardian_api_key = os.getenv("GUARDIAN_API_KEY")
default_page_size = int(os.getenv("GUARDIAN_PAGE_SIZE", 5))


if not router_api_key:
    raise ValueError("Missing OPENROUTER_API_KEY in .env file.")
if not guardian_api_key:
    raise ValueError("Missing GUARDIAN_API_KEY in .env file.")


# OpenRouter client
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=router_api_key)


# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Financial Insights AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ═══════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "results" not in st.session_state:
    st.session_state.results = {}


# ═══════════════════════════════════════════════════════════════════════
# ENHANCED CSS STYLING
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* Main Background */
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
}

.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

/* Header Container */
.app-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    text-align: center;
    animation: slideDown 0.6s ease-out;
}

.app-title {
    font-size: 3rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.app-subtitle {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1.2rem;
    font-weight: 400;
}

/* Cards */
.info-card {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
    border-left: 4px solid #667eea;
}

.info-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
}

/* Step Cards */
.step-card {
    background: white;
    border-radius: 15px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    border-top: 5px solid #667eea;
    animation: fadeInUp 0.5s ease-out;
}

.step-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #667eea;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.step-number {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.2rem;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2.5rem !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    cursor: pointer !important;
}

.stButton button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6) !important;
}

.stButton button:active {
    transform: translateY(-1px) !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 1rem;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar Enhanced Glassmorphism Cards */
section[data-testid="stSidebar"] .stMarkdown {
    color: white !important;
}

section[data-testid="stSidebar"] h3 {
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.2) !important;
    margin: 1rem 0 !important;
}

/* Date Input Styling in Sidebar */
section[data-testid="stSidebar"] .stDateInput label {
    color: white !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] .stDateInput input {
    color: #2d3748 !important;
    background: white !important;
    border-radius: 8px !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    font-weight: 600 !important;
}

/* Number Input Styling in Sidebar */
section[data-testid="stSidebar"] .stNumberInput label {
    color: white !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] .stNumberInput input {
    color: #2d3748 !important;
    background: white !important;
    border-radius: 8px !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    font-weight: 600 !important;
}

/* Sidebar Text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] span {
    color: white !important;
}

/* Success/Error Messages */
.stSuccess {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    border: none !important;
}

.stError {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    border: none !important;
}

.stInfo {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    border: none !important;
}

.stWarning {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    border: none !important;
}

/* Dataframe Styling */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08) !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #667eea !important;
}

[data-testid="stMetricLabel"] {
    font-weight: 600 !important;
    color: #4b5563 !important;
}

/* Progress Bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    border-radius: 10px !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: white !important;
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    transition: all 0.3s ease !important;
    font-weight: 600 !important;
}

.streamlit-expanderHeader:hover {
    background: #f8fafc !important;
    border-color: #667eea !important;
}

/* Download Links */
a {
    color: #667eea !important;
    text-decoration: none !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

a:hover {
    color: #764ba2 !important;
    text-decoration: underline !important;
}

/* Animations */
@keyframes slideDown {
    from {
        opacity: 0;
        transform: translateY(-30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.7;
    }
}

/* Spinner */
.stSpinner > div {
    border-top-color: #667eea !important;
}

/* Text Area */
.stTextArea textarea {
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

.stTextArea textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Images */
img {
    border-radius: 15px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="app-title">📊 Financial Insights AI</div>
    <div class="app-subtitle">Generate investor-grade thematic reports powered by AI</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    from_date = st.date_input(
        "📅 From Date",
        pd.to_datetime("2023-01-01"),
        help="Start date for article scraping"
    )
    
    to_date = st.date_input(
        "📅 To Date",
        pd.to_datetime("today"),
        help="End date for article scraping"
    )
    
    page_size = st.number_input(
        "📰 Articles Per Fetch",
        min_value=2,
        max_value=500,
        value=default_page_size if default_page_size >= 2 else 10,
        step=10,
        help="Number of articles to fetch (minimum 2 for clustering)"
    )
    
    st.markdown("---")
    
    # System Status with FIXED STYLING
    st.markdown("### 📊 System Status")
    
    cuda_status = "🟢 CUDA Available" if torch.cuda.is_available() else "🟡 CPU Mode"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    st.markdown(f"""
    <div style='
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
        font-weight: 500;
    '>
        <div style='margin-bottom: 0.5rem;'>
            <strong style='color: white;'>Compute:</strong> 
            <span style='color: white;'>{cuda_status}</span>
        </div>
        <div>
            <strong style='color: white;'>Device:</strong> 
            <span style='color: white;'>{device_name}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # API Status Indicators
    st.markdown("### 🔑 API Status")
    
    api_status = "🟢 Connected" if router_api_key else "🔴 Disconnected"
    guardian_status = "🟢 Active" if guardian_api_key else "🔴 Inactive"
    
    st.markdown(f"""
    <div style='
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
    '>
        <div style='margin-bottom: 0.5rem;'>
            <strong style='color: white;'>OpenRouter:</strong> 
            <span style='color: white;'>{api_status}</span>
        </div>
        <div>
            <strong style='color: white;'>Guardian API:</strong> 
            <span style='color: white;'>{guardian_status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Info with Icons
    st.markdown("### 💡 Pipeline Steps")
    st.markdown("""
    <div style='
        font-size: 0.9rem; 
        line-height: 2;
        color: white;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        backdrop-filter: blur(10px);
    '>
        <div style='margin: 0.5rem 0;'>1️⃣ <strong>Scrape</strong> financial news</div>
        <div style='margin: 0.5rem 0;'>2️⃣ <strong>Summarize</strong> articles</div>
        <div style='margin: 0.5rem 0;'>3️⃣ <strong>Cluster</strong> by themes</div>
        <div style='margin: 0.5rem 0;'>4️⃣ <strong>Generate</strong> insights</div>
        <div style='margin: 0.5rem 0;'>5️⃣ <strong>Create</strong> PDF report</div>
        <div style='margin: 0.5rem 0;'>6️⃣ <strong>Evaluate</strong> quality</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Info
    st.markdown("### 🤖 AI Models")
    st.markdown("""
    <div style='
        font-size: 0.85rem;
        color: white;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        backdrop-filter: blur(10px);
    '>
        <div style='margin: 0.3rem 0;'>
            <strong>Summarizer:</strong><br>
            <span style='opacity: 0.9;'>BART-Large-CNN</span>
        </div>
        <div style='margin: 0.5rem 0;'></div>
        <div style='margin: 0.3rem 0;'>
            <strong>Embeddings:</strong><br>
            <span style='opacity: 0.9;'>MiniLM-L6-v2</span>
        </div>
        <div style='margin: 0.5rem 0;'></div>
        <div style='margin: 0.3rem 0;'>
            <strong>Clustering:</strong><br>
            <span style='opacity: 0.9;'>K-Means</span>
        </div>
        <div style='margin: 0.5rem 0;'></div>
        <div style='margin: 0.3rem 0;'>
            <strong>LLM:</strong><br>
            <span style='opacity: 0.9;'>DeepSeek Chat v3.1</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tips Section with Better Styling
    st.markdown("### 💎 Pro Tips")
    st.markdown("""
    <div style='
        font-size: 0.85rem;
        color: white;
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%);
        border-radius: 12px;
        padding: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    '>
        <div style='margin: 0.5rem 0;'>
            ✨ <strong>Larger datasets</strong> = better clusters
        </div>
        <div style='margin: 0.5rem 0;'>
            ⚡ <strong>CUDA</strong> = faster processing
        </div>
        <div style='margin: 0.5rem 0;'>
            🎯 <strong>Date range</strong> impacts themes
        </div>
        <div style='margin: 0.5rem 0;'>
            📊 <strong>Monitor</strong> evaluation scores
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Footer with better styling
    st.markdown("""
    <div style='
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.85rem;
        padding: 1rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    '>
        <div style='font-weight: 600; margin-bottom: 0.5rem;'>
            © 2025 Zeineb Moalla
        </div>
        <div style='opacity: 0.8;'>
            Financial Data Science
        </div>
        <div style='margin-top: 0.5rem; opacity: 0.7; font-size: 0.75rem;'>
            v1.0 • Powered by AI
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# BANNER IMAGE
# ═══════════════════════════════════════════════════════════════════════
image_path = Path(r'../images/fin_insights.png')
if image_path.exists():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(str(image_path), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════
def find_best_k(embeddings, k_min=2, k_max=None):
    """
    Hybrid method: Compute WCSS + Silhouette, detect elbow, choose best k
    """
    n = embeddings.shape[0]
    
    if k_max is None:
        k_max = min(12, max(4, int(np.sqrt(max(4, n)) * 2)))
    
    k_max = min(k_max, n - 1)
    
    if n <= k_min:
        raise ValueError(f"Not enough samples ({n}) for clustering with k_min={k_min}")
    
    k_values = list(range(k_min, k_max + 1))
    wcss = []
    silhouettes = []
    
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(embeddings)
        wcss.append(km.inertia_)
        sil = silhouette_score(embeddings, km.labels_) if k > 1 else float("nan")
        silhouettes.append(sil)
    
    metrics_df = pd.DataFrame({
        "k": k_values,
        "wcss": wcss,
        "silhouette": silhouettes
    })
    
    try:
        kl = KneeLocator(k_values, wcss, curve="convex", direction="decreasing")
        elbow_k = kl.elbow
    except Exception:
        elbow_k = None
    
    silhouette_k = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])
    
    if elbow_k is not None:
        best_k = int(round((elbow_k + silhouette_k) / 2))
    else:
        best_k = silhouette_k
    
    # Plot WCSS
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(k_values, wcss, marker='o', linewidth=2, markersize=8, color='#667eea')
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("WCSS (Inertia)", fontsize=12, fontweight='bold')
    ax1.set_title("Elbow Method - WCSS", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if elbow_k is not None:
        ax1.axvline(elbow_k, color='red', linestyle='--', linewidth=2, label=f'Elbow at k={elbow_k}')
        ax1.legend(fontsize=10)
    elbow_plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig1.savefig(elbow_plot_path.name, bbox_inches='tight', dpi=150)
    plt.close(fig1)
    
    # Plot Silhouette
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(k_values, silhouettes, marker='o', linewidth=2, markersize=8, color='#764ba2')
    ax2.set_xlabel("Number of Clusters (k)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Silhouette Score", fontsize=12, fontweight='bold')
    ax2.set_title("Silhouette Scores", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(silhouette_k, color='green', linestyle='--', linewidth=2, label=f'Max at k={silhouette_k}')
    ax2.legend(fontsize=10)
    silhouette_plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig2.savefig(silhouette_plot_path.name, bbox_inches='tight', dpi=150)
    plt.close(fig2)
    
    return best_k, elbow_plot_path.name, silhouette_plot_path.name, metrics_df


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE BUTTON
# ═══════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_pipeline = st.button(
        "🚀 Run Full AI Pipeline",
        type='primary',
        use_container_width=True
    )

if run_pipeline:
    st.session_state.pipeline_running = True
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: SCRAPE ARTICLES
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="step-card">
        <div class="step-header">
            <div class="step-number">1</div>
            <span>Scraping Financial News</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("📰 Fetching articles from The Guardian..."):
        endpoint = "https://content.guardianapis.com/search"
        params = {
            "q": "financial news",
            "section": "business",
            "from-date": from_date.isoformat(),
            "to-date": to_date.isoformat(),
            "show-fields": "bodyText,webPublicationDate",
            "page-size": page_size,
            "api-key": guardian_api_key
        }
        resp = requests.get(endpoint, params=params)
        resp.raise_for_status()
        articles = resp.json()["response"]["results"]
        df = pd.DataFrame([
            {"publicationDate": pd.to_datetime(a["webPublicationDate"]).tz_localize(None),
             "article_content": a["fields"]["bodyText"]}
            for a in articles
        ]).sort_values("publicationDate", ascending=False)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📰 Articles Scraped", len(df))
    with col2:
        st.metric("📅 Date Range", f"{from_date} to {to_date}")
    with col3:
        st.metric("🔍 Source", "The Guardian")
    
    with st.expander("📋 View Scraped Articles"):
        st.dataframe(df[["publicationDate"]].head(20), use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: SUMMARIZATION
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="step-card">
        <div class="step-header">
            <div class="step-number">2</div>
            <span>AI-Powered Summarization</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🤖 Summarizing articles with BART..."):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt", device=0 if torch.cuda.is_available() else -1)
        dataset = Dataset.from_pandas(df)
        
        def summarize_batch(batch):
            texts = [t[:4000] for t in batch["article_content"]]
            results = summarizer(texts, max_length=150, min_length=40, do_sample=False, batch_size=8)
            batch["summary"] = [r["summary_text"] for r in results]
            return batch
        
        dataset = dataset.map(summarize_batch, batched=True, batch_size=8)
        df = dataset.to_pandas()
    
    st.success(f"✅ Successfully summarized {len(df)} articles")
    
    with st.expander("📝 View Sample Summaries"):
        for idx in range(min(3, len(df))):
            st.markdown(f"**Summary {idx + 1}:**")
            st.info(df.iloc[idx]["summary"])
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: CLUSTERING
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="step-card">
        <div class="step-header">
            <div class="step-number">3</div>
            <span>Intelligent Clustering & Theme Detection</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🧠 Generating embeddings and clustering..."):
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = embedder.encode(df["summary"].tolist(), batch_size=64, convert_to_numpy=True, show_progress_bar=True)
        best_k, elbow_plot_path, silhouette_plot_path, metrics_df = find_best_k(embeddings, k_min=2, k_max=10)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(elbow_plot_path, caption=f"Elbow Method Analysis (Selected k={best_k})", use_container_width=True)
    with col2:
        st.image(silhouette_plot_path, caption="Silhouette Score Analysis", use_container_width=True)
    
    st.success(f"✅ Optimal number of themes detected: **{best_k}**")
    
    # Fit final KMeans
    df["theme_id"] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(embeddings)
    
    # Clustering metrics
    silhouette = silhouette_score(embeddings, df["theme_id"])
    davies = davies_bouldin_score(embeddings, df["theme_id"])
    calinski = calinski_harabasz_score(embeddings, df["theme_id"])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Silhouette Score", f"{silhouette:.3f}", help="Higher is better (0-1)")
    with col2:
        st.metric("📉 Davies-Bouldin", f"{davies:.3f}", help="Lower is better")
    with col3:
        st.metric("📈 Calinski-Harabasz", f"{calinski:.1f}", help="Higher is better")
    
    # Save metrics
    metrics = {
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies,
        "calinski_harabasz_index": calinski,
        "best_k": int(best_k)
    }
    metrics_summary_df = pd.DataFrame([metrics])
    metrics_excel_path = "clustering_evaluation_metrics.xlsx"
    metrics_summary_df.to_excel(metrics_excel_path, index=False)
    
    with open(metrics_excel_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="clustering_metrics.xlsx">⬇️ Download Clustering Metrics</a>', unsafe_allow_html=True)
    
    # t-SNE Visualization
    # t-SNE Visualization
    st.markdown("#### 🎨 t-SNE Cluster Visualization")

    n_samples = embeddings.shape[0]
    if n_samples > 2:
        with st.spinner("Generating t-SNE visualization..."):
            adjusted_perplexity = min(15, n_samples - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=adjusted_perplexity, init="random")
            reduced = tsne.fit_transform(embeddings)
            
            fig_tsne, ax_tsne = plt.subplots(figsize=(12, 8))
            scatter = ax_tsne.scatter(
                reduced[:, 0],
                reduced[:, 1],
                c=df["theme_id"],
                cmap="tab10",
                s=100,
                alpha=0.7,
                edgecolors='white',
                linewidths=1
            )
            ax_tsne.set_title(f"t-SNE Projection of {best_k} Themes", fontsize=16, fontweight='bold')
            ax_tsne.grid(True, alpha=0.3)
            legend1 = ax_tsne.legend(*scatter.legend_elements(), title="Theme", loc="best")
            ax_tsne.add_artist(legend1)
            
            tsne_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig_tsne.savefig(tsne_path.name, bbox_inches="tight", dpi=150)
            plt.close(fig_tsne)
            
            st.image(tsne_path.name, use_container_width=True)  # ✅ FIXED
    else:
        st.warning("⚠️ Not enough articles for t-SNE visualization (need at least 3)")

    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: THEME LABELING
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="step-card">
        <div class="step-header">
            <div class="step-number">4</div>
            <span>Generating Human-Readable Theme Names</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    theme_names = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, tid in enumerate(sorted(df["theme_id"].unique())):
        status_text.text(f"Labeling theme {idx + 1}/{best_k}...")
        joined = "\n\n".join(df[df["theme_id"] == tid]["summary"].tolist())
        prompt = f"""
        You are a financial analyst. Given these summaries, suggest a concise (3–5 word) label capturing their theme.
        Return only the theme name as plain text without quotes or formatting.
        
        Summaries:
        {joined}
        """
        res = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[{"role": "user", "content": prompt}],
            extra_headers={"HTTP-Referer": "https://openrouter.ai", "X-Title": "Theme Label"}
        )
        raw_output = res.choices[0].message.content.strip()
        clean_output = re.sub(r'\\boxed\{["\']?(.*?)["\']?\}', r'\1', raw_output)
        theme_names[tid] = clean_output.strip()
        progress_bar.progress((idx + 1) / best_k)
    
    status_text.empty()
    progress_bar.empty()
    
    st.success("✅ Theme labels generated successfully!")
    
    st.markdown("#### 🏷️ Detected Themes:")
    for tid, label in theme_names.items():
        article_count = len(df[df["theme_id"] == tid])
        st.markdown(f"""
        <div class="info-card">
            <strong>Theme {tid + 1}:</strong> {label}<br>
            <small>📰 {article_count} articles</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Save clustered data
    df["theme"] = df["theme_id"].map(theme_names)
    labeled_excel_path = "clustered_articles_with_themes.xlsx"
    df[["article_content", "summary", "theme"]].to_excel(labeled_excel_path, index=False)
    
    with open(labeled_excel_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="clustered_articles.xlsx">⬇️ Download Clustered Articles</a>', unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: GENERATE INSIGHTS & PDF
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="step-card">
        <div class="step-header">
            <div class="step-number">5</div>
            <span>Generating Investor Report & PDF</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    def generate_insight(texts, tid):
        prompt = f"""
        Generate a report section titled '{theme_names.get(tid, f"Theme {tid + 1}")}'.
        
        Based on these summaries:
        "{texts}"
        
        Write a structured analysis with:
        **Executive Summary**
        **Key Trends**
        **Implications for Investors**
        """
        r = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[{"role": "user", "content": prompt}]
        )
        return r.choices[0].message.content.strip()
    
    with st.spinner("📄 Generating PDF report..."):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        c = canvas.Canvas(tmp.name, pagesize=letter)
        width, height = letter
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(width / 2, height - 50, "Investor-Grade Thematic Report")
        c.setFont("Helvetica", 12)
        c.drawCentredString(width / 2, height - 70, f"Generated on {datetime.now().strftime('%B %d, %Y')}")
        y = height - 100
        
        all_content = []
        for tid in sorted(df["theme_id"].unique()):
            summaries = df[df["theme_id"] == tid]["summary"].tolist()
            content = generate_insight("\n\n".join(summaries), tid)
            all_content.append(content)
            
            theme_title = theme_names.get(tid, f"Theme {tid + 1}")
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, y, theme_title)
            y -= 20
            c.setFont("Helvetica", 11)
            for line in content.split("\n"):
                for subline in textwrap.wrap(line, 95):
                    if y < 50:
                        c.showPage()
                        c.setFont("Helvetica", 11)
                        y = height - 50
                    c.drawString(40, y, subline)
                    y -= 14
                y -= 8
            y -= 16
        
        c.save()
    
    st.success("✅ PDF report generated successfully!")
    
    with open(tmp.name, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="investor_report.pdf">📥 Download Full PDF Report</a>', unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 6: LLM EVALUATION
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="step-card">
        <div class="step-header">
            <div class="step-number">6</div>
            <span>AI Quality Evaluation (LLM-as-a-Judge)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🎯 Evaluating report quality..."):
        report_content = "\n\n".join(all_content)
        
        judge_prompt = f"""
        You are an expert financial report evaluator.
        Evaluate this investor report on:
        - Coherence (1-10)
        - Accuracy of insights (1-10)
        - Value to investors (1-10)
        - Clarity of writing (1-10)
        
        Provide scores and brief justifications.
        
        Report:
        {report_content}
        """
        
        judge_res = client.chat.completions.create(
            model="google/gemma-3n-e2b-it:free",
            messages=[{"role": "user", "content": judge_prompt}]
        )
        evaluation_text = judge_res.choices[0].message.content.strip()
    
    st.markdown("#### 📊 Quality Evaluation Results:")
    st.text_area("", evaluation_text, height=300)
    
    # Parse and save evaluation
    eval_data = []
    for line in evaluation_text.split("\n"):
        match = re.match(r"^(.*?):\s*([0-9]+)/?10?.*", line.strip(), re.IGNORECASE)
        if match:
            criterion = match.group(1).strip()
            score = match.group(2).strip()
            eval_data.append({"criterion": criterion, "score": score})
    if not eval_data:
        eval_data.append({"criterion": "Full Evaluation", "score": evaluation_text})
    
    eval_df = pd.DataFrame(eval_data)
    eval_excel_path = "llm_evaluation.xlsx"
    eval_df.to_excel(eval_excel_path, index=False)
    
    with open(eval_excel_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="quality_evaluation.xlsx">⬇️ Download Evaluation Report</a>', unsafe_allow_html=True)
    
    st.session_state.pipeline_running = False
    st.balloons()


# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 2rem; border-top: 2px solid #e2e8f0;'>
    <p style='color: #667eea; font-weight: 600; font-size: 1.1rem;'>© 2025 Zeineb Moalla | Financial Data Science Master Student</p>
    <p style='color: #94a3b8; font-size: 0.9rem;'>Powered by AI • Built with ❤️</p>
</div>
""", unsafe_allow_html=True)

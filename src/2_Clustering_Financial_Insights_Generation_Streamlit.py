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
from datasets import Dataset   # 🔹 for efficient batching

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

st.set_page_config(page_title="Investor Thematic Report Generator", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Investor Thematic Report Generator")
st.markdown("Generate polished, investor-grade PDF reports from financial news summaries.")

#CUDA check
if torch.cuda.is_available():
    st.success(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    st.error("❌ CUDA not available, running on CPU")




# Image Banner
image_path = Path(r'../images/fin_insights.png')
if image_path.exists():
    st.image(str(image_path), caption="GenAI Financial Insights", width=500)

# Sidebar Settings
st.sidebar.header("Settings")
from_date = st.sidebar.date_input("From Date", pd.to_datetime("2023-01-01"))
to_date = st.sidebar.date_input("To Date", pd.to_datetime("today"))

page_size = st.sidebar.number_input(
    "Guardian API Page Size",
    min_value=2,
    max_value=500,
    value=default_page_size if default_page_size >= 2 else 10,
    step=10,
    help="Number of articles to fetch per API call (⚠️ At least 2 needed for clustering)"
)

# =========================
# Helper: Find Best K (Hybrid: elbow (WCSS) + silhouette)
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

def find_best_k(embeddings, k_min=2, k_max=None):
    """
    Hybrid method:
      - Compute WCSS (inertia) for k in range
      - Compute silhouette for k in same range
      - Detect elbow via KneeLocator on WCSS
      - Choose best k = rounded mean of elbow_k and silhouette_k
    Returns:
      best_k, elbow_plot_path, silhouette_plot_path, metrics_df
    """
    n = embeddings.shape[0]

    # Default rule if not provided
    if k_max is None:
        k_max = min(12, max(4, int(np.sqrt(max(4, n)) * 2)))

    # 🔹 Ensure k_max never exceeds n_samples - 1
    k_max = min(k_max, n - 1)

    # If dataset too small, fall back
    if n <= k_min:
        raise ValueError(f"Not enough samples ({n}) for clustering with k_min={k_min}. "
                         f"Need at least {k_min + 1} samples.")

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

    # --- Find elbow using KneeLocator ---
    try:
        kl = KneeLocator(k_values, wcss, curve="convex", direction="decreasing")
        elbow_k = kl.elbow
    except Exception:
        elbow_k = None

    # --- Find k with max silhouette ---
    silhouette_k = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])

    # --- Compute best_k as mean of elbow_k and silhouette_k ---
    if elbow_k is not None:
        best_k = int(round((elbow_k + silhouette_k) / 2))
    else:
        best_k = silhouette_k

    # --- Plot WCSS (elbow) ---
    fig1, ax1 = plt.subplots()
    ax1.plot(k_values, wcss, marker='o')
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("WCSS (Inertia)")
    ax1.set_title("Elbow Method - WCSS (Inertia)")
    if elbow_k is not None:
        ax1.axvline(elbow_k, color='red', linestyle='--', label=f'Elbow at k={elbow_k}')
        ax1.legend()
    elbow_plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig1.savefig(elbow_plot_path.name, bbox_inches='tight')
    plt.close(fig1)

    # --- Plot Silhouette ---
    fig2, ax2 = plt.subplots()
    ax2.plot(k_values, silhouettes, marker='o')
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Scores vs k")
    ax2.axvline(silhouette_k, color='green', linestyle='--', label=f'Max silhouette k={silhouette_k}')
    ax2.legend()
    silhouette_plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig2.savefig(silhouette_plot_path.name, bbox_inches='tight')
    plt.close(fig2)

    return best_k, elbow_plot_path.name, silhouette_plot_path.name, metrics_df

# =========================
# Run Full Pipeline Button
# =========================
if st.button("🚀 Run Full Pipeline", type='primary', use_container_width=True):

    # Step 1: Scrape Articles
    st.subheader("Step 1: Scrape Financial News")
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
    st.success(f"Scraped {len(df)} articles")
    st.dataframe(df[["publicationDate"]].head(10))

    # Step 2: Summarization (Batched 🚀)
    st.subheader("Step 2: Summarize Articles")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt", device=0)  

    

    # Convert to HuggingFace Dataset for batching
    dataset = Dataset.from_pandas(df)

    def summarize_batch(batch):
        texts = [t[:4000] for t in batch["article_content"]]  # truncate
        results = summarizer(texts, max_length=150, min_length=40, do_sample=False, batch_size=8)
        batch["summary"] = [r["summary_text"] for r in results]
        return batch

    dataset = dataset.map(summarize_batch, batched=True, batch_size=8)
    df = dataset.to_pandas()
    st.success("Articles summarized")

    # Step 3: Clustering (Batched Embeddings 🚀)
    st.subheader("Step 3: Embeddings & Clustering")
    #
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(df["summary"].tolist(), batch_size=64, convert_to_numpy=True, show_progress_bar=True)

    # Use hybrid elbow + silhouette method
    best_k, elbow_plot_path, silhouette_plot_path, metrics_df = find_best_k(embeddings, k_min=2, k_max=10)

    # Show WCSS and Silhouette side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(elbow_plot_path, caption=f"Elbow Method (WCSS). Selected k candidates around elbow. Selected best_k={best_k}", use_container_width=True)
    with col2:
        st.image(silhouette_plot_path, caption="Silhouette scores across k", use_container_width=True)

    st.success(f"Selected number of clusters: {best_k}")

    # Fit final KMeans with best_k
    df["theme_id"] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(embeddings)
    st.success(f"Generated {best_k} clusters")

    # 🔹 Evaluate clustering (final)
    silhouette = silhouette_score(embeddings, df["theme_id"])
    davies = davies_bouldin_score(embeddings, df["theme_id"])
    calinski = calinski_harabasz_score(embeddings, df["theme_id"])

    metrics = {
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies,
        "calinski_harabasz_index": calinski,
        "best_k": int(best_k)
    }

    metrics_summary_df = pd.DataFrame([metrics])
    metrics_excel_path = "clustering_evaluation_metrics.xlsx"
    metrics_summary_df.to_excel(metrics_excel_path, index=False)

    st.dataframe(metrics_summary_df)
    with open(metrics_excel_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="clustering_evaluation_metrics.xlsx">⬇️ Download Clustering Metrics</a>', unsafe_allow_html=True)

    # Also provide the per-k metrics df (wcss + silhouette)
    per_k_excel = "clustering_per_k_metrics.xlsx"
    metrics_df.to_excel(per_k_excel, index=False)
    with open(per_k_excel, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="clustering_per_k_metrics.xlsx">⬇️ Download Per-k Metrics (WCSS + Silhouette)</a>', unsafe_allow_html=True)

    # Step 3.2: t-SNE visualization of final clusters (matplotlib static)
    st.subheader("t-SNE Visualization of Final Clusters")

    n_samples = embeddings.shape[0]

    if n_samples <= 2:
        st.warning("⚠️ t-SNE requires at least 3 articles. Skipping visualization.")
    else:
        try:
            # adapt perplexity: must be < n_samples
            adjusted_perplexity = min(15, n_samples - 1)

            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=adjusted_perplexity,
                init="random"
            )
            reduced = tsne.fit_transform(embeddings)

            fig_tsne, ax_tsne = plt.subplots(figsize=(8, 6))
            scatter = ax_tsne.scatter(
                reduced[:, 0],
                reduced[:, 1],
                c=df["theme_id"],
                cmap="tab10",
                s=35
            )
            ax_tsne.set_title(f"t-SNE projection (k={best_k}, perplexity={adjusted_perplexity})")
            legend1 = ax_tsne.legend(*scatter.legend_elements(), title="Cluster")
            ax_tsne.add_artist(legend1)

            tsne_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig_tsne.savefig(tsne_path.name, bbox_inches="tight")
            plt.close(fig_tsne)

            st.image(tsne_path.name, use_container_width=True)
        except Exception as e:
            st.error(f"t-SNE failed: {e}")

    # Step 3.1: Cluster Labels (unchanged)
    st.subheader("Step 4: Generate Human-Readable Theme Names")
    st.info("Labeling clusters with meaningful names...")
    theme_names = {}
    for tid in sorted(df["theme_id"].unique()):
        joined = "\n\n".join(df[df["theme_id"] == tid]["summary"].tolist())
        prompt = f"""
        You are a financial analyst. Given these summaries, suggest a concise (3–5 word) label capturing their theme.
        Return only the theme name as plain text. Do not include any quotes, symbols, or formatting.

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

    st.success("Theme labels generated.")
    st.write("### 🧠 Interpreted Labels:")
    for tid, label in theme_names.items():
        st.markdown(f"**Theme {tid + 1}:** {label}")

    # Save clustered + labeled data to Excel
    df["theme"] = df["theme_id"].map(theme_names)
    labeled_excel_path = "clustered_articles_with_themes.xlsx"
    df[["article_content", "summary", "theme"]].to_excel(labeled_excel_path, index=False)

    with open(labeled_excel_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="clustered_articles_with_themes.xlsx">⬇️ Download Clustered Articles with Themes</a>', unsafe_allow_html=True)

    # Step 4: Generate Insights & PDF
    def generate_insight(texts, tid):
        prompt = f"""
        # Generate a report section titled '{theme_names.get(tid, f"Theme {tid + 1}")}'.

        Based on these summaries:
        "{texts}"

        Write a structured analysis with the following sections:

        **Executive Summary**
        **Key Trends**
        **Implications and Insights for Investors**
        """
        r = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[{"role": "user", "content": prompt}]
        )
        return r.choices[0].message.content.strip()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp.name, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 50, "Investor-Grade Thematic Report")
    y = height - 80

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

    with open(tmp.name, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">⬇️ Download Report</a>', unsafe_allow_html=True)

    # Step 5: LLM-as-a-Judge Evaluation
    st.subheader("Step 5: Evaluate Report with LLM-as-a-Judge")
    report_content = "\n\n".join(all_content)

    judge_prompt = f"""
    You are an expert financial evaluator.
    Evaluate the following investor report for:
    - Coherence
    - Accuracy of financial insights
    - Value to investors
    - Clarity of writing

    Give scores (1-10) for each criterion and a short justification.

    Report Content:
    {report_content}
    """

    judge_res = client.chat.completions.create(
        model="google/gemma-3n-e2b-it:free",
        #model = "google/gemini-2.0-flash-exp:free" , "tngtech/deepseek-r1t-chimera:free" , "z-ai/glm-4.5-air:free"
        messages=[{"role": "user", "content": judge_prompt}]
    )
    evaluation_text = judge_res.choices[0].message.content.strip()
    st.text_area("📑 LLM-as-a-Judge Evaluation", evaluation_text, height=250)

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
    eval_excel_path = "llm_judge_evaluation.xlsx"
    eval_df.to_excel(eval_excel_path, index=False)

    with open(eval_excel_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="llm_judge_evaluation.xlsx">⬇️ Download Judge Evaluation</a>', unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2025 Zeineb Moalla | Financial Data Science Master Student")
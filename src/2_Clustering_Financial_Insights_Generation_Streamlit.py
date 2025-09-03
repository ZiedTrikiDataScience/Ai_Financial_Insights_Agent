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
# from sklearn.metrics import silhouette_score
# import numpy as np
# import matplotlib.pyplot as plt
# from openai import OpenAI
# from pathlib import Path
# import textwrap
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib.units import inch
# import re

# # Load environment variables from .env

# load_dotenv()
# router_api_key = os.getenv("OPENROUTER_API_KEY")
# guardian_api_key = os.getenv("GUARDIAN_API_KEY")
# if not router_api_key:
#     raise ValueError("Missing OPENROUTER_API_KEY in .env file.")
# if not guardian_api_key:
#     raise ValueError("Missing GUARDIAN_API_KEY in .env file.")

# client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=router_api_key)

# st.set_page_config(page_title="Investor Thematic Report Generator", layout="wide", initial_sidebar_state="expanded")
# st.markdown("""
#     <style>
#         h2 { color: black !important; font-weight: bold !important; }
#     </style>
# """, unsafe_allow_html=True)
# st.title("\U0001F4CA Investor Thematic Report Generator")
# st.markdown("Generate polished, investor-grade PDF reports from financial news summaries.")

# image_path = Path(r"C:\Users\triki\Desktop\MLOps and GenAi\Ai_Financial_Insights_Agent\images\fin_insights.png")
# if image_path.exists():
#     st.image(str(image_path), caption="GenAI Financial Insights", width=500)
# else:
#     st.warning("Banner image not found at:\n" + str(image_path))

# st.sidebar.header("Settings")
# # num_themes = st.sidebar.slider("Number of Themes", 2, 10, 4)
# from_date = st.sidebar.date_input("From Date", pd.to_datetime("2023-01-01"))
# to_date = st.sidebar.date_input("To Date", pd.to_datetime("today"))

# st.sidebar.markdown("---")
# st.sidebar.markdown("**API keys loaded from .env**")
# def find_best_k(embeddings, k_min=2, k_max=5):
#     scores = []
#     k_values = list(range(k_min, k_max + 1))
#     for k in k_values:
#         kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
#         score = silhouette_score(embeddings, kmeans.labels_)
#         scores.append(score)
#     best_k = k_values[np.argmax(scores)]

#     fig, ax = plt.subplots()
#     ax.plot(k_values, scores, marker='o')
#     ax.set_xlabel("Number of Clusters (k)")
#     ax.set_ylabel("Silhouette Score")
#     ax.set_title("Elbow Method - Silhouette Score")
#     elbow_plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
#     fig.savefig(elbow_plot_path.name)
#     plt.close(fig)
#     return best_k, elbow_plot_path.name

# if st.button("\U0001F680 Run Full Pipeline", type='primary', use_container_width=True):
#     st.subheader("Step 1: Scrape Financial News")
#     st.info("Fetching articles from The Guardian API...")
#     endpoint = "https://content.guardianapis.com/search"
#     params = {
#         "q": "financial news",
#         "section": "business",
#         "from-date": from_date.isoformat(),
#         "to-date": to_date.isoformat(),
#         "show-fields": "bodyText,webPublicationDate",
#         "page-size": 10,
#         "api-key": guardian_api_key
#     }
#     resp = requests.get(endpoint, params=params)
#     resp.raise_for_status()
#     articles = resp.json()["response"]["results"]
#     rows = [
#         {
#         "publicationDate": pd.to_datetime(a["webPublicationDate"]).tz_localize(None),
#         "article_content": a["fields"]["bodyText"]
#         } 
#         for a in articles]
#     df = pd.DataFrame(rows).sort_values("publicationDate", ascending=False)
#     st.success(f"Scraped {len(df)} articles")
#     st.dataframe(df[["publicationDate"]].head(5))

#     st.subheader("Step 2: Summarize Articles")
#     st.info("Summarizing content with BART...")
#     summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")
#     df["summary"] = df["article_content"].apply(lambda t: summarizer(t[:4000], max_length=150, min_length=40, do_sample=False)[0]["summary_text"])
#     st.success("Articles summarized")
#     st.dataframe(df[["summary"]].head(3))

#     st.subheader("Step 3: Embed & Determine Optimal Number of Clusters")
#     st.info("Creating embeddings and clustering...")
#     embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#     embeddings = embedder.encode(df["summary"].tolist(), convert_to_numpy=True)
#     best_k, elbow_plot = find_best_k(embeddings)
#     st.image(elbow_plot, caption=f"Optimal number of clusters: {best_k}", use_container_width=True)
    
#     st.write("### \U0001F913 The best number of clusters is : ", best_k)


#     st.subheader("Step 3.1: Clustering Summaries")
#     df["theme_id"] = KMeans(n_clusters=best_k, random_state=42).fit_predict(embeddings)
#     st.success(f"Generated {best_k} clusters")

#     # Save the DataFrame with summaries and themes to Excel
#     excel_path = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx").name
#     df.to_excel(excel_path, index=False)

#     # Provide a download link in Streamlit
#     with open(excel_path, "rb") as f:
#         b64 = base64.b64encode(f.read()).decode()
#         st.markdown(
#             f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="summaries_clustered.xlsx">⬇️ Download Excel File</a>',
#             unsafe_allow_html=True
#         )

#     st.subheader("Step 3.1: Generate Human-Readable Theme Names")
#     st.info("Labeling clusters with meaningful names...")
#     theme_names = {}
#     for tid in sorted(df["theme_id"].unique()):
#         joined = "\n\n".join(df[df["theme_id"] == tid]["summary"].tolist())
#         prompt = f"""
#         You are a financial analyst. Given these summaries, suggest a concise (3–5 word) label capturing their theme.
#         Return only the theme name as plain text. Do not include any quotes, symbols, or formatting like LaTeX or Markdown.

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

#     st.success("Theme labels generated.")
#     st.write("### \U0001F9E0 Interpreted Labels:")
#     for tid, label in theme_names.items():
#         st.markdown(f"**Theme {tid + 1}:** {label}")

#     def generate_insight(texts, tid):
#         prompt = f"""
# Generate a report section titled '{theme_names.get(tid, f"Theme {tid + 1}")}'.

# Based on these summaries:
# "{texts}"

# Write a structured analysis with the following sections:

# **Executive Summary** : where you summarize the theme from its articles in 3/4 lines

# **Key Trends** 

# **Implications and Insights for Investors** : where you give valuable insights and advices for investors based on the happenings so that they can take
# better financial decisions.

# Each section should be clearly labeled.
# """
#         r = client.chat.completions.create(
#             model="deepseek/deepseek-chat-v3.1:free",
#             messages=[{"role": "user", "content": prompt}],
#             extra_headers={"HTTP-Referer": "https://openrouter.ai", "X-Title": "Insight Gen"}
#         )
#         return r.choices[0].message.content.strip()

#     st.subheader("Step 4: Generate PDF Report")
#     st.info("Formatting report sections into PDF using ReportLab...")

#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
#     c = canvas.Canvas(tmp.name, pagesize=letter)
#     width, height = letter
#     c.setFont("Helvetica-Bold", 16)
#     c.drawCentredString(width / 2, height - 50, "Investor-Grade Thematic Report")
#     y = height - 80

#     for tid in sorted(df["theme_id"].unique()):
#         summaries = df[df["theme_id"] == tid]["summary"].tolist()
#         content = generate_insight("\n\n".join(summaries), tid)
#         theme_title = theme_names.get(tid, f"Theme {tid + 1}")

#         c.setFont("Helvetica-Bold", 14)
#         c.drawString(40, y, theme_title)
#         y -= 20
#         c.setFont("Helvetica", 11)
#         for line in content.split("\n"):
#             for subline in textwrap.wrap(line, 95):
#                 if y < 50:
#                     c.showPage()
#                     c.setFont("Helvetica", 11)
#                     y = height - 50
#                 c.drawString(40, y, subline)
#                 y -= 14
#             y -= 8
#         y -= 16

#     c.setFont("Helvetica-Oblique", 10)
#     c.drawCentredString(width / 2, 30, "Created by: Zeineb Moalla, Financial Data Science Master Student")
#     c.save()

#     with open(tmp.name, "rb") as f:
#         b64 = base64.b64encode(f.read()).decode()
#         st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">⬇️ Download Report</a>', unsafe_allow_html=True)

# st.markdown("---")
# st.caption("© 2025 Zeineb Moalla | Financial Data Science Master Student")










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
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from pathlib import Path
import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re

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

st.set_page_config(page_title="Investor Thematic Report Generator", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Investor Thematic Report Generator")
st.markdown("Generate polished, investor-grade PDF reports from financial news summaries.")

# Image Banner
image_path = Path(r"C:\Users\triki\Desktop\MLOps and GenAi\Ai_Financial_Insights_Agent\images\fin_insights.png")
if image_path.exists():
    st.image(str(image_path), caption="GenAI Financial Insights", width=500)

# Sidebar Settings
st.sidebar.header("Settings")
from_date = st.sidebar.date_input("From Date", pd.to_datetime("2023-01-01"))
to_date = st.sidebar.date_input("To Date", pd.to_datetime("today"))

# Helper: Find Best K
def find_best_k(embeddings, k_min=2, k_max=5):
    scores = []
    k_values = list(range(k_min, k_max + 1))
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
        score = silhouette_score(embeddings, kmeans.labels_)
        scores.append(score)
    best_k = k_values[np.argmax(scores)]

    fig, ax = plt.subplots()
    ax.plot(k_values, scores, marker='o')
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Elbow Method - Silhouette Score")
    elbow_plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(elbow_plot_path.name)
    plt.close(fig)
    return best_k, elbow_plot_path.name

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
        "page-size": 10,
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

    # Step 2: Summarization
    st.subheader("Step 2: Summarize Articles")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")
    df["summary"] = df["article_content"].apply(
        lambda t: summarizer(t[:4000], max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
    )
    st.success("Articles summarized")

    # Step 3: Clustering
    st.subheader("Step 3: Embeddings & Clustering")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(df["summary"].tolist(), convert_to_numpy=True)
    best_k, elbow_plot = find_best_k(embeddings)
    st.image(elbow_plot, caption=f"Optimal clusters: {best_k}", use_container_width=True)

    df["theme_id"] = KMeans(n_clusters=best_k, random_state=42).fit_predict(embeddings)
    st.success(f"Generated {best_k} clusters")

    # 🔹 Evaluate clustering with 3 metrics
    silhouette = silhouette_score(embeddings, df["theme_id"])
    davies = davies_bouldin_score(embeddings, df["theme_id"])
    calinski = calinski_harabasz_score(embeddings, df["theme_id"])

    metrics = {
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies,
        "calinski_harabasz_index": calinski,
        "best_k": int(best_k)
    }

    # Save metrics to Excel
    metrics_df = pd.DataFrame([metrics])
    metrics_excel_path = "clustering_evaluation_metrics.xlsx"
    metrics_df.to_excel(metrics_excel_path, index=False)

    st.dataframe(metrics_df)
    with open(metrics_excel_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="clustering_evaluation_metrics.xlsx">⬇️ Download Clustering Metrics</a>', unsafe_allow_html=True)

    # Step 3.1: Cluster Labels
    st.subheader("Step 3.1: Generate Human-Readable Theme Names")
    st.info("Labeling clusters with meaningful names...")
    theme_names = {}
    for tid in sorted(df["theme_id"].unique()):
        joined = "\n\n".join(df[df["theme_id"] == tid]["summary"].tolist())
        prompt = f"""
        You are a financial analyst. Given these summaries, suggest a concise (3–5 word) label capturing their theme.
        Return only the theme name as plain text. Do not include any quotes, symbols, or formatting like LaTeX or Markdown.

        Summaries:
        {joined}
        """
        res = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            #qwen/qwen3-coder:free
            messages=[{"role": "user", "content": prompt}],
            extra_headers={"HTTP-Referer": "https://openrouter.ai", "X-Title": "Theme Label"}
        )
        raw_output = res.choices[0].message.content.strip()
        clean_output = re.sub(r'\\boxed\{["\']?(.*?)["\']?\}', r'\1', raw_output)
        theme_names[tid] = clean_output.strip()

    st.success("Theme labels generated.")
    st.write("### \U0001F9E0 Interpreted Labels:")
    for tid, label in theme_names.items():
        st.markdown(f"**Theme {tid + 1}:** {label}")

    # 🔹 Save clustered + labeled data to Excel
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

        **Executive Summary** : where you summarize the theme from its articles in 3/4 lines

        **Key Trends** 

        **Implications and Insights for Investors** : where you give valuable insights and advices for investors based on the happenings so that they can take
        better financial decisions.

        Each section should be clearly labeled.
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

    all_content = []  # for judge evaluation
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

    # 🔹 Step 5: LLM-as-a-Judge Evaluation
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
        model="google/gemini-2.5-flash-image-preview:free",
        messages=[{"role": "user", "content": judge_prompt}]
    )
    evaluation_text = judge_res.choices[0].message.content.strip()
    st.text_area("📑 LLM-as-a-Judge Evaluation", evaluation_text, height=250)

    # Try to parse evaluation into structured format (fallback to raw if parsing fails)
    eval_data = []
    for line in evaluation_text.split("\n"):
        match = re.match(r"^(.*?):\s*([0-9]+)/?10?.*", line.strip(), re.IGNORECASE)
        if match:
            criterion = match.group(1).strip()
            score = match.group(2).strip()
            eval_data.append({"criterion": criterion, "score": score})
    if not eval_data:  # fallback to raw
        eval_data.append({"criterion": "Full Evaluation", "score": evaluation_text})

    eval_df = pd.DataFrame(eval_data)
    eval_excel_path = "llm_judge_evaluation.xlsx"
    eval_df.to_excel(eval_excel_path, index=False)

    with open(eval_excel_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="llm_judge_evaluation.xlsx">⬇️ Download Judge Evaluation</a>', unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2025 Zeineb Moalla | Financial Data Science Master Student")

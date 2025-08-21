"""
*** Script Content : 
    Web Scraping of Financial News from the Guardian API 
    and 
    Scraped Text Summarization with LLM (Facebook BART Large CNN) 

*** Input:
    The Guardian API Scraping Parameters

*** Output : 
    DataFrame containing: 
    - publicationDate
    - article_content
    - summarized_article_content
        
*** Date : 
    Version1 : 02/06/2024

*** Developed By:
    Zied Triki, AI Engineer
"""

# Importing the necessary libraries
import os
import requests
import pandas as pd
from transformers import pipeline
from loguru import logger
from dotenv import load_dotenv
from Text_Processing_Financial_News import process_data

# Load environment variables from .env
load_dotenv()
guardian_api_key = os.getenv("GUARDIAN_API_KEY")

if not guardian_api_key:
    logger.error("Missing GUARDIAN_API_KEY in environment. Please set it in your .env file.")
    raise SystemExit("Missing GUARDIAN_API_KEY")

# Initialize the BART summarization pipeline
logger.info("Initializing the BART summarization pipeline...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")
logger.info("Pipeline initialized successfully.")

# Guardian API endpoint and parameters
endpoint = "https://content.guardianapis.com/search"
params = {
    "q": "financial news",
    "api-key": guardian_api_key,
    "section": "business",
    "from-date": "2022-11-01",
    "to-date":   "2024-12-30",
    "show-fields": "bodyText,webPublicationDate",
    "page-size": 2,
    "page": 1
}

# Fetch articles
logger.info("Making a request to the Guardian API...")
response = requests.get(endpoint, params=params)

if response.status_code != 200:
    logger.error(f"Error fetching data: HTTP {response.status_code}")
    raise SystemExit(f"Guardian API request failed ({response.status_code})")

logger.info("Request successful. Parsing JSON response...")
data = response.json()
articles = data["response"]["results"]

# Prepare storage lists
publication_dates = []
article_contents    = []
summarized_articles = []

logger.info(f"Processing {len(articles)} articles...")
for article in articles:
    pub_date = article["webPublicationDate"]
    content  = article["fields"]["bodyText"]

    # Pre-process
    content = process_data(column_name=content)

    # Summarize
    logger.info(f"Summarizing article published on {pub_date}...")
    snippet = content[:4000]
    summary = summarizer(snippet, max_length=150, min_length=40, truncation=True, do_sample=False)
    summary_text = summary[0]["summary_text"]

    publication_dates.append(pub_date)
    article_contents.append(content)
    summarized_articles.append(summary_text)

# Build DataFrame
logger.info("Creating the final DataFrame...")
df = pd.DataFrame({
    "publicationDate":          publication_dates,
    "article_content":          article_contents,
    "summarized_article_content": summarized_articles
})

# Clean up dates and sort
df["publicationDate"] = pd.to_datetime(df["publicationDate"]).dt.tz_localize(None)
df = df.sort_values("publicationDate", ascending=False)
df = df[["publicationDate", "article_content", "summarized_article_content"]]

# Save outputs
logger.info("Saving DataFrame to Excel and Parquet formats...")
df.to_excel("summarized_articles_financial_news.xlsx", index=False)
df.to_parquet("summarized_articles_financial_news.parquet", index=False)
logger.info("Summarized articles saved successfully.")

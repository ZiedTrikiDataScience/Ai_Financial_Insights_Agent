"""
*** Script Content : 
        Processing of the text input columns before Feeding to the LLM Model  

*** Input:
        Raw Unprocessed scraped data


*** Output : 
        2 Processed Dataframes :
            1. processed_summarised_financial_news_processed : processed input dataframe
            2. processed_summarised_articles_financial_news_test : processed test dataframe
        
*** Date : 
        Version1 : 02/06/2024

                      
*** Developed By:
        Zied Triki , MLOps Enigneer
        
"""


# Importing the necessary libraries :
import pandas as pd
import re
#import spacy
import contractions
from bs4 import BeautifulSoup
from tqdm import tqdm
from loguru import logger


tqdm.pandas()

# Loading the Spacy model :
#nlp = spacy.load('en_core_web_lg', exclude=['ner', 'parser', 'senter'])

# Defining all the necessary text processing functions :
def _remove_html_tags(text):
    """Remove html tags."""
    logger.info("Removing html tags")
    return BeautifulSoup(text, 'html.parser').get_text()

def _remove_links(text):
    """Remove web links."""
    logger.info("Removing links")
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def _remove_emails(text):
    """Remove emails."""
    logger.info("Removing emails")
    return re.sub(r'\S+@\S+', '', text)



def process_data(column_name):
    """
    Function that processes the text before feeding it to the LLM Model

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        column_name (str): Column name of the text data to be cleaned.

    Returns:
        df (pd.DataFrame): DataFrame with cleaned text in 'processed_summarized_article_content'
    """
    logger.info("Converting text to string dtype")
    #processed_column  = column_name.astype(str)
    processed_column  = column_name

    
    logger.info("Removing html tags")
    processed_column  = _remove_html_tags(processed_column)
    
    logger.info("Removing links")
    processed_column  = _remove_links(processed_column)
    
    logger.info("Removing emails")
    processed_column  = _remove_emails(processed_column)
    
    return processed_column

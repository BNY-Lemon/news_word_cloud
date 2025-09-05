# News Word Cloud Chatbot

## Overview
This project is a lightweight tool to explore recent news articles using word clouds and a question-answering chatbot. It combines:

- **News retrieval**: Fetches and caches recent news articles.
- **Word cloud generation**: Visualizes frequently occurring words in headlines and content.
- **RAG-powered chatbot**: Uses a local language model with FAISS embeddings to answer questions based on the retrieved news content, providing article titles and concise summaries.

The chatbot is designed to help users understand why certain words appear frequently in the word cloud and provide context for trending news topics.

## Structure
- `word_cloud_test.ipynb`: Jupyter notebook for generating word clouds and interacting with the chatbot.
- `create_vector.py`: Processes cached articles, chunks them, and creates a FAISS vector database for retrieval.
- `chatbot.py`: Handles querying the FAISS database and interacting with a local language model.

## Features
- Filters out anti-scraping/security content from articles.
- Chunking ensures the language model receives manageable context.
- RAG (Retrieval-Augmented Generation) ensures answers are grounded in actual articles.
- Interactive Jupyter interface using `ipywidgets`.

## Usage
1. Run `word_cloud_test.ipynb` to generate word clouds from cached news articles.
2. Use the interactive input box at the bottom of the notebook to ask questions about words from the word cloud.
3. The chatbot will return relevant article titles and concise summaries.

## Requirements
- Python 3.10+
- `langchain`, `langchain-ollama`, `ipywidgets`, `pandas`, `matplotlib`, `wordcloud`, `faiss-cpu` (or `faiss-gpu`)
- A local Ollama-compatible language model (e.g., TinyLlama) running on a specified port.

## Notes
- Only free-tier NewsAPI data from the last 30 days is used.
- Ensure API keys are kept private and `.env` or cache files are excluded from GitHub via `.gitignore`.


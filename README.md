# SET UP
1. Install all required packages from `requirements.txt`
2. Sign up at [LangSmith](https://smith.langchain.com/) to enable logging
3. Create `.env` file and add the following variables:
    - `LANGCHAIN_RAG_OPENAI_API_KEY= "..."` _(Insert your OpenAI API Key for this project)_
    - `LANGCHAIN_TRACING_V2=true`
    - `LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"`
    - `LANGCHAIN_API_KEY="..."` _(Insert your LangSmith API Key)_
    - `LANGCHAIN_PROJECT="..."`_(Insert your chosen LangSmith project name)_
4. Open a Terminal in the projectr directory and enter `streamlit run app.py`

# OVERVIEW
This repo establishes a RAG system using Langchain and then surfaces a chat functionality for the user interact with the content
The content used is the recent (at the time of creating) 2024 Week 6 NFL results using online match reports/summaries

# RAG
RAG Application generally has two main components:
- Indexing: A pipeline for ingesting data from a source and indexing (usually done offline)
- Retrieval and Generation: The actual RAG chain, which takes a user's query and retrieves relevant data from the index which is also passed to the model

Both stages are included here for simplicity of sharing
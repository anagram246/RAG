## packages to parse content and load docs
import bs4
import re
import requests
from typing import List
from langchain_community.document_loaders import WebBaseLoader

## packages for splitting content into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

## packages for embedding and storing data
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def load_articles(pff_url: str) -> List[langchain_core.documents.base.Document]:
    """
    Description: Function that takes the pff nfl week 6 overview article and parses it
    to store the text and find the links to each individual game summary

    Some of the articles included are behind a paywall which stops the function finding their content
    Have left these as-is to see how the chat handles questions about these games

    Currently very specific for this article - might generalise to other weeks (not tested)

    Parameters: input url for overview article

    Returns: the loaded docs

    """
    
    # Parse the HTML content
    response = requests.get(pff_url)
    soup = bs4.BeautifulSoup(response.text, 'html.parser')

    # Find and store url to each individual game summary too
    game_recaps = []

    for link in soup.find_all(href=re.compile('nfl-week-6')):
        game_recaps.append(link.get('href'))

    urls = [pff_url] + list(set(game_recaps))

    # Filter for main article body only
    bs4_strainer = bs4.SoupStrainer(class_=("m-longform-copy"))
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs={"parse_only": bs4_strainer}
    )

    # Return docs
    docs = loader.load()
    return docs


def split_articles(docs: List[langchain_core.documents.base.Document]) -> List[langchain_core.documents.base.Document]:

    """
    Description: Splits the loaded articles into overlapping chunks

    Parameters: Takes a list of docs as the input

    Returns: A list of the chunked docs    
    
    """

    # initiate text splitter, apply to all docs and return splits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True
    )
    
    all_splits = text_splitter.split_documents(docs)

    return all_splits


def create_vector_store(all_splits: List[langchain_core.documents.base.Document], vector_db_collection_name: str, vector_db_dir: str):
    """
    Description: Function that takes the chunked up articles, embeds them using OpenAI text embedding model and
    store them in a vector database

    Parameters: Input of the split documents, the vector database collection name and the vector database directory

    Returns: Nothing
    
    """

    # Initiate embeddings using API key
    embeddings=OpenAIEmbeddings(model='text-embedding-3-small', api_key=os.getenv("LANGCHAIN_RAG_OPENAI_API_KEY"))

    # Create and save the vector db
    vectorstore = Chroma(
        collection_name=vector_db_collection_name,
        embedding_function=embeddings,
        persist_directory=vector_db_dir
    )

    # Add the docs to the vector db
    vectorstore.add_documents(all_splits)



### IMPORTS ###
# Utilities
import os
from dotenv import load_dotenv

# Langchain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

# App
import streamlit as st

# load environment variables
load_dotenv()

# check if vector database already exists. if no, run "load" functions
vector_db_coll_nm = "nfl_articles"
vector_db_dir = "./nfl_vector_db"
vector_db_flag = os.path.isdir(vector_db_dir)

if vector_db_flag == False:
    
    import Indexing.load as load
    url = "https://www.pff.com/news/nfl-scores-and-recaps-for-every-week-6-game"
    docs = load.load_articles(url)
    all_splits = load.split_articles(docs)
    load.create_vector_store(all_splits, vector_db_coll_nm, vector_db_dir)

# initialise LLM and embedding model
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("LANGCHAIN_RAG_OPENAI_API_KEY"))
embeddings=OpenAIEmbeddings(model='text-embedding-3-small', api_key=os.getenv("LANGCHAIN_RAG_OPENAI_API_KEY"))

# load vector db and set up as retriever
vector_db = Chroma(
    collection_name=vector_db_coll_nm,
    embedding_function=embeddings,
    persist_directory=vector_db_dir  # Directory 
)

retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# set up prompt template and chain
prompt_template = ChatPromptTemplate([
    ("system", """You are an assistant for question-answering tasks.
     Use the chat history along with with the following pieces of retrieved context to answer the question.
     If you don't know the answer, just say that you don't know.
     Use three sentences maximum and keep the answer concise.
     Chat History: {history}
     Context: {context} 
     Question: {question}
     Answer:""")
])

# small functions to help pass context and history to the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(chat_history):
    return "\n\n".join(message['role'] + ": " + message['content'] for message in chat_history)

def retrieve_context(inputs):
    question = inputs['question']
    # Retrieve relevant documents based on the question
    return retriever.invoke(question)

# create streamlit app
st.title("NFL Week 6 ChatBot")

# check if session already has messages, otherwise start a new message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi - I know about the Week 6 NFL games. How can I help?"}
    ]

# Define chain
# parallel runnables allow us to path the user question to the retriever and the chat history to the prompt separately    
rag_chain = (
    RunnableParallel( 
        context=RunnableLambda(retrieve_context) | format_docs,
        question=RunnablePassthrough(),
        history=RunnablePassthrough())
    | prompt_template
    | llm
    | StrOutputParser()
)

# write chat history to front end
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# receive user input, add to chat history and write to front end
if user_prompt := st.chat_input("Ask me about the Week 6 games"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # format chat history to pass to prompt
    chat_history = format_history(st.session_state.messages)
    
    # stream response from model
    with st.chat_message("assistant"):
        response = st.write_stream(rag_chain.stream({
            "question": user_prompt,
            "history": chat_history
        }))
    
    # add model response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})




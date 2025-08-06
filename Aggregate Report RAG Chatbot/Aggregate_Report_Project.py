import streamlit as st
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import Tool, initialize_agent, AgentType
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="Medical Review Q&A", layout="wide")
st.title("🩺 Medical Review Question Answering App")

# Load and preprocess CSV
df = pd.read_csv(r"c:\\Users\\Jeevan\\Downloads\\MedicalReviews.csv")
df = df[['Drug Name', 'Condition', 'Rating', 'Content']]
df = df[df["Rating"] != "No rating available"]
df = df.dropna()
df = df.sample(500)

# Prepare data for embedding
text = df['Content'].astype('str').tolist()
metadata = df[['Drug Name', 'Condition', 'Rating']].to_dict(orient="records")

# Load embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_texts(
    text,
    embedding=embeddings,
    metadatas=metadata,
    persist_directory="reviews_db"
)
retriever = vectorstore.as_retriever()

# LLM setup
llm = ChatGroq(model="Gemma2-9b-It", temperature=0)

# Prompt and chain setup
prompt_template = """
You are analyzing patient reviews of medications. Answer questions using ONLY the provided reviews.

Guidelines:
- Always mention drug names and conditions
- Include rating numbers (1-10) when available
- Be specific - say "3 out of 5 reviews mentioned..." when possible
- If unsure, say "The reviews don't mention this"

Reviews:
{context}

Question: {input}

Answer clearly and specifically:
"""
prompt = ChatPromptTemplate([prompt_template])
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

# Create Pandas Agent
pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True)

# Define tools
review_tool = Tool(
    name="review_tool",
    func=lambda x: rag_chain.invoke({"input": x}),
    description="Useful for answering medical/natural language questions based on review content, drug effects, descriptions, etc."
)
pandas_tool = Tool(
    name="pandas_tool",
    func=pandas_agent.invoke,
    description="Useful for answering questions about the structured dataset like record counts, averages, filtering, ratings etc."
)

# Initialize agent
agent = initialize_agent(
    tools=[review_tool, pandas_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Input box for user query
query = st.text_input("Ask a question about the medical reviews:", "What are the side effects of Mirena?")

# When the user submits a query
if st.button("Get Answer"):
    with st.spinner("Analyzing reviews..."):
        try:
            response = agent.invoke(query)
            st.success("Answer:")
            st.write(response['output'])
        except Exception as e:
            st.error(f"Error: {str(e)}")

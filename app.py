import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, provider, model, temperature, max_tokens):
    """
    Generate response using either Groq or Hugging Face based on provider selection.
    """
    if provider == "Groq":
        llm = ChatGroq(
            model_name=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:  # Hugging Face
        llm = HuggingFaceEndpoint(
            endpoint_url=f"https://api-inference.huggingface.co/models/{model}",
            huggingfacehub_api_token=api_key,
            temperature=temperature,
            max_new_tokens=max_tokens
        )

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Title of the app
st.title("Enhanced Q&A Chatbot With Free Models")

# Sidebar for settings
st.sidebar.title("Settings")

# Select provider (Groq or Hugging Face)
provider = st.sidebar.selectbox("Select Provider", ["Groq", "Hugging Face"])

# Input API key based on provider
if provider == "Groq":
    api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
else:
    api_key = st.sidebar.text_input("Enter your Hugging Face API Key:", type="password")

# Select model based on provider
if provider == "Groq":
    model = st.sidebar.selectbox("Select Groq Model", ["llama3-8b-8192", "mistral-saba-24b", "gemma2-9b-it"])
else:
    model = st.sidebar.selectbox("Select Hugging Face Model", [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "google/flan-t5-large"
    ])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=800, value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

# Handle user input and API key validation
if user_input and api_key:
    try:
        response = generate_response(user_input, api_key, provider, model, temperature, max_tokens)
        st.write(response)
    except Exception as e:
        st.error(f"Error: {str(e)}")
elif user_input:
    st.warning(f"Please enter the {provider} API Key in the sidebar")
else:
    st.write("Please provide the user input")
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import WikipediaLoader
from urllib.parse import urlparse, unquote


# Streamlit app configuration
st.set_page_config(page_title="Summarize & Translate Text From Any Website", page_icon="🦜")
st.title("🦜 Summarize & Translate Text From Any Website")
st.subheader("URL:")

url= st.text_input("URL", label_visibility="collapsed")  

# Sidebar for Groq API key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    if not groq_api_key.strip():
        st.error("Please provide the API key")

llm = ChatGroq(
    model="gemma2-9b-it",
    groq_api_key=groq_api_key,
    temperature=0.3,
    max_tokens=500  # Cap for ~300-word summary
)

language= st.selectbox("Select the language", ["English", "Hindi", "Marathi", "Kannada", "Tamil", "Telugu"])

prompt_template= """Please create a concise and well-structured summary of the following text in {language}, adhering to these guidelines:

1. Length: less than 400 words.  
2. Style: casual + academic
3. Focus: Emphasize key arguments, main findings, critical dates
4. Use bullet points, headings, subheadings, emojis when necessary
5. Include significant quotes (shortened if needed).

Original Text:
{text}

Guidelines:
- Maintain factual accuracy
- Preserve important names, dates, and statistics
- Use clear, concise language
- Avoid personal opinions or interpretations
- Write only in {language}, with no English or repetitive text.
- Use clear, concise language.

Summary:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text", "language"])


if st.button("Summarize"):
    if not url.strip():
        st.error("Please provide the URL")
    elif not validators.url(url):
        st.error("Please enter a valid URL")
    else:
        try:
            with st.spinner("Please Wait..."):
                parsed_url = urlparse(url)
                if "wikipedia.org" in parsed_url.netloc:
                    # ✅ Use WikipediaLoader
                    title = unquote(parsed_url.path.split("/")[-1])
                    loader = WikipediaLoader(query=title, lang="en", load_max_docs=1)
                    docs = loader.load()
                else:
                    loader= UnstructuredURLLoader([url], ssl_verify= False, headers={
                                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                            })
                    docs=loader.load()

                # Summarize chain
                chain= load_summarize_chain(llm=llm, prompt= prompt)
                summary = chain.invoke({"input_documents": docs, "language": language})
                st.success(summary['output_text'])
        except Exception as e:
            st.exception(f"Error: {e}") 
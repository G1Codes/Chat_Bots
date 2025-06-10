import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3 # Keep this import, even if not used directly, as `create_sql_agent` might implicitly rely on it for some internal checks or broader compatibility, or remove if confident it's not needed.
import sqlalchemy # Import sqlalchemy for sqlalchemy.text
from langchain_groq import ChatGroq

st.set_page_config(page_title="LangChain: Chat with MySQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with MySQL DB")

# --- Configuration (MySQL Only) ---
# We no longer need LOCALDB or radio_opt as MySQL is the only option
MYSQL_ONLY = "USE_MYSQL" # Define a constant for clarity

st.sidebar.title("MySQL Database Settings") # Updated sidebar title

# Always ask for MySQL details
# Added default values for host and user
mysql_host = st.sidebar.text_input("MySQL Host:", value="localhost:3306")
mysql_user = st.sidebar.text_input("MySQL User:", value="root")
mysql_password = st.sidebar.text_input("MySQL password:", type="password")
mysql_db = st.sidebar.text_input("MySQL database name:")

api_key = st.sidebar.text_input(label="Groq API Key:", type="password")

# --- Input Validation ---
# Check if all necessary credentials are provided
if not (mysql_host and mysql_user and mysql_password and mysql_db):
    st.info("Please provide all MySQL connection details in the sidebar to proceed.")
    st.stop() # Stop execution if details are missing

if not api_key:
    st.info("Please add your Groq API key in the sidebar.")
    st.stop() # Stop execution if API key is missing

## LLM model (initialized only if API key is present)
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

@st.cache_resource(ttl="2h")
def configure_db_for_mysql(mysql_host, mysql_user, mysql_password, mysql_db):
    """
    Configures and returns an SQLDatabase object specifically for MySQL.
    This function is cached for 2 hours to avoid repeated setup.
    """
    try:
        # Construct the MySQL connection string
        db_connection_str = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
        print(f"Attempting to connect to MySQL DB: {mysql_db}@{mysql_host}")

        # Create the SQLAlchemy engine
        engine = create_engine(db_connection_str)

        # Test the connection to catch errors early
        with engine.connect() as connection:
            connection.execute(sqlalchemy.text("SELECT 1")) # Use sqlalchemy.text for explicit SQL
        print("Successfully tested MySQL connection.")

        # Return the LangChain SQLDatabase wrapper
        return SQLDatabase(engine)
    except Exception as e:
        st.error(f"Failed to connect to MySQL database: {e}")
        st.stop() # Stop the app if connection fails
        return None # Should not be reached due to st.stop()

# Configure the database
# We directly call the MySQL-specific configuration function
db = configure_db_for_mysql(mysql_host, mysql_user, mysql_password, mysql_db)

# Ensure db is not None before proceeding, though st.stop() should handle it
if db is None:
    st.stop()

## Toolkit for the SQL Agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Define a custom system message to guide the agent, preventing overthinking
# This prompt tells the agent what it is, what its goal is, and how to use its tools.
# It explicitly tells it to prioritize querying the database.
SYSTEM_MESSAGE = """
You are an AI assistant that can answer questions by interacting with a SQL database.
You have access to the following tools:
{tools}

You should only use the provided tools to answer the user's questions.
If the user asks a question about data, you must query the database to get the answer.
If the user asks about a specific table or schema, use the appropriate schema tool.
When performing a query, always return the full result of the query.
Be concise and to the point in your final answer.
If you need to query the database, always try to use sql_db_query first if you know the table and column names.
Only use sql_db_schema if you need to understand the table structure, and only if absolutely necessary.
Do not make up information. If you cannot find the answer in the database, state that.
Always format your final answer in natural language.
"""

# Create the SQL Agent
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True, # Set to True to see the agent's thought process in the Streamlit UI
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # agent_type_kwargs={
    #     "system_message": SYSTEM_MESSAGE
    # },
    max_iterations=15, # Increased iterations slightly, but the prompt should reduce unnecessary ones
    handle_parsing_errors=True # Crucial for robustness
)

# --- Chat History Management ---
# Initialize or clear message history
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I can chat with your MySQL database. What would you like to know?"}]

# Display existing messages in the chat UI
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- User Input and Agent Response ---
user_query = st.chat_input(placeholder="Ask anything from your MySQL database (e.g., 'What are the names of all students?', 'How many students have marks greater than 90?').")

if user_query:
    # Append user query to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        # Set up Streamlit callback handler to display agent's thoughts
        streamlit_callback = StreamlitCallbackHandler(st.container())
        try:
            with st.spinner("Thinking..."): # Show a spinner while the agent is processing
                # Run the agent with the user's query
                response = agent.run(user_query, callbacks=[streamlit_callback])
            # Append agent's response to chat history and display it
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
        except Exception as e:
            # Handle potential errors during agent execution
            error_message = f"An error occurred: {e}. The agent might be struggling or hit a limit. Please try rephrasing your question or clear the history."
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})


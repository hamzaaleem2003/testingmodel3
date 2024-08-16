__import__('pysqlite3')  # Dynamically imports the pysqlite3 module
import sys  # Imports the sys module necessary to modify system properties
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')  # Replaces the sqlite3 entry in sys.modules with pysqlite3
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import time
import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
class ChatBot():
    def __init__(self):
        # Initialize the API key for Google Generative AI
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        # Load the embeddings model
        embeddings = HuggingFaceEmbeddings()
        
        # Define the collection name and persistent directory for Chroma
        collection_name = "urdu_book_collection"
        persist_directory = "urdubookcollection"  # Specify the persistent directory

        # Initialize Chroma by loading the existing collection
        self.knowledge = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

        # Initialize the Google Generative AI model
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
        
        # Define the template for prompting
        self.template = """
        this is the data from the book "Jami-Khalq-par-Huzoor-ki-Rahmat-o-Shafqat" written by huzoor sheikh ul islam dr muhammad tahir ul qadri, whatever question is asked you have to answer that properly and comprehensively. Whatever question is asked you have to answer properly and comprehensively also, whenever a question is asked from this book you always have to answer the question in Urdu language no matter if in prompt it mentions to answer in Urdu or not, but if it specifies to answer in some other language, only then you have to change the language in giving a response.

        Context: {context}

        Question: {question}

        Answer:
        """
        
        # Initialize the prompt template
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )
        
        # Define the RAG chain for retrieval and generation
        self.rag_chain = (
            {"context": self.knowledge.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

# Create an instance of the ChatBot class
bot = ChatBot()
# input = input("Ask me anything: ")
# result = bot.rag_chain.invoke(input)
# print(result)

st.set_page_config(page_title="Urdu Book Bot")
with st.sidebar:
    st.title('Urdu Book Bot')

# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain.invoke(input)
    return result

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, ask me anything from this book"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
            
        # with st.spinner("Getting your answer from mystery stuff.."):
            response = generate_response(input)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)


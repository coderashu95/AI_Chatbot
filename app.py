import streamlit as st  # Used to create interactive web applications using Python
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter #SplittingTextIntoSmallerChunks
from langchain_community.vectorstores import FAISS #VectorDatabase
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings #UsedToConverChunksIntoPositionalEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core import retry
import google.generativeai as genai
import time
import logging

logging.basicConfig(level=logging.ERROR)

st.set_page_config(page_title="Personal Document Chatbot Tool", layout="wide")

st.markdown("""
## Document Chatbot: Get instant insights from your document
## How it starts:

Follow the instructions as below:

1. **Enter API Key** - You will need Google API key for Chatbot to access Google's generative AI models. Obtain your API key here : https://developers.google.com/maps/documentation/javascript/get-api-key#create-api-keys \n
2. **Upload your documents** - System accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.
3. **Ask a question** - After getting "Processing Done!" message, ask any question related to the content of your uploaded documents.
""")

api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

# This is a function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""

    # Code has been modified to take care of error handling
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
            continue
    if text:
        st.info("Text extraction complete.") 
    else:
        st.warning("No text extracted. Please check your PDF files.")
    return text

# This is a function to split pdf text to text chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# This is a function to convert text chunks into positional embeddings, and store those embeddings into vector database
def get_vector_store(text_chunks, api_key):
    
    # Modified code to handle timeout issues while creating embeddings 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key, request_options={'retry':retry.Retry()})
    vector_store = None
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            vector_store = FAISS.from_texts(text_chunks, embeddings)
            vector_store.save_local("faiss_index")
            st.success("Embeddings created and saved successfully.")
            break
        except Exception as e:
            st.error(f"Error embedding content: {e}. Retrying... ({attempt + 1}/{retry_attempts})")
            time.sleep(2)  # Wait for 2 seconds before retrying
            continue
    if vector_store is None:
        st.error("Failed to create embeddings after multiple attempts.")

def load_vector_store(api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store

# This function will allow user to see the chain of conversation for a particular topic and help build on context for more accurate answers
def get_conversational_chain(api_key):
    
    #This is where prompt engineering happens. It is crucial to get this right as the length and articluation of prompt affects response quality.
    prompt_template = """
    Answer the following question in as detailed a manner as possible from the provided context.
    Make sure to provide all the details. If the answer is not available, don't hallucinate and do not provide a wrong answer,
    simply inform that you are unable to find a relevant answer.

    Context: \n {context}?\n

    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    vector_store = load_vector_store(api_key)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# This snippet will be responsible for setting up the Streamlit page
def main():
    st.header("Document Chatbot panel:")

    user_question = st.text_input("Ask a question from uploaded PDF files", key="user_question")

    if user_question and api_key:
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your documents", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Done!")

if __name__ == "__main__":
    main()

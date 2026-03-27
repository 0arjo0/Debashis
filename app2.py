import streamlit as st
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from pdfminer.high_level import extract_text
import tempfile
import json
import httpx

st.set_page_config(page_title="AI Travel Planner ✈️", layout="wide")
client = httpx.Client(verify=False)

st.title("🌍 AI Travel Itinerary Summarizer")
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "locations" not in st.session_state:
    st.session_state.locations = []

uploaded_files = st.file_uploader(
    "Upload Files",
    type=["pdf","txt","json"],
    accept_multiple_files=True
)

temperature = st.slider("Temperature",0.0,1.0,0.3)

all_text = ""

if uploaded_files:
    for file in uploaded_files:
        if file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                all_text += extract_text(tmp.name)
        elif file.type == "application/json":
            data = json.load(file)
            all_text += json.dumps(data)
            if isinstance(data,list):
                for i in data:
                    if "source" in i:
                        st.session_state.locations.append(i["source"])
                    if "destination" in i:
                        st.session_state.locations.append(i["destination"])
        else:
            all_text += str(file.read(),"utf-8")

if all_text and st.session_state.rag_chain is None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(all_text)

    embedding_model = OpenAIEmbeddings(
        base_url="https://genailab.tcs.in",
        model="azure/genailab-maas-text-embedding-3-large",
        api_key="YOUR_KEY",
        http_client=client
    )

    vectordb = Chroma.from_texts(chunks, embedding_model)

    llm = ChatOpenAI(
        base_url="https://genailab.tcs.in",
        model="azure/genailab-maas-gpt-35-turbo",
        api_key="YOUR_KEY",
        http_client=client,
        temperature=temperature
    )

    st.session_state.rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever()
    )

if st.session_state.rag_chain:
    locations = list(set(st.session_state.locations)) or ["Delhi","Mumbai","Goa"]

    source = st.selectbox("Source", locations)
    destination = st.selectbox("Destination", locations)

    start = st.date_input("Start Date")
    end = st.date_input("End Date")

    if st.button("Generate"):
        query = f"Travel from {source} to {destination} from {start} to {end}"
        result = st.session_state.rag_chain.invoke(query)
        st.write(result["result"])

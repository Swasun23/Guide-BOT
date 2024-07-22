import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader
from langchain_core.documents import Document

load_dotenv()

if "emb" not in st.session_state:
    st.session_state["emb"] = OpenAIEmbeddings()

embedding = st.session_state["emb"]

if "db" not in st.session_state:
    st.session_state["db"] = AstraDBVectorStore(
    embedding=embedding,
    namespace="default_keyspace",
    collection_name="testchat",
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
)

db = st.session_state["db"]

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("The GUIDE")

if st.button("Clear Context"):
    st.session_state.messages =[]
    st.write("Context cleared.")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_similar_doc(ip):
    results = db.similarity_search_with_score(ip, k=4)
    #Fetch vectors that have a similarity score of more that 0.85
    content = [res[0].page_content for res in results if(res[1]>0.85)]
    if(len(content)):
        return ''.join(content)
    else:
        return ""

def get_full_prompt(ip,context=""):
    if(context==""):
        #if no vectors similar to query there is no context added
        return ip
    else:
        return ip+", context:"+f"{context}"

uploaded_files = st.file_uploader("Choose files", type=["txt", "csv", "pdf", "docx"], accept_multiple_files=True)

#Split the document into chunks
splitter = CharacterTextSplitter(separator='\n',chunk_size=800,chunk_overlap=50,length_function=len)

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        st.write(f"Filename: {uploaded_file.name} uploaded successfully")

        # Handle PDF files
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            text = ""
            docs = []
            for i,page in enumerate(reader.pages):
                content = page.extract_text()
                if content:
                    text += content
            dbtext = splitter.split_text(text)
            metadta = {'source':f'{uploaded_file.name}'}
            for d in dbtext:
                doc = Document(page_content=d,metadata=metadta)
                docs.append(doc)

            #upload received douments to the vector db
            db.add_documents(docs)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    rev_doc = get_similar_doc(prompt)
    FullPrompt = get_full_prompt(prompt,rev_doc)
    st.session_state.messages.append({"role": "user", "content": FullPrompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
        stream=True,
            )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})



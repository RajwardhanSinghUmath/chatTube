import streamlit as st
import os
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="ChatTube", page_icon="📹", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #fafafa;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

st.title("📹 Chat with YouTube Content")
st.markdown("### Paste a video link, extract wisdom, and chat via GPT-4o-mini.")

# Sidebar
with st.sidebar:
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.warning("⚠️ Access to GPT-4o-mini requires an OpenAI API Key.")
        
    st.markdown("---")
    st.markdown("**Powered by:**")
    st.markdown("- 🦜 LangChain")
    st.markdown("- 🧠 GPT-4o-mini")
    st.markdown("- 🔍 ChromaDB")
    st.markdown("- 🤗 HuggingFace Embeddings")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_url" not in st.session_state:
    st.session_state.processed_url = ""

# Input Section
col1, col2 = st.columns([3, 1])
with col1:
    youtube_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
with col2:
    process_btn = st.button("Process Video", type="primary")

# Helper function to get video ID
def get_video_id(url):
    from urllib.parse import urlparse, parse_qs
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            p = parse_qs(query.query)
            return p['v'][0]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
    return None

def fetch_details_custom(url):
    import requests
    import re
    try:
        response = requests.get(url)
        matches = re.findall(r'<title>(.*?)</title>', response.text)
        title = matches[0].replace(" - YouTube", "") if matches else "Unknown Video"
        return title, "Unknown Author"
    except:
        return "Unknown Video", "Unknown Author"

# Custom Transcript Fetcher
def safe_load_transcript(url):
    import requests
    from langchain_core.documents import Document
    
    video_id = get_video_id(url)
    if not video_id:
        raise ValueError("Could not parse Video ID")

    # Try 1: User provided API
    api_url = f"https://tubetext.vercel.app/youtube/transcript?video_id={video_id}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            full_text = " ".join([item.get('text', '') for item in data])
        else:
             raise Exception(f"API Error {response.status_code}")
             
    except Exception as e:
        # Fallback to local library if API fails
        print(f"External API failed ({e}), trying local library...")
        from youtube_transcript_api import YouTubeTranscriptApi
        
        try:
             # Try new object-oriented API
             api = YouTubeTranscriptApi()
             transcript_list = api.list(video_id)
             try:
                 transcript = transcript_list.find_transcript(['en', 'en-US'])
             except:
                 transcript = next(iter(transcript_list))
            
             data = transcript.fetch()
             # FetchedTranscript returns objects in newer versions
             full_text = " ".join([item.text for item in data])
             
        except AttributeError:
            # Fallback to static API (returns dicts)
            data = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join([item['text'] for item in data])

    # Get metadata manually
    title, author = fetch_details_custom(url)
    
    return [Document(page_content=full_text, metadata={"source": url, "title": title, "author": author})]

# Processing Logic
if process_btn:
    if not api_key:
        st.error("Please provide an OpenAI API Key in the sidebar.")
    elif not youtube_url:
        st.error("Please enter a valid YouTube URL.")
    else:
        with st.status("Processing video...", expanded=True) as status:
            try:
                st.write("📥 Fetching transcript...")
                
                # Use custom loader
                docs = safe_load_transcript(youtube_url)
                
                title = docs[0].metadata.get("title", "Video")
                st.write(f"✅ Found: **{title}**")
                
                st.write("✂️ Splitting text chunks...")
                # 2. Split Text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                st.write("🧠 Generating embeddings (HuggingFace)...")
                # 3. Embeddings (Open Source)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                st.write("💾 Storing in ChromaDB...")
                # 4. Store in Chroma
                vector_store = Chroma.from_documents(
                    documents=splits, 
                    embedding=embeddings,
                    collection_name="chat_tube_session"
                )
                
                # Update session state
                st.session_state.vector_store = vector_store
                st.session_state.processed_url = youtube_url
                st.session_state.chat_history = [] 
                
                status.update(label="Complete! Ready to chat.", state="complete", expanded=False)
                st.rerun()
                
            except Exception as e:
                status.update(label="Failed", state="error", expanded=True)
                st.error(f"Error occurred: {str(e)}")

# Chat Interface
if st.session_state.vector_store:
    st.markdown("---")
    st.markdown(f"#### 💬 Chatting about: *{st.session_state.processed_url}*")
    
    # Setup Chain
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.3)
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
    
    system_prompt = (
        "You are a helpful assistant designed to answer questions about a YouTube video based on its transcript. "
        "Use the provided context to answer the user's question accurately. "
        "If the answer is not in the context, clearly state that you don't know based on the video info. "
        "Keep the tone friendly and informative."
        "\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Display History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input
    if user_input := st.chat_input("Ask a question about the video..."):
        # User message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context..."):
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

elif not st.session_state.vector_store and not process_btn:
    st.info("👆 Enter a YouTube link and click 'Process Video' to start!")

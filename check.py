import streamlit as st
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from pytube import YouTube
import os
import re
import time
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from typing import Any
load_dotenv()

# Set up environment variables


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "video_info" not in st.session_state:
    st.session_state.video_info = None
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0


# Function to process YouTube video
def process_youtube_video(url):
    try:
        # Extract video ID
        video_id = re.findall(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)[0]
        
        # Load YouTube transcript
        loader = YoutubeLoader.from_youtube_url(f"https://www.youtube.com/watch?v={video_id}", add_video_info=True)
        transcript = loader.load()

        # Get additional video info
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        
        video_info = {
            "title": yt.title,
            "author": yt.author,
            "description": yt.description,
            "views": yt.views,
            "length": time.strftime("%H:%M:%S", time.gmtime(yt.length)),
            "thumbnail": yt.thumbnail_url
        }

        

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(transcript)

        # Create vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        return vectorstore, video_info
    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")

# Function to set up RAG chain



def setup_rag_chain(vectorstore: Any) -> ConversationalRetrievalChain:
    """
    Set up a Retrieval-Augmented Generation (RAG) chain for question answering about a video.

    Args:
        vectorstore (Any): The vector store containing the video information.

    Returns:
        ConversationalRetrievalChain: The configured RAG chain.
    """
    llm = ChatGroq(temperature=0.1, model_name="llama3-70b-8192")
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    prompt = PromptTemplate.from_template("""
    You are a helpful assistant that can answer questions about the video.
    Given the following context and question about the video, provide a relevant answer.
    
    Context: {context}
    Question: {question}
    
    Answer: Let's address this based on the information provided about the video.
    
    {answer}
    
    Is there anything else I can help you with regarding the video?
    """)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return chain



# Streamlit UI
st.set_page_config(page_title="YouTube Video Chat", page_icon="🎥", layout="wide")

st.title("🎥💬 YouTube Video Chat")

# Sidebar for YouTube URL input and video info
with st.sidebar:
    st.header("Video Input")
    youtube_url = st.text_input("Enter YouTube Video URL:")
    process_button = st.button("Process Video")

    if process_button and youtube_url:
        with st.spinner("Processing video..."):
            try:
                st.session_state.vectorstore, st.session_state.video_info = process_youtube_video(youtube_url)
                st.session_state.rag_chain = setup_rag_chain(st.session_state.vectorstore)
                st.success("Video processed successfully!")
            except Exception as e:
                st.error(str(e))

    if st.session_state.video_info:
        st.header("Video Information")
        st.image(st.session_state.video_info["thumbnail"], use_column_width=True)
        st.write(f"**Title:** {st.session_state.video_info['title']}")
        st.write(f"**Author:** {st.session_state.video_info['author']}")
        st.write(f"**Views:** {st.session_state.video_info['views']:,}")
        st.write(f"**Length:** {st.session_state.video_info['length']}")
        with st.expander("Video Description"):
            st.write(st.session_state.video_info['description'])

    st.header("Chat Statistics")
    st.write(f"Total Tokens Used: {st.session_state.total_tokens:,}")


# Main chat interface
st.header("Chat")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Ask a question about the video:"):
    if not st.session_state.vectorstore:
        st.error("Please process a YouTube video first!")
    else:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                with get_openai_callback() as cb:
                    response = st.session_state.rag_chain({"question": user_input})
                    ai_response = response['answer']
                    
                    # Update token usage and cost
                    st.session_state.total_tokens += cb.total_tokens
                    
                
                # Add AI response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                st.markdown(ai_response)

# Control buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()
with col2:
    if st.button("Reset App"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("Powered by Langchain, OpenAI, and Groq | Created with Streamlit")
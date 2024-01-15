from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import YoutubeLoader
from dotenv import load_dotenv
from pytube import YouTube
import streamlit as st
import faiss
import pickle
import os

load_dotenv()
main_placeholder = st.empty()
main_placeholder.text("Place the Youtube URL on the sidebar 🙏🙏🙏")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=100, memory_key="chat_history", return_messages=True)
template = """I'm a helpful bot, ready to answer your question based on the provided context. I'll carefully analyze the information and provide a concise, accurate response. If I can't find the answer within the context, I'll honestly say so rather than make up an answer. Feel free to ask me anything!

Context:
{context}

{question}

Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=template)


@st.cache_resource
def loadembed(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    docs = loader.load()

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docsvec = r_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore_openAI = FAISS.from_documents(docsvec, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")

    vectorstore_openAI.save_local("vectorstore_openAI")

    vectorstore = FAISS.load_local("vectorstore_openAI", embeddings)

    main_placeholder.text("done ✅")
    main_placeholder.empty()

    return vectorstore


@st.cache_data
def get_video_info(url):
    try:
        yt = YouTube(url)
        title = yt.title
        thumbnail = yt.thumbnail_url
        duration = yt.length
        views = yt.views
        uploader = yt.author

        return title, thumbnail, duration, views, uploader
    except Exception as e:
        st.error("Error retrieving video details. Please check the URL.")
        return None, None, None, None, None


def llmqa(word, retriever):
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=retriever,
                                           chain_type_kwargs={
                                               "prompt": QA_CHAIN_PROMPT},
                                           memory=memory)
    result = qa_chain({"query": word})
    answer = result["result"]
    return answer


def chatbot(url):
    vectorstore = st.session_state.get('vectorstore')
    retriever = st.session_state.get('retriever')

    if vectorstore is None or retriever is None:
        vectorstore = loadembed(url)
        retriever = vectorstore.as_retriever()
        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = retriever

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if user_input := st.chat_input("Enter the Question about the video: "):
        # Display user message in chat message container
        st.chat_message("user").markdown(user_input)
        # Add user message to chat history
        st.session_state.messages.append(
            {"role": "user", "content": user_input})

        # Reverse the word
        try:
            response = llmqa(user_input, retriever)
        except:
            response = "please enter the youtube url on the sidebar"

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response})


st.title("YouTube Video Details")
video_url = st.sidebar.text_input("Enter YouTube Video URL:")
if video_url:
    try:
        title, thumbnail, duration, views, uploader = get_video_info(
            video_url)
        if title:
            st.sidebar.caption("If want to put another url reload the page")
            st.sidebar.write(f"Uploader: {uploader}")
            st.sidebar.image(thumbnail, caption='Video Thumbnail')
            st.sidebar.write(f"Title: {title}")
            min = int(duration)//60
            sec = duration - (min*60)
            st.sidebar.write(f"Duration: {min} minutes , {sec} seconds")
            st.sidebar.write(f"Views: {views}")

        chatbot(video_url)

    except Exception as e:
        st.write("Please put the correct Youtube Link 😔😔")
        st.write(e)

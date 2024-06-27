from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import faiss
import pickle
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import YoutubeLoader
numvid = input("enter the number of youtube videos")
urls = []
for i in range(int(numvid)):
    url = input("enter the youtube url")
    urls.append(url)


save_dir = "docs/youtube/"
docs = []
for url in urls:
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    docs += (loader.load())

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)

docsvec = r_splitter.split_documents(docs)

load_dotenv()
embeddings = OpenAIEmbeddings()
vectorstore_openAI = FAISS.from_documents(docsvec, embeddings)
with open("faiss_store_openai.pkl", "wb") as f:
    pickle.dump(vectorstore_openAI, f)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

with open("faiss_store_openai.pkl", "rb") as f:
    vectorstore = pickle.load(f)

question = input("enter the question")

memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=100, memory_key="chat_history", return_messages=True)

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you its not  in the video, don't try to make up an answer.Keep the answer as concise as possible. Always say "thanks for asking!" in the next line at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=template)

retriever = vectorstore.as_retriever()
'''
qa = ConversationalRetrievalChain.from_chain_type(
    llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
'''

qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=retriever,
                                       # return_source_documents=True,
                                       chain_type_kwargs={
                                           "prompt": QA_CHAIN_PROMPT},
                                       memory=memory)


result = qa_chain({"query": question})

print(result)

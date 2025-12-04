from dotenv import load_dotenv

load_dotenv()
from typing import Any, Dict, List

# from langchain import hub
from langsmith import Client
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import \
    create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import \
    create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from consts import INDEX_NAME

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docsearch = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    # chat = ChatOpenAI(verbose=True, temperature=0)
    chat = ChatDeepSeek(model="deepseek-chat")

    client = Client()
    rephrase_prompt = client.pull_prompt("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = client.pull_prompt("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_llm2(query: str, chat_history: List[Dict[str, Any]] = []):
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docsearch = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    # chat = ChatOpenAI(model="gpt-4o-mini", verbose=True, temperature=0)
    chat = ChatDeepSeek(model="deepseek-chat")

    client = Client()
    rephrase_prompt = client.pull_prompt("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = client.pull_prompt("langchain-ai/retrieval-qa-chat")

    rag_chain = (
        {
            "context": docsearch.as_retriever() | format_docs,
            "input": RunnablePassthrough(),
        }
        | retrieval_qa_chat_prompt
        | chat
        | StrOutputParser()
    )

    retrieve_docs_chain = (lambda x: x["input"]) | docsearch.as_retriever()

    chain = RunnablePassthrough.assign(context=retrieve_docs_chain).assign(
        answer=rag_chain
    )

    result = chain.invoke({"input": query, "chat_history": chat_history})
    return result

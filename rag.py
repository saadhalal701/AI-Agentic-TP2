import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv.ipython import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

prompt_template = """
Answer the following question based only on the provided context:
<context>
    {context}
</context>
<question>
    {input}
</quesrtion>
"""

llm = ChatOpenAI(model="gpt-4o", temperature=0)


def main():
    st.set_page_config(page_title="RAG", layout="wide")
    st.subheader("Retrieval Augmented generation", divider="blue")

    with st.sidebar:
        st.sidebar.title("Data loader")
        st.image("rag.png")
        pdf_docs = st.file_uploader(label="Load your pdfs", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Loading"):
                content = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        content += page.extract_text()

                splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=512, chunk_overlap=16
                )

                chunks = splitter.split_text(content)
                st.write(chunks)

                embedding_model = OpenAIEmbeddings()
                vector_store = Chroma.from_texts(
                    chunks,
                    embedding_model,
                    collection_name="data_collection",
                )
                retriever = vector_store.as_retriever(
                    kwargs={"k": 5},
                )

                st.session_state.retriever = retriever
    st.subheader("Chatbot")
    user_question = st.text_input("Ask Your Question")
    if user_question:
        context_docs = st.session_state.retriever.invoke(user_question)
        context_list = [d.page_content for d in context_docs]
        context_text = ". ".join(context_list)
        # st.write(context_text)
        prompt = prompt_template.format(context=context_text, input=user_question)

        resp = llm.invoke(prompt)

        st.write(resp.content)


if __name__ == "__main__":
    main()

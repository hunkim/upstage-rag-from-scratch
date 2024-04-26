# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st
from langchain_upstage import (
    UpstageLayoutAnalysisLoader,
    GroundednessCheck,
    ChatUpstage as Chat,
    UpstageEmbeddings as Embeddings,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import tempfile, os

from langchain import hub

st.title("LangChain Upstage Solar ChatDoc")
st.write(
    "This is a conversational AI that can chat with you about your documents! Get your KEY at https://console.upstage.ai/"
)

llm = Chat()
# https://smith.langchain.com/hub/hunkim/rag-qa-with-history
chat_with_history_prompt = hub.pull("hunkim/rag-qa-with-history")

groundedness_check = GroundednessCheck()


def get_response(user_query, chat_history, retrieved_docs):
    chain = chat_with_history_prompt | llm | StrOutputParser()

    return chain.stream(
        {
            "chat_history": chat_history,
            "question": user_query,
            "context": retrieved_docs,
        }
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

with st.sidebar:
    st.header(f"Add your documents!")

    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file and not uploaded_file.name in st.session_state:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            with st.status("Layout Analyzing ..."):
                layzer = UpstageLayoutAnalysisLoader(file_path, split="page")
                # For improved memory efficiency, consider using the lazy_load method to load documents page by page.
                docs = layzer.load()  # or layzer.lazy_load()

                # Split
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=100
                )
                splits = text_splitter.split_documents(docs)

                st.write(f"Number of splits: {len(splits)}")

            with st.status(f"Vectorizing {len(splits)} splits ..."):
                # Embed
                vectorstore = Chroma.from_documents(
                    documents=splits, embedding=Embeddings()
                )

                st.write("Vectorizing the document done!")

                st.session_state.retriever = vectorstore.as_retriever(k=10)
                # processed
                st.session_state[uploaded_file.name] = True

        st.success("Ready to Chat!")


for message in st.session_state.messages:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        # if message.response_metadata.get("context"):
        #    with st.status("Got Context"):
        #        st.write(message.response_metadata.get("context"))
        st.markdown(message.content)

if prompt := st.chat_input("What is up?", disabled=not st.session_state.retriever):
    st.session_state.messages.append(
        HumanMessage(
            content=prompt,
        )
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Getting context..."):
            st.write("Retrieving...")
            retrieved_docs = st.session_state.retriever.invoke(prompt)
            st.write(retrieved_docs)

        response = st.write_stream(
            get_response(prompt, st.session_state.messages, retrieved_docs)
        )
        gc_result = groundedness_check.run(
            {
                "context": f"Context:{retrieved_docs}\n\nQuestion{prompt}",
                "query": response,
            }
        )

        if gc_result == "grounded":
            gc_mark = "✅"
            st.success("✅ Groundedness check passed!")
        else:
            gc_mark = "❌"
            st.error("❌ Groundedness check failed!")

    st.session_state.messages.append(
        AIMessage(content=f"{gc_mark} {response}"),
    )

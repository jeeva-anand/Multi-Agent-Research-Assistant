import streamlit as st


def sidebar():

    with st.sidebar:

        st.header("📚 Knowledge Base")

        files = st.file_uploader(
            "Upload Research Papers",
            type="pdf",
            accept_multiple_files=True
        )

        st.divider()

        st.subheader("⚙️ Model")

        model = st.selectbox(
            "LLM",
            [
                "llama3.2:3b",
                "mistral:7b"
            ]
        )

        k = st.slider(
            "Retrieved Documents",
            1,
            10,
            4
        )

    return files, model, k

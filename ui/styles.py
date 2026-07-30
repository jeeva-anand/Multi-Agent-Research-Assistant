import streamlit as st


def load_css():

    st.markdown(
        """
        <style>

        .main {
            background-color: #0E1117;
        }


        .stChatMessage {

            border-radius:15px;
            padding:10px;

        }


        h1 {

            font-size:42px;
            font-weight:700;

        }


        </style>

        """,
        unsafe_allow_html=True
    )

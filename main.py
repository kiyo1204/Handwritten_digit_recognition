import streamlit as st

st.set_page_config(page_title="ホーム")
pages = {
    "ページ" : [
    st.Page(page="pages/home.py", title="ホーム"),
    st.Page(page="pages/model_save.py", title="モデルの保存")
    ]
}

if "callbacks" not in st.session_state:
    st.session_state["callbacks"] = {}

page = st.navigation(pages)
page.run()
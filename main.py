import streamlit as st

st.set_page_config(page_title="ホーム")
pages = {
    "ページ" : [
    st.Page(page="pages/home.py", title="ホーム"),
    st.Page(page="pages/model_save.py", title="モデルの保存")
    ]
}

page = st.navigation(pages)
page.run()
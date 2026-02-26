import streamlit as st

st.set_page_config(page_title="ホーム")
pages = {
    "ページ" : [
    st.Page(page="pages/home.py", title="ホーム"),
    st.Page(page="pages/model_save.py", title="モデルの保存"),
    st.Page(page="pages/calculate.py", title="計算")
    ]
}

states = ["callbacks", "play_calculate"]

for state in states:
    if state == "callbacks" and  state not in st.session_state:
        st.session_state[state] = {}
    elif state not in st.session_state:
        st.session_state[state] = False

page = st.navigation(pages)
page.run()
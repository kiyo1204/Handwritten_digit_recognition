import streamlit as st

# ページ情報の初期化
def init_page():
    st.set_page_config(page_title="ホーム")
    pages = {
        "ページ" : [
        st.Page(page="pages/home.py", title="ホーム"),
        st.Page(page="pages/model_save.py", title="モデルの保存")
        ]
    }
    page = st.navigation(pages)
    return page

# ページ共通の関数の初期化
def init_session_state():
    states = ["callbacks", "play_calculate", "key_num"]

    for state in states:
        if state == "callbacks" and  state not in st.session_state:
            st.session_state[state] = {}
        elif state == "key_num" and state not in st.session_state:
            st.session_state[state] = 0
        elif state not in st.session_state:
            st.session_state[state] = False

if __name__ == "__main__":
    init_session_state()
    # ページの実行
    init_page().run()
    
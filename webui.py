import os
from streamlit_option_menu import option_menu
import streamlit as st

from sympy import use


if __name__ == "__main__":
    st.set_page_config(
        page_title="YLangchain-Chatchat",
        page_icon="./img/chatchat_icon_blue_square_v2.png",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/YLangChain/YLangchain-Chatchat/issues",
            "Report a bug": "https://github.com/YLangChain/YLangchain-Chatchat/issues",
            "About": "欢迎使用YLangchain-Chatchat！",
        },
    )
    with st.sidebar:
        st.image("img\logo-long-chatchat-trans-v2.png", use_column_width=True)
        st.caption(f"""<p align="right">当前版本: 0.0.1</p>""", unsafe_allow_html=True)
        selected_option = option_menu(
            menu_title="", options=["对话", "查询"], default_index=0
        )
        

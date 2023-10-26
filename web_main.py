# -*- coding: utf-8 -*-
"""
@author: Rato
"""

import os
from pathlib import Path

import streamlit as st
from PIL import Image


def check_sessionstate(id: str):
    if id not in st.session_state:
        st.session_state[id] = False


def design_experiments():

    from design_of_experiment.method import ExpDesign
    check_sessionstate("doe_cls")

    st.session_state["doe_cls"] = False
    _pathset = Path(os.path.dirname(__file__))

    # サイドバー_フォーマットのダウンロード機能
    st.sidebar.write()
    st.sidebar.subheader("InputFormat:")
    _path = _pathset.joinpath('design_of_experiment\\')
    format_path = _path.joinpath("Design_of_experiment_format.xlsx")
    st.sidebar.write("""Obtain format for variable range
                     setting from below download button.""")
    st.sidebar.download_button(

        label="Download",
        data=open(str(format_path), "rb"),
        file_name=format_path.name)
    # メイン画面
    doe_col0, doe_col1 = st.columns(2)  # 表示場所確保
    method_list = [
        "d_optimal_select", "latin_hypercube",
        "monte_carlo", "defscreen"]
    method = doe_col0.selectbox(
        "Select_method", method_list)
    h_mess = "This is ignored if defscreen is selected."
    sampling_num = int(doe_col1.number_input(
        "Planned number of experiments", 1, value=10, help=h_mess))

    # 設計範囲条件のインプット
    uploaded_file = st.file_uploader(
        'upload DOE condition format', type=['xlsx'])
    if uploaded_file:
        if uploaded_file is not None:
            _cwd = os.getcwd()
            this_doe = ExpDesign(
                uploaded_file, sheet_name='condition',
                save_path=f"{_cwd}\\exp_design_result",
                method=method)
            st.session_state["doe_cls"] = this_doe

    doe_col10, doe_col11 = st.columns(2)
    if st.session_state["doe_cls"]:
        doe_start = doe_col10.button('Run')
        if doe_start:  # 実行毎にクラス内で結果の上書き
            doe_df, mess, doe_fig_path = this_doe.run(
                sampling_num=sampling_num)
            st.subheader(mess)
            st.write(doe_df)
            image = Image.open(f"{doe_fig_path}.png")
            st.image(image)
            xlsx_name = this_doe.to_excel(start_row=1, start_col=1)
            doe_col11.download_button(
                label=xlsx_name.name, data=open(xlsx_name, "rb"),
                file_name=xlsx_name.name)


def main():
    st.set_page_config(
        page_title="Rato-12 Apps",
        layout="wide",
        initial_sidebar_state="expanded",
        )
    with st.sidebar:
        # image Image.open('kunishou.png')
        # st.image (inage, width=120)
        # image Image.open('space.jpg')
        # st.image (image, use_column_width=True)

        st.title("Experimental condition design")

    design_experiments()

    st.sidebar.markdown("")
    st.sidebar.markdown("[GitHub] (https://github.com/Rato-12/pyDOE_rato)")


if __name__ == '__main__':
    main()

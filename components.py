"""
このファイルは、画面表示に特化した関数定義のファイルです。
"""

import logging
import streamlit as st
import constants as ct

def display_app_title():
    """
    タイトル表示
    """
    st.markdown(f"## {ct.APP_NAME}")


def display_sidebar():
    """
    サイドバーの表示
    """
    with st.sidebar:
        st.markdown("## AIエージェント機能の利用有無")

        col1, col2 = st.columns([100, 1])
        with col1:
            st.session_state.agent_mode = st.selectbox(
                label="",
                options=[ct.AI_AGENT_MODE_ON, ct.AI_AGENT_MODE_OFF],
                label_visibility="collapsed"
            )
        
        st.markdown("## 問い合わせモード")

        col1, col2 = st.columns([100, 1])
        with col1:
            st.session_state.contact_mode = st.selectbox(
                label="",
                options=[ct.CONTACT_MODE_OFF, ct.CONTACT_MODE_ON],
                label_visibility="collapsed"
            )
        
        st.divider()

        st.markdown("**【AIエージェントとは】**")
        st.code("質問に対して適切と考えられる回答を生成できるまで、生成AIロボット自身に試行錯誤してもらえる機能です。自身の回答に対して評価・改善を繰り返すことで、より優れた回答を生成できます。", wrap_lines=True)
        st.warning("AIエージェント機能を利用する場合、回答生成により多くの時間を要する可能性が高いです。", icon=":material/warning:")

        st.markdown("**【問い合わせモードとは】**")
        st.code("問い合わせモードを「ON」にしてメッセージを送信すると、担当者に直接届きます。", wrap_lines=True)


def display_initial_ai_message():
    """
    AIメッセージの初期表示
    """
    with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
        st.success("こちらは弊社に関する質問にお答えする生成AIチャットボットです。AIエージェントの利用有無を選択し、画面下部のチャット欄から質問してください。")
        st.warning("具体的に入力したほうが期待通りの回答を得やすいです。", icon=ct.WARNING_ICON)


def display_conversation_log(chat_message):
    """
    会話ログの一覧表示
    """
    for index, message in enumerate(st.session_state.messages):
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar=ct.AI_ICON_FILE_PATH):
                st.markdown(message["content"])
                display_after_feedback_message(index, chat_message)
        else:
            with st.chat_message(message["role"], avatar=ct.USER_ICON_FILE_PATH):
                st.markdown(message["content"])
                display_after_feedback_message(index, chat_message)
        
def display_after_feedback_message(index, chat_message):
    """
    ユーザーフィードバック後のメッセージ表示

    Args:
        result: LLMからの回答
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    if st.session_state.feedback_no_flg and chat_message:
        st.session_state.feedback_no_flg = False

    if index == len(st.session_state.messages) - 1:
        if st.session_state.feedback_yes_flg:
            st.caption(ct.FEEDBACK_YES_MESSAGE)
            st.session_state.feedback_yes_flg = False
        if st.session_state.feedback_no_flg:
            st.caption(ct.FEEDBACK_NO_MESSAGE)
            st.session_state.dissatisfied_reason = st.text_area("", label_visibility="collapsed")
            if st.button(ct.FEEDBACK_BUTTON_LABEL):
                if st.session_state.dissatisfied_reason:
                    logger.info({"dissatisfied_reason": st.session_state.dissatisfied_reason})
                st.session_state.feedback_no_flg = False
                st.session_state.feedback_no_reason_send_flg = True
                st.rerun()
        if st.session_state.feedback_no_reason_send_flg:
            st.session_state.feedback_no_reason_send_flg = False
            st.caption(ct.FEEDBACK_THANKS_MESSAGE)

def display_llm_response(result):
    """
    LLMからの回答表示

    Args:
        result: LLMからの回答
    """
    st.markdown(result)
    if st.session_state.contact_mode == ct.CONTACT_MODE_OFF:
        if st.session_state.answer_flg:
            st.caption(ct.FEEDBACK_REQUIRE_MESSAGE)


def display_feedback_button():
    """
    フィードバックボタンの表示
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    if st.session_state.answer_flg:
        col1, col2, col3 = st.columns([1, 1, 5])
        with col1:
            if st.button(ct.FEEDBACK_YES):
                logger.info({"feedback": ct.SATISFIED})
                st.session_state.answer_flg = False
                st.session_state.feedback_yes_flg = True
                st.rerun()
        with col2:
            if st.button(ct.FEEDBACK_NO):
                logger.info({"feedback": ct.DISSATISFIED})
                st.session_state.answer_flg = False
                st.session_state.feedback_no_flg = True
                st.rerun()
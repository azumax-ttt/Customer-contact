"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

from dotenv import load_dotenv
import logging
import streamlit as st
import utils
from initialize import initialize
import components as cn
import constants as ct

st.set_page_config(
    page_title=ct.APP_NAME
)

load_dotenv()

logger = logging.getLogger(ct.LOGGER_NAME)

try:
    initialize()
except Exception as e:
    logger.error(f"{ct.INITIALIZE_ERROR_MESSAGE}\n{e}")
    st.error(utils.build_error_message(ct.INITIALIZE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    st.stop()

if not "initialized" in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)


cn.display_app_title()

cn.display_sidebar()

cn.display_initial_ai_message()


st.markdown(ct.STYLE, unsafe_allow_html=True)

chat_message = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)

try:
    cn.display_conversation_log(chat_message)
except Exception as e:
    logger.error(f"{ct.CONVERSATION_LOG_ERROR_MESSAGE}\n{e}")
    st.error(utils.build_error_message(ct.CONVERSATION_LOG_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    st.stop()

if chat_message:
    input_tokens = len(st.session_state.enc.encode(chat_message))
    if input_tokens > ct.MAX_ALLOWED_TOKENS:
        with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
            st.error(ct.INPUT_TEXT_LIMIT_ERROR_MESSAGE)
            st.stop()
    st.session_state.total_tokens += input_tokens

    logger.info({"message": chat_message})

    res_box = st.empty()
    with st.chat_message("user", avatar=ct.USER_ICON_FILE_PATH):
        st.markdown(chat_message)
    
    res_box = st.empty()
    try:
        if st.session_state.contact_mode == ct.CONTACT_MODE_OFF:
            with st.spinner(ct.SPINNER_TEXT):
                result = utils.execute_agent_or_chain(chat_message)
        else:
            with st.spinner(ct.SPINNER_CONTACT_TEXT):
                result = utils.notice_slack(chat_message)
    except Exception as e:
        logger.error(f"{ct.MAIN_PROCESS_ERROR_MESSAGE}\n{e}")
        st.error(utils.build_error_message(ct.MAIN_PROCESS_ERROR_MESSAGE), icon=ct.ERROR_ICON)
        st.stop()
    
    utils.delete_old_conversation_log(result)

    with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
        try:
            cn.display_llm_response(result)

            logger.info({"message": result})
        except Exception as e:
            logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")
            st.error(utils.build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            st.stop()
    
    st.session_state.messages.append({"role": "user", "content": chat_message})
    st.session_state.messages.append({"role": "assistant", "content": result})

if st.session_state.contact_mode == ct.CONTACT_MODE_OFF:
    cn.display_feedback_button()
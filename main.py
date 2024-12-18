import os
import streamlit as st
from src.ui.login import login_page, register_page
from src.ui.user_data_storage import create_folder_if_not_exist, read_credentials, write_credentials, Credentials
from src.qa.load import load_model
from src.qa.generate import generate_answer, output_kw, select_context, generate_direct_answer
from src.match.match_utils import match_entities_and_save
from src.kg.disease_to_cypher import DiseaseToCypher


def main(is_admin, usname):  # , model, tokenizer, model_name):

    st.title(f"医疗智能问答机器人")

    with st.sidebar:
        col1, _ = st.columns([0.6, 0.6])
        with col1:
            st.image("logo.jpg", use_container_width=True)

        st.caption(
            f"""<p align="left">欢迎您，{'管理员' if is_admin else '用户'}{usname}！当前版本：{1.0}</p>""",
            unsafe_allow_html=True,
        )

        if 'chat_windows' not in st.session_state:
            st.session_state.chat_windows = [[]]
            st.session_state.messages = [[]]

        if st.button('新建对话窗口'):
            st.session_state.chat_windows.append([])
            st.session_state.messages.append([])

        window_options = [f"对话窗口 {i + 1}" for i in range(len(st.session_state.chat_windows))]
        selected_window = st.selectbox('请选择对话窗口:', window_options)
        active_window_index = int(selected_window.split()[1]) - 1

        selected_option = st.selectbox(
            label='请选择大语言模型:',
            options=['Qwen 2.5', 'Llama 3.1', 'Huatuo']
        )
        if selected_option != st.session_state.model_name:
            st.session_state.model_name = selected_option
            st.session_state.model, st.session_state.tokenizer, _ = load_model(st.session_state.model_name)
            print("模型重载完成")
        
        st.session_state.is_RAG = st.selectbox(
            label='请选择是否包含知识图谱RAG:',
            options=['LLM', 'LLM+RAG']
        )        
        
        # choice = 'qwen:32b' if selected_option == 'Qwen 1.5' else 'llama2-chinese:13b-chat-q8_0'

        show_ent = show_int = show_prompt = True  # False

        if is_admin:
            show_ent = st.sidebar.checkbox("显示实体识别结果")
            show_int = st.sidebar.checkbox("显示意图识别结果")
            show_prompt = st.sidebar.checkbox("显示查询的知识库信息")
            if st.button('修改知识图谱'):
            # 显示一个链接，用户可以点击这个链接在新标签页中打开百度
                st.markdown('[点击这里修改知识图谱](http://127.0.0.1:7474/)', unsafe_allow_html=True)

        if st.button("返回登录"):
            st.session_state.logged_in = False
            st.session_state.admin = False
            st.rerun()
    
    # client = py2neo.Graph('http://localhost:7474', user='neo4j', password='wei8kang7.long', name='neo4j')

    current_messages = st.session_state.messages[active_window_index]

    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    #         if message["role"] == "assistant":
    #             if show_ent:
    #                 with st.expander("实体识别结果"):
    #                     st.write(message.get("ent", ""))
    #             if show_int:
    #                 with st.expander("意图识别结果"):
    #                     st.write(message.get("yitu", ""))
    #             if show_prompt:
    #                 with st.expander("点击显示知识库信息"):
    #                     st.write(message.get("prompt", ""))

    if query := st.chat_input("Ask me anything!", key=f"chat_input_{active_window_index}"):
        current_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        response_placeholder = st.empty()

        if st.session_state.is_RAG == 'LLM':
            show_ent = show_int = show_prompt = False
            response_placeholder.text("正在由大模型直接生成回复...")
            last = generate_direct_answer(query, st.session_state.model, st.session_state.tokenizer)

        elif st.session_state.is_RAG == 'LLM+RAG':
            response_placeholder.text("正在识别意图...")
            query = current_messages[-1]["content"]
            data = output_kw(query, st.session_state.model, st.session_state.tokenizer)

            response_placeholder.text("意图已识别，正在匹配知识图谱...")
            diseases_names = match_entities_and_save(data)
            disease_to_cypher = DiseaseToCypher()
            if diseases_names['Type'] == 'symptom':
                symptoms = diseases_names['match']
                diseases_names = disease_to_cypher.get_diseases_by_fuzzy_symptoms(symptoms, debug=False)
                diseases_info = disease_to_cypher.get_disease_info(diseases_names)
            else:
                diseases_info = disease_to_cypher.get_disease_info(diseases_names['match'][0])

            response_placeholder.text("匹配已完成，正在生成回复...")
            diseases_info = select_context(diseases_info, data)
            last = generate_answer(query, diseases_info, data['intent'], st.session_state.model, st.session_state.tokenizer)
        else:
            raise ValueError("未知的模型选择。")

        response_placeholder.empty()
        response_placeholder.markdown(last)
        response_placeholder.markdown("")

        # knowledge = re.findall(r'<提示>(.*?)</提示>', prompt)
        # zhishiku_content = "\n".join([f"提示{idx + 1}, {kn}" for idx, kn in enumerate(knowledge) if len(kn) >= 3])
        with st.chat_message("assistant"):
            st.markdown(last)
            if show_ent:
                with st.expander("实体识别结果"):
                    st.write(data['entity'])
            if show_int:
                with st.expander("意图识别结果"):
                    st.write(data['intent'])
            if show_prompt:                
                with st.expander("点击显示知识库信息"):
                    st.write(diseases_info)
        if st.session_state.is_RAG == 'LLM+RAG':
            # current_messages.append({"role": "assistant", "content": last, "yitu": yitu, "prompt": zhishiku_content, "ent": str(entities)})
            current_messages.append({"role": "assistant", "content": last, "yitu": data['intent'], "prompt": diseases_info, "ent": data["entity"]})
        elif st.session_state.is_RAG == 'LLM':
            current_messages.append({"role": "assistant", "content": last})

    st.session_state.messages[active_window_index] = current_messages


if __name__ == "__main__":

    # 文件存储位置
    storage_folder = "tmp"
    storage_file = os.path.join(storage_folder, "user_credentials.json")

    # 确保文件夹存在
    create_folder_if_not_exist(storage_folder)

    # 读取现有的用户数据
    credentials = read_credentials(storage_file)

    # 如果初始文件为空，则初始化管理员账户
    if not credentials:
        admin = Credentials("admin", "admin123", True)
        credentials['admin'] = admin
        write_credentials(storage_file, credentials)

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'admin' not in st.session_state:
        st.session_state.admin = False
    if 'usname' not in st.session_state:
        st.session_state.usname = ""
    if 'model' not in st.session_state:
        st.session_state.model_name = 'Qwen 2.5'
        st.session_state.model, st.session_state.tokenizer, _ = load_model(st.session_state.model_name)

    if not st.session_state.logged_in:
        # 显示注册和登录选项
        st.sidebar.title("导航")
        app_mode = st.sidebar.selectbox("选择操作", ["登录", "注册"])
        if app_mode == "登录":
            login_page(credentials)
        elif app_mode == "注册":
            register_page(credentials)
    else:
        main(st.session_state.admin, 
             st.session_state.usname,)
            #  st.session_state.model, 
            #  st.session_state.tokenizer,
            #  st.session_state.model_name)
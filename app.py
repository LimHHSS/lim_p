import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
import tempfile
from langchain.document_loaders import PyPDFLoader

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
uploaded_file = st.sidebar.file_uploader("PDF 파일 업로드", type="pdf")

system_message = '''
너는 배려 넘치는 어시스턴트 챗봇이야.
항상 친근하게 대답해줘. 영어로 질문해도 무조건 한글로 답변해줘.
답변할 때 업로드된 PDF 파일의 내용을 기반으로 정확하고 관련된 정보를 제공해줘. 
짧게 물어보더라도 가능한 한 구체적이고 자세하게 설명해 주고, 사용자에게 도움이 될 만한 추가 정보도 제공해줘.
'''

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_message}]
    st.session_state.history = []

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    data = loader.load()

    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(data, embeddings)

    chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0.0, model_name='gpt-4o'), retriever=vectors.as_retriever())

    def conversational_chat(query):
        # "Running..." 메시지 표시
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("답변을 준비중이에요..! 잠시만 기다려주세요. 😊")  # 사용자에게 로딩 중임을 알림

            # ConversationalRetrievalChain을 통해 답변을 가져오기
            result = chain({"question": query, "chat_history": st.session_state.history})
            message_placeholder.markdown(result["answer"] + " 😊")

        st.session_state.history.append((query, result["answer"]))
        return result["answer"]

    st.title("ChatGPT 형태의 챗봇")

    # 채팅 메시지 출력
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 채팅 입력
    if prompt := st.chat_input("PDF 내용을 질문해보세요!"):
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(prompt)

        assistant_message_content = conversational_chat(prompt)
        assistant_message = {"role": "assistant", "content": assistant_message_content}

        st.session_state.messages.append(assistant_message)
else:
    st.warning("📂 PDF 파일을 먼저 업로드 해주세요!")
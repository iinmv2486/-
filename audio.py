import os
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage
from sqlalchemy import create_engine

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from dotenv import load_dotenv

class ChatHistoryManager:
    def __init__(self, file_path: str, db_path: str, session_id: str):
        self.file_path = file_path
        self.db_path = db_path
        self.session_id = session_id
        self.connection_string = f"sqlite:///{db_path}"
        
        # 데이터베이스 초기화
        self.initialize_database()
        
        # SQLAlchemy 엔진 생성
        self.engine = create_engine(self.connection_string)
        
    def initialize_database(self):
        if not os.path.exists(self.db_path):
            # 데이터베이스 파일이 없는 경우에 db 생성
            open(self.db_path, 'w').close()
            
    def get_session_history(self, session_id=None):
        if session_id is None:
            session_id = self.session_id
        return SQLChatMessageHistory(session_id=session_id, connection=self.engine)
    
    def store_to_db(self, msg: str, session_id=None):
        if session_id is None:
            session_id = self.session_id
        # 세션 기록 객체 생성
        message_history = self.get_session_history(session_id)
        
        # 사용자 메시지 추가
        message_history.add_user_message(HumanMessage(content=msg))
    
    def store_file_contents_to_db(self):
        if os.path.exists(self.file_path) and self.file_path.endswith('.txt'):
            with open(self.file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    self.store_to_db(line.strip())
    
    def display_all_messages(self, session_id=None):
        if session_id is None:
            session_id = self.session_id
        # 모든 메시지 조회
        message_history = self.get_session_history(session_id)
        messages = message_history.messages
        for msg in messages:
            print(msg.content)


def get_session_history(session_id):
    db_path = "/root/LLM_Bootcamp/음성인식 회의록 작성/data/memory.db"
    connection_string = f"sqlite:///{db_path}"
    engine = create_engine(connection_string)
    return SQLChatMessageHistory(session_id, connection=engine)

if __name__ == "__main__":
    
    load_dotenv()

    model = AzureChatOpenAI(
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        temperature=1.0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a capable secretary. Summarize the conversation after understanding it well"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = prompt | model | StrOutputParser()

    # 저장할 파일 경로들
    file_path_list = {
        "example_session": "/root/LLM_Bootcamp/음성인식 회의록 작성/data/history.txt",
        "example_session1": "/root/LLM_Bootcamp/음성인식 회의록 작성/data/history1.txt"
    }

    db_path = "/root/LLM_Bootcamp/음성인식 회의록 작성/data/memory.db"

    # 각 세션의 대화 기록을 데이터베이스에 저장
    for session_id, file_path in file_path_list.items():
        manager = ChatHistoryManager(file_path, db_path, session_id)
        manager.store_file_contents_to_db()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    msg = chain_with_history.invoke(
        {"input": "내용을 요약해 주세요."},
        config={"configurable": {"session_id": "example_session"}}
    )

    print(f"AI: {msg}")

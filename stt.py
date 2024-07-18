import requests
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def STT(input_path, output_path):
    load_dotenv()

    # Azure OpenAI API 키와 엔드포인트 URL
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = "whisper"  # Whisper API 모델 이름
    api_version = os.getenv("OPENAI_API_VERSION")

    # 요청 헤더 설정
    headers = {
        "api-key": api_key
    }

    # 엔드포인트 URL 구성 
    url = f"{endpoint}/openai/deployments/{deployment_name}/audio/transcriptions?api-version={api_version}"


    # 음성 파일 경로 설정 (예: sample.wav) - 업로드한 영상 경로가 여기에 작성될 예정
    audio_file_path = input_path

    # 음성 파일을 바이너리 형식으로 읽기
    with open(audio_file_path, "rb") as audio_file:
        files = {
            'file': (audio_file_path, audio_file, 'audio/mpeg')
        }
        
        # POST 요청 보내기
        response = requests.post(url, headers=headers, files=files)

    # 응답 출력
    if response.status_code == 200:
        result = response.json()
        text = result['text'].strip()

        voice_document = Document(
            page_content=text,
            metadata={"source": "mp3 file"}
        )

        recursive_text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n","."],
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
        )

        recursive_splitted_document = recursive_text_splitter.split_documents([voice_document])

        # tmp text 파일 링크 이름
        with open(output_path, "w", encoding="utf-8") as f:
            for document in recursive_splitted_document:
                f.write(document.page_content + "\n\n")

    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")

output_path="<output_path>"
input_path="<input_path>"

STT(input_path, output_path)
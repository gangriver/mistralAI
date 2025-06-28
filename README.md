# 국어 규범 RAG 시스템

국어 규범 기반 문서들을 바탕으로 RAG(Retrieval-Augmented Generation) 형태로 질의응답을 수행하는 FastAPI 서버입니다.

## 🚀 주요 기능

- **FastAPI 서버**: RESTful API 제공
- **Mistral 7B GPTQ 모델**: 4bit 양자화된 로컬 LLM 사용
- **LangChain RAG**: 문서 검색 및 생성 기반 질의응답
- **FAISS 벡터 데이터베이스**: 고속 유사도 검색
- **한국어 임베딩**: jhgan/ko-sroberta-multitask 모델 사용

## 📁 프로젝트 구조

```
Mistral/
├── main.py              # FastAPI 메인 서버
├── vector_db.py         # 벡터 데이터베이스 관리
├── rag_chain.py         # RAG 체인 정의
├── requirements.txt     # 의존성 패키지
├── README.md           # 프로젝트 설명서
├── korean_rules/       # 국어 규범 문서 폴더 (.txt 파일들)
├── models/             # 모델 저장 폴더
│   └── mistral-gptq/   # Mistral 7B GPTQ 모델
└── vector_store/       # FAISS 벡터 스토어
```

## 🛠️ 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 준비

`./models/mistral-gptq/` 폴더에 Mistral 7B GPTQ 모델을 다운로드하여 배치하세요.

### 3. 문서 준비

`./korean_rules/` 폴더에 국어 규범 관련 `.txt` 파일들을 배치하세요.

## 🚀 실행 방법

```bash
python main.py
```

서버가 `http://localhost:8000`에서 실행됩니다.

## 📡 API 엔드포인트

### POST /rag
국어 규범에 대한 질의응답을 수행합니다.

**요청:**
```json
{
  "question": "한글 맞춤법에 대해 설명해주세요"
}
```

**응답:**
```json
{
  "answer": "한글 맞춤법은...",
  "source_docs": [
    "./korean_rules/spelling_rules.txt",
    "./korean_rules/grammar_guide.txt"
  ]
}
```

### GET /health
서버 상태를 확인합니다.

## ⚙️ 설정 옵션

### 벡터 데이터베이스 설정 (`vector_db.py`)
- `chunk_size`: 500 (문서 청크 크기)
- `chunk_overlap`: 50 (청크 간 겹침)
- `embeddings_model`: "jhgan/ko-sroberta-multitask"

### RAG 체인 설정 (`rag_chain.py`)
- 검색 문서 수: 4개
- 프롬프트 템플릿: 한국어 최적화

## 🔧 문제 해결

### 모델 로딩 오류
- `./models/mistral-gptq/` 경로에 모델이 올바르게 배치되었는지 확인
- GPU 메모리 부족 시 `device_map="cpu"`로 변경

### 문서 로딩 오류
- `./korean_rules/` 폴더에 `.txt` 파일이 있는지 확인
- 파일 인코딩이 UTF-8인지 확인

### 벡터 스토어 재생성
벡터 스토어를 재생성하려면 `./vector_store/` 폴더를 삭제하고 서버를 재시작하세요.

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. # mistralAI

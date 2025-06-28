from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from rag_chain import RAGChain
from vector_db import VectorDatabase

app = FastAPI(title="국어 규범 RAG API", description="국어 규범 문서 기반 질의응답 시스템")

# 요청/응답 모델 정의
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    source_docs: List[str]

# 전역 변수로 RAG 체인과 벡터 DB 초기화
rag_chain = None
vector_db = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델과 벡터 DB 초기화"""
    global rag_chain, vector_db
    
    print("벡터 데이터베이스 초기화 중...")
    try:
        vector_db = VectorDatabase()
        vector_db.initialize_database()
        print("벡터 데이터베이스 초기화 완료!")
    except Exception as e:
        print(f"벡터 데이터베이스 초기화 실패: {e}")
        vector_db = None
    
    print("RAG 체인 초기화 중...")
    try:
        if vector_db is not None:
            rag_chain = RAGChain(vector_db)
            rag_chain.initialize_chain()
            print("RAG 체인 초기화 완료!")
        else:
            print("벡터 데이터베이스가 없어 RAG 체인을 초기화할 수 없습니다.")
    except Exception as e:
        print(f"RAG 체인 초기화 실패: {e}")
        rag_chain = None
    
    print("서버 초기화 완료!")

@app.post("/rag", response_model=QuestionResponse)
async def rag_query(request: QuestionRequest):
    """RAG 기반 질의응답 엔드포인트"""
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG 체인이 초기화되지 않았습니다. 모델 파일을 확인해주세요.")
    
    try:
        answer, source_docs = rag_chain.query(request.question)
        return QuestionResponse(answer=answer, source_docs=source_docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의 처리 중 오류 발생: {str(e)}")

@app.get("/health")
async def health_check():
    """서버 상태 확인 엔드포인트"""
    status = {
        "status": "healthy",
        "message": "국어 규범 RAG API가 정상적으로 실행 중입니다.",
        "vector_db_loaded": vector_db is not None,
        "rag_chain_loaded": rag_chain is not None
    }
    
    if vector_db is not None:
        status["document_count"] = vector_db.get_document_count()
    
    return status

@app.get("/info")
async def get_info():
    """시스템 정보 반환"""
    info = {
        "vector_db_status": "로드됨" if vector_db is not None else "로드되지 않음",
        "rag_chain_status": "로드됨" if rag_chain is not None else "로드되지 않음"
    }
    
    if vector_db is not None:
        info["document_count"] = vector_db.get_document_count()
    
    if rag_chain is not None:
        info["model_info"] = rag_chain.get_model_info()
    
    return info

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
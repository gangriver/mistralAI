import os
from typing import List, Tuple
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vector_db import VectorDatabase

class MockLLM(LLM):
    """모델이 없을 때 사용할 Mock LLM"""
    
    def _call(self, prompt: str, stop: List[str] = None) -> str:
        return "모델이 로드되지 않아 답변을 생성할 수 없습니다. 모델 파일을 확인해주세요."
    
    @property
    def _llm_type(self) -> str:
        return "mock"

class RAGChain:
    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db
        self.llm = None
        self.qa_chain = None
        self.model_path = "./models/mistral-gptq/"
        
        # 한국어 RAG를 위한 프롬프트 템플릿
        self.prompt_template = """다음은 국어 규범에 관한 질문과 관련 문서들입니다.

문서 내용:
{context}

질문: {question}

위의 문서 내용을 바탕으로 질문에 답변해주세요. 답변은 한국어로 작성하고, 문서에 명시된 내용을 정확히 반영해야 합니다. 문서에 없는 내용은 추측하지 마세요.

답변:"""
    
    def initialize_chain(self):
        """RAG 체인 초기화"""
        print("LLM 모델 로딩 시도 중...")
        
        # 모델 경로 확인
        if not os.path.exists(self.model_path):
            print(f"경고: 모델 경로 {self.model_path}가 존재하지 않습니다.")
            print("Mock LLM을 사용합니다.")
            self.llm = MockLLM()
        else:
            try:
                # AutoGPTQ 모델 로딩 시도
                from auto_gptq import AutoGPTQForCausalLM
                from transformers import AutoTokenizer
                
                print("AutoGPTQ 모델 로딩 중...")
                model = AutoGPTQForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    trust_remote_code=True,
                    inject_fused_attention=False,
                    inject_fused_mlp=False,
                    disable_exllama=True,
                    quantize_config=None
                )
                
                tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                
                # LangChain LLM 래퍼 생성
                from langchain.llms import HuggingFacePipeline
                from transformers import pipeline
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.15
                )
                
                self.llm = HuggingFacePipeline(pipeline=pipe)
                print("LLM 모델 로딩 완료!")
                
            except Exception as e:
                print(f"모델 로딩 중 오류 발생: {e}")
                print("Mock LLM을 사용합니다.")
                self.llm = MockLLM()
        
        # 프롬프트 템플릿 생성
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        # RetrievalQA 체인 생성
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.vector_store.as_retriever(
                search_kwargs={"k": 4}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        print("RAG 체인 초기화 완료!")
    
    def query(self, question: str) -> Tuple[str, List[str]]:
        """질의응답 수행"""
        if self.qa_chain is None:
            return "RAG 체인이 초기화되지 않았습니다.", []
        
        try:
            # RAG 체인으로 질의 수행
            result = self.qa_chain({"query": question})
            
            answer = result.get("result", "답변을 생성할 수 없습니다.")
            
            # 소스 문서 추출
            source_docs = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        source_docs.append(doc.metadata['source'])
                    else:
                        source_docs.append("알 수 없는 출처")
            
            return answer, source_docs
            
        except Exception as e:
            print(f"질의 처리 중 오류: {e}")
            return f"질의 처리 중 오류가 발생했습니다: {str(e)}", []
    
    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        if self.llm is None:
            return {"status": "모델이 로드되지 않음"}
        
        if isinstance(self.llm, MockLLM):
            return {
                "model_name": "Mock LLM",
                "model_path": "N/A",
                "status": "Mock 모드 (실제 모델 없음)"
            }
        
        return {
            "model_name": "Mistral 7B GPTQ",
            "model_path": self.model_path,
            "status": "로드됨"
        } 
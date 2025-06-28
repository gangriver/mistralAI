import os
import glob
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

class VectorDatabase:
    def __init__(self):
        self.documents_path = "./korean_rules/"
        self.embeddings_model_name = "jhgan/ko-sroberta-multitask"
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.vector_store_path = "./vector_store"
        
        self.embeddings = None
        self.vector_store = None
        self.text_splitter = None
    
    def initialize_database(self):
        """벡터 데이터베이스 초기화"""
        print("임베딩 모델 로딩 중...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embeddings_model_name,
            model_kwargs={'device': 'cpu'}
        )
        
        print("텍스트 분할기 초기화...")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        # 기존 벡터 스토어가 있으면 로드, 없으면 새로 생성
        if os.path.exists(self.vector_store_path):
            print("기존 벡터 스토어 로딩 중...")
            self.vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings
            )
        else:
            print("새로운 벡터 스토어 생성 중...")
            self._create_vector_store()
    
    def _create_vector_store(self):
        """문서를 로드하고 벡터 스토어 생성"""
        # 문서 파일들 찾기
        txt_files = glob.glob(os.path.join(self.documents_path, "**/*.txt"), recursive=True)
        
        if not txt_files:
            print(f"경고: {self.documents_path}에서 .txt 파일을 찾을 수 없습니다.")
            # 빈 벡터 스토어 생성
            self.vector_store = FAISS.from_texts(
                ["초기화"], 
                self.embeddings
            )
            return
        
        print(f"발견된 문서 파일 수: {len(txt_files)}")
        
        # 문서 로딩
        documents = []
        for file_path in txt_files:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
                print(f"로드됨: {file_path}")
            except Exception as e:
                print(f"문서 로딩 실패 {file_path}: {e}")
        
        if not documents:
            print("로드된 문서가 없습니다.")
            return
        
        # 문서 분할
        print("문서 분할 중...")
        split_docs = self.text_splitter.split_documents(documents)
        print(f"분할된 청크 수: {len(split_docs)}")
        
        # 벡터 스토어 생성
        print("벡터 스토어 생성 중...")
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        
        # 벡터 스토어 저장
        print("벡터 스토어 저장 중...")
        self.vector_store.save_local(self.vector_store_path)
        print("벡터 스토어 생성 완료!")
    
    def similarity_search(self, query: str, k: int = 4) -> List:
        """유사도 검색 수행"""
        if self.vector_store is None:
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"유사도 검색 오류: {e}")
            return []
    
    def get_document_count(self) -> int:
        """벡터 스토어의 문서 수 반환"""
        if self.vector_store is None:
            return 0
        return self.vector_store.index.ntotal 
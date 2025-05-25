import argparse
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.settings import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.storage import StorageContext
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from pydantic import Field
import numpy as np
import chromadb

# ===== 定数 =====
OLLAMA_MODEL = "llama3.1:8b"
CHROMA_DIR = "chroma_store"
HISTORY_DB = "./docs/history.db"
PDFDB_DB = "./pdfdb/documents.db"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 20

# ===== LLM / Embedding 設定 =====
llm = Ollama(model=OLLAMA_MODEL, request_timeout=120.0)
embed_model = OllamaEmbedding(model_name=OLLAMA_MODEL, request_timeout=60.0)
Settings.llm = llm
Settings.embed_model = embed_model

# ===== Chroma の設定 =====
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection("rag_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ===== プロンプトテンプレート =====
qa_template = PromptTemplate(
    """\
文脈情報は以下の通りです:
---------------------
{context_str}
---------------------

この文脈情報をもとに、以下の質問に日本語で答えてください。
質問: {query_str}

回答: """
)

# ===== Softmax類似度フィルター =====
class MySoftmaxPostprocessor(BaseNodePostprocessor):
    temperature: float = Field(default=1.0)
    top_k: int = Field(default=None)
    similarity_cutoff: float = Field(default=None)

    def _postprocess_nodes(self, nodes_with_scores, query_str=None):
        if not nodes_with_scores:
            return []

        scores = np.array([n.score for n in nodes_with_scores])
        scores = scores / self.temperature
        softmax_scores = np.exp(scores) / np.exp(scores).sum()

        for node, score in zip(nodes_with_scores, softmax_scores):
            node.score = score

        if self.similarity_cutoff is not None:
            nodes_with_scores = [n for n in nodes_with_scores if n.score >= self.similarity_cutoff]

        if self.top_k is not None:
            nodes_with_scores = sorted(nodes_with_scores, key=lambda n: n.score, reverse=True)
            nodes_with_scores = nodes_with_scores[:self.top_k]

        return nodes_with_scores

# ===== PDF用: documentsテーブル初期化 =====
def init_pdfdb(path=PDFDB_DB):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            full_text TEXT,
            summary TEXT,
            created_at TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# ===== 履歴取得 =====
def get_history_documents(path, days=7, limit=50):
    if not os.path.exists(path):
        print(f"⚠️ 履歴DBが存在しません: {path}")
        return []

    chrome_epoch = datetime(1601, 1, 1)
    cutoff = datetime.utcnow() - timedelta(days=days)
    cutoff_microseconds = int((cutoff - chrome_epoch).total_seconds() * 1_000_000)

    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("SELECT title, url, visit_time FROM history ORDER BY visit_time DESC LIMIT ?", (limit,))
    docs = [Document(text=f"{title or '(no title)'}\n{url}") for title, url, _ in cursor.fetchall()]
    conn.close()
    return docs

# ===== PDF取得 =====
def get_pdf_documents(path, limit=50):
    if not os.path.exists(path):
        print(f"⚠️ PDF DBが存在しません: {path}")
        return []

    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("SELECT title, full_text, summary, created_at FROM documents ORDER BY created_at DESC LIMIT ?", (limit,))
    docs = [
        Document(text=f"{title}\n{summary}\n\n{full_text}")
        for title, full_text, summary, _ in cursor.fetchall()
    ]
    conn.close()
    return docs

# ===== メイン処理 =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["history", "pdf"], required=True, help="読み込み元を選択（history or pdf）")
    args = parser.parse_args()

    if args.source == "pdf":
        init_pdfdb()
        docs = get_pdf_documents(PDFDB_DB)

    else:
        docs = get_history_documents(HISTORY_DB)

    if not docs:
        print("⚠️ ドキュメントが見つかりません。終了します。")
        return

    parser_ = SimpleNodeParser.from_defaults(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = parser_.get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    postprocessor = MySoftmaxPostprocessor(temperature=0.7, top_k=5)
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        response_synthesizer=get_response_synthesizer(
            response_mode="tree_summarize",
            text_qa_template=qa_template,
        ),
        node_postprocessors=[postprocessor],
    )

    while True:
        query = input("\n🔍 質問を入力してください（qで終了）: ")
        if query.strip().lower() == "q":
            break
        print("回答生成中...")
        response = query_engine.query(query)
        print(f"\n🧠 回答:\n{response}")

# ===== 実行 =====
if __name__ == "__main__":
    main()



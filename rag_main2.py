import os
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document
from llama_index.core import get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.storage import StorageContext
import chromadb
import sqlite3
from datetime import datetime, timedelta

# === 設定 ===
OLLAMA_MODEL = "llama3.1:8b"
CHROMA_DIR = "chroma_store"
HISTORY_DB = "./docs/history.db"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 20

# LLM / Embedding 設定
llm = Ollama(model=OLLAMA_MODEL, request_timeout=120.0)
embed_model = OllamaEmbedding(model_name=OLLAMA_MODEL, request_timeout=60.0)
Settings.llm = llm
Settings.embed_model = embed_model

# Chroma の設定
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection("rag_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# プロンプトテンプレート
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

# 履歴からドキュメント抽出
def get_history_documents(path=HISTORY_DB, days=7, limit=50):
    chrome_epoch = datetime(1601, 1, 1)
    cutoff = datetime.utcnow() - timedelta(days=days)
    cutoff_microseconds = int((cutoff - chrome_epoch).total_seconds() * 1_000_000)

    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT title, url, visit_time FROM history
        ORDER BY visit_time DESC LIMIT ?
    """, (limit,))
    docs = [
        Document(text=f"{title or '(no title)'}\n{url}")
        for title, url, _ in cursor.fetchall()
    ]
    conn.close()
    return docs

# ドキュメント準備 → インデックス作成
docs = get_history_documents()
if not docs:
    print("履歴ドキュメントが見つかりません。終了します。")
    exit()

parser = SimpleNodeParser.from_defaults(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
nodes = parser.get_nodes_from_documents(docs)
index = VectorStoreIndex(nodes, storage_context=storage_context)

# クエリエンジン作成
query_engine = index.as_query_engine(
    similarity_top_k=10,
    response_synthesizer=get_response_synthesizer(
        response_mode="tree_summarize",
        text_qa_template=qa_template,
    ),
    # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

# 対話
while True:
    query = input("\n🔍 質問を入力してください（qで終了）: ")
    if query.strip().lower() == "q":
        break
    print("回答生成中...")
    response = query_engine.query(query)
    print(f"\n🧠 回答:\n{response}")

import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
import chromadb
from llama_index.core import get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
import time
from datetime import datetime, timedelta
import sqlite3
from llama_index.core.schema import Document
import requests
from bs4 import BeautifulSoup
# === 1. 環境設定 ===

OLLAMA_MODEL = "7shi/tanuki-dpo-v1.0"
DOCS_DIR = "docs"
CHROMA_DIR = "chroma_store"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 20


Path(DOCS_DIR).mkdir(exist_ok=True)

def get_page_content(url: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"[WARN] ページ取得失敗: {url} ({e})")
        return ""

def get_history_documents_from_txt(file_path: str = "docs/history.txt") -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    entries = content.strip().split("\n\n")
    return [Document(text=entry) for entry in entries]

# モデルの初期化（タイムアウト設定を追加）
llm = Ollama(model=OLLAMA_MODEL, request_timeout=120.0)
embed_model = OllamaEmbedding(
    model_name=OLLAMA_MODEL,
    request_timeout=60.0,
    embed_batch_size=4,
)

# グローバル設定
Settings.llm = llm
Settings.embed_model = embed_model

# ChromaDBの設定
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection("rag_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

qa_template = PromptTemplate(
    """\
文脈情報は以下の通りです:
---------------------
{context_str}
---------------------

この文脈情報をもとに、以下の質問に日本語で答えてください。
質問内容と関係のない情報は含めないでください。
もし文脈情報から回答できない場合は「その情報は文脈から見つかりません」と答えてください。

質問: {query_str}

回答: """
)

print(f"[DEBUG] 接続中のDBファイル: {os.path.abspath('./docs/history.db')}")
def get_history_documents_from_sqlite(
    path="./docs/history.db",
    days=7,
    limit=100,
    url_filter_keyword=None  # 特定文字列でURLをフィルタ
):
    chrome_epoch = datetime(1601, 1, 1)
    cutoff = datetime.utcnow() - timedelta(days=days)
    cutoff_microseconds = int((cutoff - chrome_epoch).total_seconds() * 1_000_000)

    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    # cursor.execute("""
    #     SELECT url, title FROM urls
    #     WHERE last_visit_time > ?
    #     ORDER BY last_visit_time DESC
    #     LIMIT ?
    # """, (cutoff_microseconds, limit))
    print(f"[DEBUG] 履歴データを取得中...")
    cursor.execute("""
        SELECT title, url, visit_time FROM history
        ORDER BY visit_time DESC LIMIT ?
        """, (limit,))
    print(f"[DEBUG] 履歴データを取得中... 完了")

    docs = []
    for title, url, visit_time in cursor.fetchall():
        if url_filter_keyword and url_filter_keyword not in get_page_content(url):
            continue

        text = f"{title or '(no title)'}\n{url}"
        docs.append(Document(text=text))

    conn.close()
    return docs

# ドキュメント読み込みとノード作成
if len(os.listdir(DOCS_DIR)) == 0:
    print(f"警告: {DOCS_DIR} ディレクトリにドキュメントがありません。サンプルファイルを作成します。")
    with open(f"{DOCS_DIR}/sample.txt", "w") as f:
        f.write("これはサンプルドキュメントです。RAGシステムのテスト用に作成されました。")

print("ブラウザ履歴からドキュメントを作成中...")
history_docs = get_history_documents_from_sqlite(
      path="./docs/history.db",
      days=7,
      limit=100,
      url_filter_keyword="機械学習"
    )

parser = SimpleNodeParser.from_defaults(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
nodes = parser.get_nodes_from_documents(history_docs)
if not history_docs:
    print("条件に一致する履歴ドキュメントが見つかりません。処理を終了します。")
    exit()


# インデックス作成
try:
    print("インデックスを作成中...")
    start_time = time.time()

    # バッチ処理でインデックスを作成
    batch_size = 10
    for i in range(0, len(nodes), batch_size):
        print(f"バッチ処理中: {i}/{len(nodes)} ノード")
        current_batch = nodes[i:i+batch_size]
        if i == 0:
            index = VectorStoreIndex(current_batch, storage_context=storage_context)
        else:
            index.insert_nodes(current_batch)

    end_time = time.time()
    print(f"インデックスの作成が完了しました (所要時間: {end_time - start_time:.2f}秒)")
    
    # レスポンス合成器の設定
    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=qa_template,
    )

    # レスポンス合成器の設定
    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=qa_template,
    )

    # クエリエンジン作成
    query_engine = index.as_query_engine(
        similarity_top_k=2,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    )

    # 対話ループ
    while True:
        question = input("\n🔍 質問を入力してください（終了するには q を入力）: ")
        if question.strip().lower() == "q":
            break
            
        print("回答を生成中...")
        try:
            start_time = time.time()
            response = query_engine.query(question)
            end_time = time.time()
            print(f"\n🧠 回答:\n{response}")
            print(f"(所要時間: {end_time - start_time:.2f}秒)")
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            print("もう一度質問してみてください。")

except Exception as e:
    print(f"インデックス作成中にエラーが発生しました: {e}")

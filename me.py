import json
from pathlib import Path
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.settings import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage import StorageContext
import chromadb

# 設定
OLLAMA_MODEL = "llama3.1:8b"
CHROMA_DIR = "./chroma_store"
JSON_PATH = "ttt.json"  # ここに対象のjsonファイルを置く

# ① Ollama EmbeddingとLLMの設定
chroma_client = chromadb.PersistentClient(path="chroma_store") 
collection = chroma_client.get_or_create_collection("json_collection")

Settings.embed_model = OllamaEmbedding(model_name=OLLAMA_MODEL)
Settings.llm = Ollama(model=OLLAMA_MODEL)

# ② Chromaに接続
# ③ ChromaVectorStoreを作成
chroma_store = ChromaVectorStore(
    client=chroma_client,   # ✅ ここ「client」です
        chroma_collection=collection,
    collection_name="json_collection",
)

storage_context = StorageContext.from_defaults(vector_store=chroma_store)
storage_context = StorageContext.from_defaults(vector_store=chroma_store)

# ③ JSONファイルを読み込む
def load_json_documents(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        # JSONの各アイテムをテキスト化してDocumentに変換
        content = f"Type: {item['type']}\nPath: {item['path']}\nNaming: {item['naming_convention']}"
        doc = Document(text=content)
        documents.append(doc)

    return documents

documents = load_json_documents(JSON_PATH)

# ④ Nodeパーサでチャンク化（簡易的に）
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# ⑤ インデックス作成（ベクトルストアに登録）
index = VectorStoreIndex(nodes, storage_context=storage_context)

# 保存
index.storage_context.persist()

print("✅ JSONデータをChromaに保存しました。")


query_engine = index.as_query_engine()
response = query_engine.query("どんなフォルダ構成ですか？")
print(response)

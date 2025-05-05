import os, sqlite3, torch
from InstructorEmbedding import INSTRUCTOR
from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFacePipeline
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# 1. Embeddingモデルロード
# embedder = INSTRUCTOR('hkunlp/instructor-large')
# instruction = "Represent clinical guideline document for retrieval"
instruction = "Represent clinical guideline document for retrieval"

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db_path = "docs/db_faiss"
history_db_file = "docs/history2.db"

try:
    conn = sqlite3.connect(history_db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM history")
    rows = cursor.fetchall()
    conn.close()
    print("SQLiteデータベースからデータを正常に取得しました。")

except sqlite3.Error as e:
    print(f"SQLiteデータベースへの接続に失敗しました: {e}")
    exit(1)

texts = [row[0] for row in rows if row[0]] 
docs = [Document(page_content=text) for text in texts]

model_name = "rinna/japanese-gpt2-medium"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokens = tokenizer("私はAIが好きです。", return_tensors="pt")
    outputs = model(**tokens, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1] 
    sentence_embedding = hidden_states.mean(dim=1)
    print(sentence_embedding.shape) 

    print("モデルのロードが正常に完了しました。")
except Exception as e:
    print(f"モデルのロードに失敗しました: {e}")
    exit(1)

text_embeddings = embedding.embed_documents([doc.page_content for doc in docs])

try:
    if os.path.exists(db_path):
        print("既存のDBをロードしています...")
        db = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
    else:
        print("新しいDBを作成しています...")
        db = FAISS.from_texts(
            texts=[doc.page_content for doc in docs],
            embedding=embedding
        )
        db.save_local(db_path)
        print("DB保存完了")
except Exception as e:
    print(f"FAISSデータベースの作成またはロードに失敗しました: {e}")
    exit(1)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,  # ここに設定！
    device=0 if torch.cuda.is_available() else -1,
)

# 4. LLMロード
llm = HuggingFacePipeline(pipeline=pipe)

# 5. RAGパイプライン作成
try:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type="stuff",
    )
    print("RAGパイプラインが正常に作成されました。")
except Exception as e:
    print(f"RAGパイプラインの作成に失敗しました: {e}")
    exit(1)

print("RAGチャットを開始します。終了するには 'exit' を入力してください。")
while True:
    try:
        user_query = input("\nあなた: ")
        if user_query.lower() in ("exit", "quit"):
            print("チャットを終了します。")
            break

        # Retrieve ＋ 回答生成
        result = qa.run(user_query)

        # 出力
        print(f"RAGアシスタント: {result}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

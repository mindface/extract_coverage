import os
import sqlite3, fitz, argparse
from datetime import datetime
from llama_index.llms.ollama import Ollama

PDF_DIR = "./pdfs"
HISTORY_PDF_DIR = "./pdfs"
DB_PATH = "./docs/"
HISTORY_DB = "./dosc/history.db"
DB_PATH = "./pdfdb/documents.db"
OLLAMA_MODEL = "llama3.1:8b"

def extract_text_with_fitz(filepath):
    try:
        with open(filepath, "rb") as pdf_file:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except Exception as e:
        print(f"[!] エラー（{filepath}）: {e}")
        return ""

def summarize(text):
    if not text.strip():
        return "(本文なし)"
    prompt = f"次の文書を日本語で簡潔に要約してください:\n{text[:3000]}"
    llm = Ollama(model=OLLAMA_MODEL)
    return llm.complete(prompt).text.strip()

def initialize_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            full_text TEXT,
            summary TEXT,
            created_at DATETIME
        )
    ''')
    conn.commit()
    return conn

def init_historydb(path=HISTORY_DB):
    """history.db に history テーブルがなければ作成する"""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT,
            visit_time INTEGER
        )
    ''')
    conn.commit()
    return conn

def initialize_history_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            full_text TEXT,
            summary TEXT,
            created_at DATETIME
        )
    ''')
    conn.commit()
    return conn

def makeDocumentDB():
    conn = initialize_db()
    for filename in os.listdir(PDF_DIR):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(PDF_DIR, filename)
        print(f"[+] 処理中: {filename}")

        try:
            full_text = extract_text_with_fitz(filepath)
            if not full_text.strip():
                print(f"[!] 空のPDFスキップ: {filename}")
                continue
        except Exception as e:
            print(f"[❌] PDF抽出失敗: {filename} - {e}")
            continue

        try:
            summary = summarize(full_text)
            print(f"[✓] 要約完了: {filename}")
        except Exception as e:
            print(f"[❌] 要約失敗: {filename} - {e}")
            continue

        try:
            conn.execute('''
                INSERT INTO documents (title, full_text, summary, created_at)
                VALUES (?, ?, ?, ?)
            ''', (filename, full_text, summary, datetime.now()))
            conn.commit()
        except Exception as e:
            print(f"[❌] DB書き込み失敗: {filename} - {e}")
            continue

    conn.close()
    print("✅ 全PDF処理完了")

# TODO 要調整
def makeHstoryDb():
    conn = initialize_history_db()
    for filename in os.listdir(PDF_DIR):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(PDF_DIR, filename)
        print(f"[+] 処理中: {filename}")
        full_text = extract_text_with_fitz(filepath)
        if not full_text.strip():
            print(f"[!] 空のPDFスキップ: {filename}")
            continue

        summary = summarize(full_text)
        print(f"[✓] 要約完了: {filename}")

        conn.execute('''
            INSERT INTO documents (title, full_text, summary, created_at)
            VALUES (?, ?, ?, ?)
        ''', (filename, full_text, summary, datetime.now()))
        conn.commit()

    conn.close()
    print("✅ 全PDF処理完了")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["history", "pdf"], required=True, help="読み込み元を選択（history or pdf）")
    args = parser.parse_args()

    if args.source == "pdf":
        docs = makeDocumentDB()
    else:
        docs = makeHstoryDb()

    if not docs:
        print("⚠️ ドキュメントが見つかりません。終了します。")
        return

    # ここで docs を使ってインデックス化や検索処理に進める
    for doc in docs[:2]:  # サンプル出力（デバッグ用）
        print(f"\n📄 {doc.text[:200]}...\n")


if __name__ == "__main__":
    main()
import os
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import sys, platform, requests, time, random, json
from bs4 import BeautifulSoup
# from InstructorEmbedding import INSTRUCTOR
# from sentence_transformers import SentenceTransformer

HISTORY_TXT_FILE = "docs/history.txt"
HISTORY_DB_FILE = "docs/history2.db"
Path("docs").mkdir(exist_ok=True)
skip_urls = [
    "localhost",
    "https://gemini.google.com",
    "https://www.openwork.jp"
    "https://www.google.com",
    "https://translate.google.com",
    "https://www.google.com",
    "https://www.yahoo.co.jp",
    "https://www.amazon.co.jp",
    "https://www.apple.com",
    "https://www.microsoft.com",
    "https://www.facebook.com",
    "https://www.twitter.com",
    "https://www.instagram.com",
    "https://www.youtube.com",
    "https://www.linkedin.com",
    "https://www.pinterest.com",
    "https://www.reddit.com",
    "https://www.tumblr.com",
    "https://www.wikipedia.org",
    "https://www.quora.com",
    "https://www.slideshare.net",
    "https://www.flickr.com",
    "https://www.vimeo.com",
    "https://www.soundcloud.com",
    "https://www.mixcloud.com",
    "https://www.bandcamp.com",
    "https://www.soundcloud.com",
    "https://www.mixcloud.com",
    "https://www.bandcamp.com",
    "https://www.soundcloud.com",
    "https://www.mixcloud.com",
]
required_keywords = ["身体構造", "検出", "人体構成","","生活習慣"]

def get_page_details(url: str) -> dict:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # メタデータの抽出
        meta_description = soup.find('meta', {'name': 'description'})
        description = meta_description['content'] if meta_description else ''
        
        # 本文テキストの抽出（HTMLタグを除去）
        # 不要な要素を削除
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()

        main_content = soup.get_text(separator='\n', strip=True)

        return {
            'description': description,
            'content': main_content,
            'last_updated': datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"[⚠️] {url} のコンテンツ取得に失敗: {e}")
        return {
            'description': '',
            'content': '',
            'last_updated': datetime.utcnow().isoformat()
        }

def page_contains_keyword(url: str, keyword: str) -> bool:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # メタデータの抽出
        meta_description = soup.find('meta', {'name': 'description'})
        description = meta_description['content'] if meta_description else ''
        
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()

        main_content = soup.get_text(separator='\n', strip=True)

        return {
            'description': description,
            'content': main_content,
            'last_updated': datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"[⚠️] {url} のコンテンツ取得に失敗: {e}")
        return {
            'description': 'loss action',
            'content': 'loss action',
            'last_updated': datetime.utcnow().isoformat()
        }

def get_chrome_history_path():
    system = platform.system()
    if system == "Windows":
        path = os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data\Default\History")
    elif system == "Darwin":  # macOS
        path = os.path.expanduser("~/Library/Application Support/Google/Chrome/Default/History")
    elif system == "Linux":
        path = os.path.expanduser("~/.config/google-chrome/Default/History")
    else:
        raise RuntimeError("未対応のOSです")
    if not os.path.exists(path):
        raise FileNotFoundError("Chromeの履歴ファイルが見つかりませんでした")
    print(f"[✅] Chromeの履歴ファイル: {path}")
    return path

def collect_chrome_history(limit=100, days=7):
    original_path = get_chrome_history_path()
    temp_copy_path = "temp_history_copy.db"

    shutil.copy2(original_path, temp_copy_path)

    cutoff = datetime.utcnow() - timedelta(days=days)
    chrome_epoch = datetime(1601, 1, 1)
    cutoff_microseconds = int((cutoff - chrome_epoch).total_seconds() * 1_000_000)

    urls = []

    try:
        conn = sqlite3.connect(temp_copy_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT url, title, last_visit_time FROM urls
            WHERE last_visit_time > ?
            ORDER BY last_visit_time DESC
            LIMIT ?
        """, (cutoff_microseconds, limit))

        for url, title, visit_time in cursor.fetchall():
            if any(skip_url in url for skip_url in skip_urls):
                continue
            if not any(keyword in url or keyword in title for keyword in required_keywords):
                continue
            urls.append((title or "(no title)", url, visit_time))
            time.sleep(random.uniform(1, 3))
    finally:
        conn.close()
        os.remove(temp_copy_path)

    return urls

def save_history_to_file(history: list[tuple[str, str, int]], path: str = HISTORY_TXT_FILE):
    with open(path, "w", encoding="utf-8") as f:
        for title, url, _ in history:
            if any(skip_url in url for skip_url in skip_urls):
                continue
            f.write(f"{title}\n{url}\n\n")
    print(f"[✅] 履歴をテキストに保存しました: {path}")

def create_history_db(path: str = HISTORY_DB_FILE):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    # テーブルの作成
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT,
            visit_time INTEGER,
            description TEXT,
            content TEXT,
            last_updated TEXT,
            embedding_vector BLOB,
            tags TEXT,
            category TEXT
        )
    """)

    # インデックスの作成
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_visit_time ON history(visit_time)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_url ON history(url)
    """)
    
    conn.commit()
    conn.close()

def load_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # return INSTRUCTOR('hkunlp/instructor-base')

def save_history_to_db(history: list[tuple[str, str, int]], path: str = HISTORY_DB_FILE):
    create_history_db(path)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    
    # model = load_embedding_model()
    # model = SentenceTransformer('all-MiniLM-L6-v2')

    for title, url, visit_time in history:
        if any(skip_url in url for skip_url in skip_urls):
            continue
        if not any(keyword in url or keyword in title for keyword in required_keywords):
            continue

        details = get_page_details(url)

        # ★追加：description+contentからベクトル生成
        full_text = (details['description'] or '') + '\n' + (details['content'] or '')
        embedding_vector = [0.0] * 768

        cursor.execute("""
            INSERT INTO history (
                title, 
                url, 
                visit_time, 
                description, 
                content,
                last_updated,
                embedding_vector,
                tags,
                category
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            title,
            url,
            visit_time,
            details['description'],
            details['content'],
            details['last_updated'],
            json.dumps(embedding_vector),
            '',
            ''
        ))
        
        if random.random() < 0.1:
            conn.commit()
    
    conn.commit()
    conn.close()
    print(f"[✅] 履歴＋埋め込みをデータベースに保存しました: {path}")

if __name__ == "__main__":
    try:
        history = collect_chrome_history()
        save_history_to_file(history)
        save_history_to_db(history)
    except Exception as e:
        print(f"[❌] エラー: {e}")
        sys.exit(1)

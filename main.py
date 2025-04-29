# main.py
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import MeCab
import regex as re
import os

# 日本語用のリソースをダウンロード（初回のみ必要）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DocumentSimilarityAnalyzer:
    def __init__(self, use_japanese=True):
        """
        文書類似度とカバレッジを分析するためのクラス
        
        Parameters:
        -----------
        use_japanese : bool
            日本語処理を行うかどうか（TrueならMeCabを使用、FalseならNLTKを使用）
        """
        self.use_japanese = use_japanese
        if use_japanese:
            self.mecab = MeCab.Tagger("-Owakati")
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenize,token_pattern=None)
        self.tfidf_matrix = None
        self.vocab = None
        self.document_collection = []
        self.category_map = {}  # 単語とカテゴリのマッピング
        
    def tokenize_and_preprocess(self, text):
        """
        テキストのトークン化と前処理を行う
        """
        if self.use_japanese:
            # MeCabを使用した日本語の分かち書き
            tokens = self.mecab.parse(text).strip().split()
            # 記号、数字、一文字の助詞などのフィルタリング
            tokens = [t for t in tokens if re.match(r'[^\p{P}\d]', t) and len(t) > 1]
        else:
            tokens = nltk.word_tokenize(text.lower())
            try:
                stop_words = set(nltk.corpus.stopwords.words('english'))
            except LookupError:
                nltk.download('stopwords')
                stop_words = set(nltk.corpus.stopwords.words('english'))

            tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        
        return tokens
    
    def tokenize(self, text):
        """
        TfidfVectorizerで使用するトークナイザー関数
        """
        return self.tokenize_and_preprocess(text)
    
    def add_documents(self, documents):
        """
        分析対象の文書を追加し、TF-IDF行列を計算する
        """
        self.document_collection.extend(documents)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.document_collection)
        self.vocab = self.tfidf_vectorizer.get_feature_names_out()
    
    def calculate_importance(self, tokens):
        """
        トークンの重要度（TF-IDF値）を計算する
        """
        if self.tfidf_matrix is None:
            raise ValueError("先にadd_documentsメソッドを呼び出してください。")
        
        importance = {}
        for token in tokens:
            if token in self.vocab:
                idx = np.where(self.vocab == token)[0]
                if len(idx) > 0:
                    # TF-IDF行列の平均値を取得
                    importance[token] = np.mean(self.tfidf_matrix[:, idx[0]].toarray())
            else:
                importance[token] = 0.1  # 未知語のデフォルト重要度
                
        return importance
    
    def set_category_map(self, category_map):
        """
        単語とカテゴリのマッピングを設定する
        
        Parameters:
        -----------
        category_map : dict
            単語とカテゴリのマッピング辞書
            例: {"犬": "動物", "猫": "動物", "私": "人物", "彼": "人物"}
        """
        self.category_map = category_map
    
    def extract_coverage(self, doc1, doc2):
        """
        2つの文書間のカバレッジを抽出する
        
        Parameters:
        -----------
        doc1 : str
            1つ目の文書
        doc2 : str
            2つ目の文書
            
        Returns:
        --------
        final_coverage : float
            総合カバレッジスコア（0-1の範囲）
        coverage_metrics : dict
            詳細なカバレッジ指標
        """
        # 1. 単語分割と前処理
        tokens1 = self.tokenize_and_preprocess(doc1)
        tokens2 = self.tokenize_and_preprocess(doc2)
        
        # 単語の集合を作成
        set1 = set(tokens1)
        set2 = set(tokens2)
        common_words = set1.intersection(set2)
        all_words = set1.union(set2)
        
        # 2. 基本的なジャッカード係数
        jaccard = len(common_words) / len(all_words) if len(all_words) > 0 else 0
        
        # 3. 単語の重要度計算
        # 文書がまだ追加されていない場合は追加する
        if len(self.document_collection) == 0:
            self.add_documents([doc1, doc2])
        
        word_importance = self.calculate_importance(list(all_words))
        
        # 4. 重要度に基づく共通単語のカバレッジスコア計算
        common_importance = sum(word_importance.get(word, 0) for word in common_words)
        max_importance = sum(word_importance.get(word, 0) for word in all_words)
        weighted_coverage = common_importance / max_importance if max_importance > 0 else 0
        
        # 5. カテゴリカルカバレッジ（カテゴリマップが設定されている場合）
        categorical_coverage = {}
        if self.category_map:
            categories = defaultdict(list)
            
            # 単語をカテゴリ別に分類
            for word in all_words:
                category = self.category_map.get(word, "その他")
                categories[category].append(word)
            
            # カテゴリごとのカバレッジを計算
            for category, words in categories.items():
                cat_common = set(words).intersection(common_words)
                cat_coverage = len(cat_common) / len(words) if len(words) > 0 else 0
                categorical_coverage[category] = cat_coverage

        # 6. カバレッジ指標の計算
        coverage_metrics = {
            "raw_jaccard": jaccard,
            "weighted_coverage": weighted_coverage,
            "common_word_count": len(common_words),
            "common_words": list(common_words),
            "common_importance": common_importance,
            "max_importance": max_importance,
            "coverage_ratio": weighted_coverage / jaccard if jaccard > 0 else 0,
            "categorical_coverage": categorical_coverage
        }
        
        # 7. 総合カバレッジスコア
        alpha = 0.4  # ジャッカード係数の重み
        beta = 0.6   # 重要度ベースカバレッジの重み
        final_coverage = alpha * jaccard + beta * weighted_coverage
        
        return final_coverage, coverage_metrics
    
    def extract_coverage_from_jaccard(self, doc1, doc2):
        """
        ジャッカード係数からカバレッジを抽出する簡易版
        """
        tokens1 = self.tokenize_and_preprocess(doc1)
        tokens2 = self.tokenize_and_preprocess(doc2)
        
        set1 = set(tokens1)
        set2 = set(tokens2)
        common_words = set1.intersection(set2)
        all_words = set1.union(set2)
        
        jaccard = len(common_words) / len(all_words) if len(all_words) > 0 else 0
        
        # 共通単語の割合をカバレッジとする
        coverage = {
            "jaccard": jaccard,
            "common_words": list(common_words),
            "unique_words_doc1": list(set1 - set2),
            "unique_words_doc2": list(set2 - set1)
        }
        
        return jaccard, coverage
    
    def keyword_based_coverage(self, doc1, doc2, keywords):
        """
        特定のキーワードに基づくカバレッジを計算
        
        Parameters:
        -----------
        doc1, doc2 : str
            比較する2つの文書
        keywords : dict
            重要キーワードとその重要度のマッピング
            例: {"犬": 0.8, "猫": 0.7, "好き": 0.6}
        """
        tokens1 = set(self.tokenize_and_preprocess(doc1))
        tokens2 = set(self.tokenize_and_preprocess(doc2))
        
        # 各文書で見つかったキーワード
        found_in_doc1 = tokens1.intersection(keywords.keys())
        found_in_doc2 = tokens2.intersection(keywords.keys())
        
        # 共通のキーワードと全キーワード
        common_keywords = found_in_doc1.intersection(found_in_doc2)
        all_found_keywords = found_in_doc1.union(found_in_doc2)
        
        # 単純なジャッカードベースのキーワードカバレッジ
        if len(all_found_keywords) > 0:
            simple_coverage = len(common_keywords) / len(all_found_keywords)
        else:
            simple_coverage = 0
        
        # 重み付きカバレッジ
        if all_found_keywords:
            common_weight = sum(keywords.get(k, 0) for k in common_keywords)
            total_weight = sum(keywords.get(k, 0) for k in all_found_keywords)
            weighted_coverage = common_weight / total_weight if total_weight > 0 else 0
        else:
            weighted_coverage = 0
        
        # カバレッジスコア
        coverage_score = 0.4 * simple_coverage + 0.6 * weighted_coverage
        
        return coverage_score, {
            "simple_coverage": simple_coverage,
            "weighted_coverage": weighted_coverage,
            "common_keywords": list(common_keywords),
            "all_keywords": list(all_found_keywords)
        }


def main():
    # 使用例
    analyzer = DocumentSimilarityAnalyzer(use_japanese=True)

    # サンプル文書
    doc1 = "私は犬が好きです"
    doc2 = "彼は猫が好きです"
    
    # カテゴリマップの設定（オプション）
    category_map = {
        "私": "主語", "彼": "主語",
        "犬": "目的語", "猫": "目的語",
        "好き": "述語", "です": "助動詞"
    }
    analyzer.set_category_map(category_map)
    
    # 文書を追加
    analyzer.add_documents([doc1, doc2])
    
    # カバレッジ抽出
    coverage, metrics = analyzer.extract_coverage(doc1, doc2)
    
    # 結果表示
    print("文書1:", doc1)
    print("文書2:", doc2)
    print("\n基本情報:")
    print(f"ジャッカード係数: {metrics['raw_jaccard']:.4f}")
    print(f"重み付きカバレッジ: {metrics['weighted_coverage']:.4f}")
    print(f"総合カバレッジスコア: {coverage:.4f}")
    
    print("\n共通単語:")
    for word in metrics['common_words']:
        print(f"- {word}")
    
    print("\nカテゴリ別カバレッジ:")
    for category, score in metrics['categorical_coverage'].items():
        print(f"- {category}: {score:.4f}")
    
    # キーワードベースのカバレッジ（オプション）
    keywords = {"犬": 0.8, "猫": 0.7, "好き": 0.6, "です": 0.2, "は": 0.1}
    keyword_coverage, keyword_metrics = analyzer.keyword_based_coverage(doc1, doc2, keywords)
    
    print("\nキーワードベースのカバレッジ:")
    print(f"スコア: {keyword_coverage:.4f}")
    print(f"共通キーワード: {', '.join(keyword_metrics['common_keywords'])}")


if __name__ == "__main__":
    main()
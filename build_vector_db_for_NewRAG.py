"""
build_vector_db_for_NewRAG.py
── BOOK Step 1-2: NewRAG 最小構成パイプライン

[FILE] build_vector_db_for_NewRAG.py
NOTE: このファイルは Phase 1（NewRAG/ピュア構築）専用の最小構成版です。
      運用フェーズ（MixedRAG/除霊編）は build_vector_db_for_MixedRAG.py を参照してください。

設計方針:
  - 設定1行（DOCS_ROOT）で動作する極限のシンプルさ
  - extract_section を廃止し、Frontmatter の summary を唯一の正とする
  - 連結順序: Title → Tags → Meta → Summary(Frontmatter) → Body
  - ナレッジマップはタグ判定で knowledge_meta コレクションへルーティング
  - requires/enables は vector_text には含めないが、Firestore フィールドとして保存する
"""

import os
import sys
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from sentence_transformers import SentenceTransformer
import frontmatter

# ==========================================
# ⚙️ 設定（ここだけ書き換えれば動く）
# ==========================================
DOCS_ROOT = "./docs"

# ==========================================
# 🔒 パイプライン定数
# ==========================================
# ベクトル化除外タグ（spec_knowledge_map_routing.md 準拠）
SKIP_VECTORIZATION_TAGS = ["ナレッジマップ"]

# vector_text から除外するキー（frontmatter_spec_socrates.yml §2 準拠）
# requires/enables はナレッジグラフ用であり、128トークン予算を圧迫するため除外
EXCLUDE_FROM_VECTOR = [
    "title", "tags", "summary", "date", "url",
    "requires", "enables",
]

# ==========================================
# 🚀 初期化
# ==========================================
key_path = "serviceAccountKey.json"
if not os.path.exists(key_path):
    print(f"❌ Error: {key_path} が見つかりません。")
    print("   Firebase Console からダウンロードし、このスクリプトと同じフォルダに配置してください。")
    sys.exit(1)

if not firebase_admin._apps:
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()
print("⏳ Embeddingモデルをロード中...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# ==========================================
# 🎯 メイン処理
# ==========================================
def process_new_rag():
    print(f"\n🚀 NewRAG パイプライン開始 (Target: {DOCS_ROOT})\n")

    if not os.path.isdir(DOCS_ROOT):
        print(f"❌ Error: フォルダ '{DOCS_ROOT}' が見つかりません。")
        sys.exit(1)

    success_count = 0
    error_count = 0
    map_count = 0

    for root, _, files in os.walk(DOCS_ROOT):
        md_files = sorted(f for f in files if f.endswith(".md"))

        for filename in md_files:
            file_path = os.path.join(root, filename)

            # --- 0. ファイル読み込み ---
            try:
                post = frontmatter.load(file_path)
            except Exception as e:
                print(f"❌ [{filename}] パースエラー: {e}")
                error_count += 1
                continue

            meta = post.metadata
            body = post.content.strip()
            doc_id = os.path.splitext(filename)[0]

            # タイトル補完
            title = meta.get("title", doc_id)

            # タグのサニタイズ（辞書混入対策）
            raw_tags = meta.get("tags", [])
            if not isinstance(raw_tags, list):
                raw_tags = []
            tags = [str(t) for t in raw_tags if t is not None]

            # --- 1. ルーティング判定 (spec_knowledge_map_routing.md 準拠) ---
            if any(tag in SKIP_VECTORIZATION_TAGS for tag in tags):
                try:
                    ref = db.collection("knowledge_meta").document(doc_id)
                    data = {
                        "title": doc_id,
                        "display_title": title,
                        "category": meta.get("category", ""),
                        "tags": tags,
                        "content": body,
                        "requires": meta.get("requires", []),
                        "enables": meta.get("enables", []),
                        "level": meta.get("level", ""),
                        "doc_type": "knowledge_map",
                        "updated_at": firestore.SERVER_TIMESTAMP,
                    }
                    data = {k: v for k, v in data.items() if v is not None}
                    ref.set(data, merge=True)
                    print(f"📋 [MAP] {filename} -> メタデータ登録（ベクトル化スキップ）")
                    map_count += 1
                except Exception as e:
                    print(f"❌ [MAP] {filename} -> 登録エラー: {e}")
                    error_count += 1
                continue

            # --- 2. vector_text 構築 ---
            # 連結順序: Title → Tags → Meta → Summary(Frontmatter) → Body
            # MiniLM-L12-v2 は内部で先頭128トークンのみをEmbeddingに使用する。
            # Title + Tags + Meta + Summary がこの窓に収まるよう設計している。
            # Body は窓の外に出るが、Firestore には全文保存されるため
            # LLM が Context として参照する段階では全文が使われる。
            summary = meta.get("summary", "")

            meta_str = "\n".join(
                f"{k}: {v}"
                for k, v in meta.items()
                if k not in EXCLUDE_FROM_VECTOR
                and isinstance(v, (str, int, float))
            )

            vector_text = (
                f"Title: {title}\n"
                f"Tags: {', '.join(tags)}\n"
                f"{meta_str}\n"
                f"{summary}\n\n"
                f"{body}"
            )

            embedding = model.encode(vector_text).tolist()

            # --- 3. Firestore登録 (tech_docs) ---
            try:
                # date フィールドの型変換（date → datetime）
                date_val = meta.get("date")
                if isinstance(date_val, datetime.date) and not isinstance(
                    date_val, datetime.datetime
                ):
                    date_val = datetime.datetime.combine(
                        date_val, datetime.time.min
                    )

                doc_ref = db.collection("tech_docs").document(doc_id)
                doc_data = {
                    "title": doc_id,
                    "display_title": title,
                    "category": meta.get("category", ""),
                    "content": body,
                    "summary": summary,
                    "tags": tags,
                    "level": meta.get("level", ""),
                    "date": date_val,
                    "requires": meta.get("requires", []),
                    "enables": meta.get("enables", []),
                    "embedding": embedding,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                }
                doc_data = {k: v for k, v in doc_data.items() if v is not None}
                doc_ref.set(doc_data, merge=True)

                mark_sum = "○" if summary else "×"
                print(f"✅ [DOC] {filename} -> ベクトル化完了 (要約{mark_sum})")
                success_count += 1

            except Exception as e:
                print(f"❌ [DOC] {filename} -> 登録エラー: {e}")
                error_count += 1

    # --- 4. 診断レポート ---
    print("\n" + "=" * 50)
    print(f"🎉 NewRAG パイプライン完了")
    print(f"   ベクトル化: {success_count} 件")
    print(f"   メタデータ: {map_count} 件")
    print(f"   エラー:     {error_count} 件")
    print("=" * 50)

    if error_count > 0:
        print("\n⚠️  エラーが発生しています。上のログを確認してください。")


if __name__ == "__main__":
    process_new_rag()

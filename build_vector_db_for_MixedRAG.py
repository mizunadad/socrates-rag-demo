"""
build_vector_db_for_MixedRAG.py
── BOOK Step 4: MixedRAG 実戦運用パイプライン

[FILE] build_vector_db_for_MixedRAG.py
NOTE: このファイルは Phase 3（MixedRAG/制御の道）専用です。
      Phase 1（NewRAG/ピュア構築）は build_vector_db_for_NewRAG.py を参照してください。

設計方針:
  - NewRAG のベクトル化ロジックを継承（Frontmatter summary 優先、同一連結順序）
  - カテゴリ・スコープによる亡霊データの自動検知と物理消去（DELETE_FIELD）
  - --dry-run フラグによる安全なシミュレーション機能
  - dry-run 時はリストアップのみ。Firestore への書き込みは一切行わない

除霊の仕組み:
  ローカルに存在するファイルと Firestore 上のドキュメントを、カテゴリ単位で
  差分比較する。Firestore にあるがローカルにないドキュメントを「亡霊」と判定し、
  embedding フィールドのみを DELETE_FIELD で物理削除する。
  ドキュメント自体（title, category, tags 等）は残るが、ベクトル検索の対象からは
  自動的に外れる（streamlit_app.py の次元チェックで弾かれるため）。
"""

import os
import sys
import argparse
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
EXCLUDE_FROM_VECTOR = [
    "title", "tags", "summary", "date", "url",
    "requires", "enables",
]

# ==========================================
# 🚀 初期化と引数解析
# ==========================================
parser = argparse.ArgumentParser(
    description="MixedRAG 同期 & 除霊パイプライン",
    epilog="例: python %(prog)s --dry-run  # まず確認してから実行する",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="除霊候補をリストアップするだけで、Firestore への書き込みは一切行いません",
)
args = parser.parse_args()

key_path = "serviceAccountKey.json"
if not os.path.exists(key_path):
    print(f"❌ Error: {key_path} が見つかりません。")
    print("   Firebase Console からダウンロードし、このスクリプトと同じフォルダに配置してください。")
    sys.exit(1)

if not firebase_admin._apps:
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# dry-run 時は Embedding モデルのロードをスキップ（時間短縮）
if not args.dry_run:
    print("⏳ Embeddingモデルをロード中...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
else:
    model = None


# ==========================================
# 🎯 メイン処理
# ==========================================
def process_mixed_rag():
    mode_label = " (DRY-RUN MODE)" if args.dry_run else ""
    print(f"\n👻 MixedRAG パイプライン開始{mode_label}\n")

    if not os.path.isdir(DOCS_ROOT):
        print(f"❌ Error: フォルダ '{DOCS_ROOT}' が見つかりません。")
        sys.exit(1)

    # ==============================================
    # Pass 1: ローカルファイルの走査とデータ準備
    # ==============================================
    # カテゴリ別のローカル doc_id セット（除霊のスコープ判定用）
    local_ids_by_cat = {}  # {category: set(doc_ids)}
    # 登録対象のドキュメントリスト
    valid_docs = []
    parse_errors = 0

    for root, _, files in os.walk(DOCS_ROOT):
        md_files = sorted(f for f in files if f.endswith(".md"))

        for filename in md_files:
            file_path = os.path.join(root, filename)
            doc_id = os.path.splitext(filename)[0]

            try:
                post = frontmatter.load(file_path)
            except Exception as e:
                print(f"❌ [{filename}] パースエラー: {e}")
                parse_errors += 1
                continue

            meta = post.metadata
            cat = meta.get("category", "")
            raw_tags = meta.get("tags", [])
            if not isinstance(raw_tags, list):
                raw_tags = []
            tags = [str(t) for t in raw_tags if t is not None]

            is_map = any(tag in SKIP_VECTORIZATION_TAGS for tag in tags)

            if not is_map:
                # ベクトル化対象 → カテゴリ別のローカルIDセットに登録
                if cat not in local_ids_by_cat:
                    local_ids_by_cat[cat] = set()
                local_ids_by_cat[cat].add(doc_id)

            valid_docs.append((doc_id, filename, post, cat, tags, is_map))

    print(f"   ローカル文書: {len(valid_docs)} 件（うちナレッジマップ: "
          f"{sum(1 for v in valid_docs if v[5])} 件）")
    if parse_errors:
        print(f"   パースエラー: {parse_errors} 件")

    # ==============================================
    # Pass 2: 除霊フェーズ（Ghost Cleanup）
    # ==============================================
    print("\n🧹 データベースの亡霊（不整合）をチェック中...")
    ghost_total = 0

    for cat, local_ids in local_ids_by_cat.items():
        # Firestore から同カテゴリの全ドキュメントIDを取得
        remote_docs = db.collection("tech_docs").where("category", "==", cat).stream()
        remote_ids = {doc.id for doc in remote_docs}

        # 差分 = 亡霊（Firestore にあるがローカルにない）
        ghosts = remote_ids - local_ids

        for ghost_id in sorted(ghosts):
            ghost_total += 1
            if args.dry_run:
                print(f"   🔍 [DRY-RUN] 👻 亡霊発見: {ghost_id} (category: {cat})")
            else:
                db.collection("tech_docs").document(ghost_id).update({
                    "embedding": firestore.DELETE_FIELD
                })
                print(f"   🔥 [除霊完了] 👻 {ghost_id} の embedding を削除 (category: {cat})")

    if ghost_total == 0:
        print("   ✨ 亡霊は見つかりませんでした。")
    else:
        print(f"   → 亡霊: {ghost_total} 件{'（DRY-RUN: 実行されていません）' if args.dry_run else ''}")

    # dry-run 時はここで終了（Firestore への書き込みは一切行わない）
    if args.dry_run:
        print("\n" + "=" * 50)
        print(f"🏁 DRY-RUN 完了（Firestore への変更はありません）")
        print(f"   亡霊候補: {ghost_total} 件")
        print(f"   → 除霊を実行するには --dry-run を外して再実行してください")
        print("=" * 50)
        return

    # ==============================================
    # Pass 3: 登録・更新フェーズ（Sync）
    # ==============================================
    print(f"\n✨ ドキュメントの同期を開始します（対象: {len(valid_docs)} 件）\n")
    success_count = 0
    map_count = 0
    error_count = 0

    for doc_id, filename, post, cat, tags, is_map in valid_docs:
        meta = post.metadata
        body = post.content.strip()
        title = meta.get("title", doc_id)

        # date フィールドの型変換（date → datetime）
        date_val = meta.get("date")
        if isinstance(date_val, datetime.date) and not isinstance(
            date_val, datetime.datetime
        ):
            date_val = datetime.datetime.combine(date_val, datetime.time.min)

        # --- A. ナレッジマップ・ルーティング ---
        if is_map:
            try:
                ref = db.collection("knowledge_meta").document(doc_id)
                data = {
                    "title": doc_id,
                    "display_title": title,
                    "category": cat,
                    "tags": tags,
                    "content": body,
                    "requires": meta.get("requires", []),
                    "enables": meta.get("enables", []),
                    "level": meta.get("level", ""),
                    "date": date_val,
                    "doc_type": "knowledge_map",
                    "updated_at": firestore.SERVER_TIMESTAMP,
                }
                data = {k: v for k, v in data.items() if v is not None}
                ref.set(data, merge=True)
                print(f"   📋 [MAP] {filename} -> 同期完了")
                map_count += 1
            except Exception as e:
                print(f"   ❌ [MAP] {filename} -> 登録エラー: {e}")
                error_count += 1
            continue

        # --- B. 通常文書のベクトル化 ---
        try:
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

            doc_ref = db.collection("tech_docs").document(doc_id)
            doc_data = {
                "title": doc_id,
                "display_title": title,
                "category": cat,
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
            print(f"   ✅ [DOC] {filename} -> 同期完了 (要約{mark_sum})")
            success_count += 1

        except Exception as e:
            print(f"   ❌ [DOC] {filename} -> 登録エラー: {e}")
            error_count += 1

    # --- 4. 診断レポート ---
    print("\n" + "=" * 50)
    print(f"🎉 MixedRAG パイプライン完了")
    print(f"   除霊:       {ghost_total} 件")
    print(f"   ベクトル化: {success_count} 件")
    print(f"   メタデータ: {map_count} 件")
    print(f"   エラー:     {error_count + parse_errors} 件")
    print("=" * 50)

    if error_count + parse_errors > 0:
        print("\n⚠️  エラーが発生しています。上のログを確認してください。")


if __name__ == "__main__":
    process_mixed_rag()

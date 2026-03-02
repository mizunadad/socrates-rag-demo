# 仕様書：ナレッジマップのルーティング対応
## ── build_vector_db.py への機能追加

作成日: 2026-02-21
ステータス: 仕様確定（実装は CS1 パイプライン検証時）

---

## 1. 背景と目的

ナレッジベースに「ナレッジマップ」（knowledge_map.md）という非ベクトル化文書が追加された。
この文書は個別技術の解説ではなく、文書群全体の依存関係と学習順序を示す索引である。

- ベクトル検索の対象にすると、あらゆるクエリに中途半端にヒットしてノイズになる
- しかしシステムからは参照可能にしておきたい（学習パス提案、全体像の提示）

**目的**: ベクトル化パイプラインに「ベクトル化除外 + メタデータ登録」のルーティング機能を追加する。

---

## 2. 判定ロジック

### 除外条件

Frontmatter の `tags` に `"ナレッジマップ"` を含む文書をベクトル化から除外する。

```python
SKIP_VECTORIZATION_TAGS = ["ナレッジマップ"]
```

**ファイル名ではなくタグで判定する理由:**

- ファイル名はリネームされる可能性がある
- タグはFrontmatter仕様の一部であり、文書作成者が意図的に付与する
- 将来、他の除外カテゴリ（例: "changelog", "index"）を追加する際にリストへの追記で済む

---

## 3. 処理フロー（変更箇所）

現在の `process_all_folders()` 内、`load_file_robust()` 成功後に分岐を追加する。

```
既存フロー:
  load_file_robust() → タグサニタイズ → ベクトル化 → Firestore登録

変更後フロー:
  load_file_robust() → タグサニタイズ → ★分岐判定★
    │
    ├─ [通常文書] → ベクトル化 → tech_docs コレクションに登録（既存処理）
    │
    └─ [ナレッジマップ] → ベクトル化スキップ → knowledge_meta コレクションに登録
```

### 挿入位置（現行コード L199 付近、タグサニタイズ完了後）

```python
# --- ここから追加 ---
SKIP_VECTORIZATION_TAGS = ["ナレッジマップ"]

if any(tag in SKIP_VECTORIZATION_TAGS for tag in tags):
    # ベクトル化せず、メタデータのみ登録
    doc_id = os.path.splitext(filename)[0]
    meta_ref = db.collection("knowledge_meta").document(doc_id)
    meta_data = {
        'title': metadata.get('title', doc_id),
        'display_title': metadata.get('title', ''),
        'category': category_key,
        'tags': tags,
        'content': content_body,        # 全文保持（LLMが参照する）
        'requires': metadata.get('requires', []),
        'enables': metadata.get('enables', []),
        'level': metadata.get('level', ''),
        'date': date_val,
        'doc_type': 'knowledge_map',    # 文書種別を明示
        'updated_at': firestore.SERVER_TIMESTAMP
    }
    meta_data = {k: v for k, v in meta_data.items() if v is not None}
    meta_ref.set(meta_data, merge=True)
    print(f"📋 [{category_key}] {filename} -> メタデータ登録（ベクトル化スキップ）")
    success_count += 1
    continue
# --- ここまで追加 ---
```

---

## 4. Firestore コレクション設計

### 既存: `tech_docs`（変更なし）

通常のベクトル化済み文書。embedding フィールドを持つ。

### 新規: `knowledge_meta`

| フィールド | 型 | 説明 |
|:-----------|:---|:-----|
| title | string | ドキュメントID（ファイル名ベース） |
| display_title | string | 表示用タイトル |
| category | string | カテゴリ |
| tags | array | タグ一覧 |
| content | string | 本文全文（LLM参照用） |
| requires | array | 依存関係 |
| enables | array | 実現関係 |
| level | string | 難易度 |
| date | timestamp | 作成日 |
| doc_type | string | 文書種別（"knowledge_map" 等） |
| updated_at | timestamp | 更新日時 |

**embedding フィールドは持たない**（ベクトル検索の対象外）。

---

## 5. streamlit_app.py 側の参照方法（概要のみ）

RAGアプリ側では、以下のようなケースで `knowledge_meta` を参照する。

**ケース1: 全体像の質問**
ユーザーが「何から読めばいい？」「全体像を教えて」と聞いた場合、
ベクトル検索ではなく `knowledge_meta` から直接取得して LLM に渡す。

**ケース2: Exit Trigger 後の推薦**
学習完了判定（Exit Trigger）後、次の文書を推薦するときに
ナレッジマップの依存関係ツリーを参照して「次はこれ」を提案する。

**ケース3: コーチングモードでのナビゲーション**
Teacher/Coaching モード切替時、現在の文書の位置づけを
ナレッジマップから取得して文脈を補強する。

※ 実装詳細は streamlit_app.py 側の改修仕様として別途定義する。

---

## 6. 拡張性

### 除外タグの追加

将来、以下のような非ベクトル化文書が増えた場合、リストに追記するだけで対応できる。

```python
SKIP_VECTORIZATION_TAGS = ["ナレッジマップ", "changelog", "index", "glossary"]
```

### doc_type による分岐

`knowledge_meta` コレクション内で `doc_type` フィールドにより文書種別を区別できる。

- `"knowledge_map"` → 依存関係ツリー・学習パス
- `"glossary"` → 用語集（将来追加時）
- `"changelog"` → 更新履歴（将来追加時）

---

## 7. 検証項目（CS1 実装時に確認）

- [ ] `"ナレッジマップ"` タグ付き文書がベクトル化されない（`tech_docs` に embedding が登録されない）
- [ ] 同文書が `knowledge_meta` コレクションに content 含めて登録される
- [ ] タグなし文書（通常の6本）は従来通り `tech_docs` に登録される
- [ ] 診断ログに `📋 メタデータ登録（ベクトル化スキップ）` が出力される
- [ ] streamlit_app.py から `knowledge_meta` の content を取得できる

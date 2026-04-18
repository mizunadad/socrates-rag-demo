# 答えを教えないRAG ── ソクラテス

**Socratic RAG**: 答えを直接教えず、ソクラテス式の問いかけで学習者を導くRAGアプリケーション。

LLMを「教師の脳」として中央に置き、Python制御層で入出力を挟む **疑似メタ認知アーキテクチャ** により、「答えない」という不自然な振る舞いをLLMに強制します。

> プロンプトは **お願い**。Pythonコードは **法律**。

## アーキテクチャ概要

```
  ユーザー入力
       │
       ▼
┌──────────────────────────────────────────┐
│  Pre-hooks（入力検閲）                     │
│  ├─ Scope Guard: ドメイン外の質問を遮断     │
│  └─ Deadlock Breaker: 質問ループの膠着検知  │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  LLM（Claude Haiku 4.5）                  │
│  モード別プロンプトで応答生成               │
│  ├─ Teacher モード: ソクラテス式問いかけ     │
│  └─ Coaching モード: レベル別ヒント提供      │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  Post-hooks（出力検閲）                    │
│  ├─ Socratic Validation: 答えを教えていないか検証
│  ├─ One-shot Retry: 違反時に1回だけ再生成   │
│  └─ Exit Trigger: 正解到達を検知して祝福    │
└──────┬───────────────────────────────────┘
       │
       ▼
  応答表示
```

## 技術スタック

| レイヤー | 技術 | 役割 |
|:--|:--|:--|
| Frontend | Streamlit | チャットUI・設定パネル |
| LLM | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) | 応答生成 |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 | ベクトル化・類似度計算 |
| Vector DB | Google Firestore | 文書ストレージ + ベクトル検索 |
| 類似度計算 | scikit-learn (`cosine_similarity`) | Scope Guard判定 |

## リポジトリ構成

```
socrates-rag-demo/
├── step3/
│   └── app.py              # NewRAG版（単一ドメイン）
├── step4/
│   └── app.py              # MixedRAG版（マルチドメイン）
├── docs/                   # デモ文書（Frontmatter付きMarkdown）
├── build_vector_db_for_NewRAG.py    # ベクトル化パイプライン（NewRAG用）
├── build_vector_db_for_MixedRAG.py  # ベクトル化パイプライン（MixedRAG用）
├── requirements.txt
├── LICENSE
└── README.md
```

**step3** と **step4** の違い:

- `step3/app.py` — 単一ドメイン（ソフトウェアテスト哲学）。最小構成で動作原理を理解するためのエントリーポイント。
- `step4/app.py` — マルチドメイン対応。step3からの差分は4箇所のみ（`DOMAIN_CONFIG`辞書の追加、アンカーテキスト、サイドバーUI、Scope Guard切替）。コードを大きく変えずにドメインを拡張できることを示す。

## セットアップ

### 1. 前提条件

- Python 3.10 以上
- Google Cloud プロジェクト（Firestore有効化済み）
- Anthropic API Key

### 2. インストール

```bash
git clone https://github.com/mizunadad/socrates-rag-demo.git
cd socrates-rag-demo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Firestore サービスアカウント

Google Cloud Console からサービスアカウントキーをダウンロードし、プロジェクトルートに配置します。

```
socrates-rag-demo/
└── serviceAccountKey.json   # .gitignore 済み
```

### 4. Streamlit Secrets の設定

`.streamlit/secrets.toml` を作成し、API Keyを設定します。

```toml
# .streamlit/secrets.toml
ANTHROPIC_API_KEY = "sk-ant-xxxxx"
```

### 5. デモ文書のベクトル化

Firestoreに文書をベクトル化して登録します。

```bash
# NewRAG（step3用）: docs/ 配下の文書をベクトル化
python build_vector_db_for_NewRAG.py

# MixedRAG（step4用）: 複数ドメインの文書をベクトル化
python build_vector_db_for_MixedRAG.py
```

### 6. アプリの起動

```bash
# step3（単一ドメイン版）
streamlit run step3/app.py

# step4（マルチドメイン版）
streamlit run step4/app.py
```

## パスワード認証の追加（オプション）

デプロイ時にアクセス制限をかけたい場合、以下の手順で簡易パスワード認証を追加できます。

### 1. secrets.toml にパスワードを追加

```toml
# .streamlit/secrets.toml
ANTHROPIC_API_KEY = "sk-ant-xxxxx"
APP_PASSWORD = "your-password-here"
```

### 2. app.py の先頭に認証チェックを追加

```python
# --- Password Gate ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True
    pwd = st.text_input("パスワードを入力してください", type="password")
    if pwd == st.secrets.get("APP_PASSWORD", ""):
        st.session_state.authenticated = True
        st.rerun()
    elif pwd:
        st.error("パスワードが正しくありません")
    return False

if not check_password():
    st.stop()
# --- End Password Gate ---
```

## 関連記事

このリポジトリは Zenn 記事シリーズ「答えを教えないRAG ── ソクラテス 開発記」の実装コードです。

<!-- TODO: 記事URL確定後に更新 -->
- Vol.1: ソクラテスに説教されるシステムが爆誕した話
- Vol.2: Frontmatterで"学習指導要領"を定義する
- Vol.3: 暴走するRAGソクラテスと、メタ認知の必要性
- Vol.4: HOOK機能で「暴走RAG」を「答えないソクラテス」へ
- Vol.5: 疑似メタ認知アーキテクチャの全体設計
- Vol.6: 128トークンに賭けるデータ戦略
- Vol.7: ベクトル化パイプライン ── 3段フォールバックの実装
- Vol.8: Streamlit UIの設計 ── 教育的チャットの作法
- Vol.9: （執筆中）

## ライセンス

MIT License. 詳細は [LICENSE](./LICENSE) を参照してください。

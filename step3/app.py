"""
step3/app.py ── Phase 1: NewRAG 検証用 Streamlit アプリ（最小構成版）

[FILE] step3/app.py
NOTE: このファイルは Step 3（NewRAG / Phase 1）専用の最小構成版です。
      運用フェーズ（MixedRAG / Phase 3）は step4/app.py を参照してください。

含まれる機能:
  - Teacher / Coaching モード切替
  - ソクラテス・レベル切替（L1 / L2 / L3）
  - Scope Guard（Pre-hook: 概念ベクトル判定）
  - Socratic Validation（Post-hook: 文字数・疑問形チェック、Coaching時のみ）
  - One-shot Retry（Validation失敗時の再生成、Coaching時のみ）
  - ベクトル検索（コサイン類似度）
  - ストリーミング表示
  - 検索ソースの表示

含まれない機能（step4/app.py または本体 streamlit_app.py で提供）:
  - カテゴリ選択UI（本ファイルではドメイン固定）
  - Deadlock Breaker / Exit Trigger
  - 会話履歴の圧縮（compress_history）
  - パスワード認証
  - ナレッジグラフ可視化
  - セーブ / ロード / 共有
"""

import streamlit as st
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from sentence_transformers import SentenceTransformer
import anthropic
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# ⚙️ 設定（ドメイン固定: ソフトウェアテスト哲学）
# ==========================================

# Firestore 検索対象カテゴリ（デモ文書の category 値に一致させる）
TARGET_CATEGORIES = ["Strategy_Design", "Technology"]

# Scope Guard 用のトピックアンカー（ドメインの概念定義）
TOPIC_ANCHOR_TEXT = """
ソフトウェアテスト、品質保証、テスト戦略、テストピラミッド、
境界値分析、シフトレフト、テスト自動化、品質管理、
テスト設計、バグ検出、回帰テスト、テスト工程、QA、
ソフトウェア品質、テストプロセス、検証と妥当性確認
"""

# Scope Guard 閾値（低いほど厳しい。0.15 は本体 streamlit_app.py と同一）
SCOPE_THRESHOLD = 0.15

# ==========================================
# 🔧 初期化
# ==========================================

st.set_page_config(
    page_title="Socratic RAG - Phase 1",
    page_icon="🏛️",
    layout="centered",
)


@st.cache_resource
def setup_firestore():
    """Firestore 接続（secrets 優先 → ローカルファイルにフォールバック）"""
    if not firebase_admin._apps:
        try:
            if "firebase" in st.secrets:
                cert_dict = json.loads(st.secrets["firebase"]["cert_json"])
                cred = credentials.Certificate(cert_dict)
            else:
                cred = credentials.Certificate("serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Firestore 接続エラー: {e}")
            return None
    return firestore.client()


@st.cache_resource
def load_embedding_model():
    """Embedding モデルのロード（初回のみ）"""
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data
def get_anchor_embedding():
    """Scope Guard 用アンカーベクトルの事前計算"""
    model = load_embedding_model()
    return model.encode([TOPIC_ANCHOR_TEXT])


# ==========================================
# 🛡️ Pre-hook: Scope Guard
# ==========================================

def pre_hook_scope_guard(query, threshold=SCOPE_THRESHOLD):
    """
    ユーザーの質問がドメイン内かどうかを概念ベクトルで判定する。
    短文の会話コマンド（「わからん」「教えて」等）はバイパスする。
    """
    try:
        # バイパス: 短文 + 会話キーワード → 無条件通過
        safe_keywords = [
            "わから", "分から", "教えて", "ヒント", "正解", "答え",
            "ありがとう", "こんにちは", "続け", "はい", "いいえ",
        ]
        if len(query) < 30 and any(kw in query for kw in safe_keywords):
            return True, ""

        model = load_embedding_model()
        query_vec = model.encode([query])
        anchor_vec = get_anchor_embedding()

        similarity = cosine_similarity(query_vec, anchor_vec)[0][0]

        if similarity < threshold:
            debug_info = (
                f"\n\n*(🕵️ Debug: Similarity **{similarity:.4f}** "
                f"< Threshold {threshold})*"
            )
            return False, (
                f"🚫 **Scope Guard**: 現在は『ソフトウェアテスト哲学』の時間です。"
                f"その話題は学習範囲外のようです。{debug_info}"
            )

        return True, ""
    except Exception:
        # エラー時は通過させる（アプリを止めない）
        return True, ""


# ==========================================
# 🛡️ Post-hook: Socratic Validation
# ==========================================

def socratic_validation(response_text, level=1):
    """
    Coaching モード専用。レベルに応じた検閲:
      - 文字数制限（レベルが上がるほど短く）
      - 末尾の疑問形チェック（答えを教えていないか）
    """
    limits = {1: 160, 2: 120, 3: 80}
    max_len = limits.get(level, 160)

    if len(response_text) > max_len:
        feedback = (
            f"回答が長すぎます（現在{len(response_text)}文字）。"
            f"情報の贅肉を削ぎ落とし、{max_len}文字以内で"
            f"本質的な『問い』に絞ってください。"
        )
        return False, feedback

    if "？" not in response_text[-10:] and "?" not in response_text[-10:]:
        feedback = (
            "教えっぱなしは禁止です。"
            "最後は必ずユーザーへの『問いかけ』で締めてください。"
        )
        return False, feedback

    return True, None


# ==========================================
# 🔍 ベクトル検索
# ==========================================

def search_documents(query, top_k=3):
    """
    Firestore からドメイン内の文書を取得し、コサイン類似度で上位を返す。
    """
    db = setup_firestore()
    model = load_embedding_model()
    if not db or not model:
        return []

    query_embedding = model.encode(query)
    query_dim = len(query_embedding)

    all_docs = []
    try:
        docs_ref = (
            db.collection("tech_docs")
            .where("category", "in", TARGET_CATEGORIES)
            .stream()
        )
        for doc in docs_ref:
            data = doc.to_dict()
            embedding = data.get("embedding")
            # 次元チェック（embedding 削除済みの亡霊を自動除外）
            if embedding and len(embedding) == query_dim:
                data["doc_id"] = doc.id
                all_docs.append(data)

        if not all_docs:
            return []

        doc_embeddings = np.array([d["embedding"] for d in all_docs])
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), doc_embeddings
        ).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for i in top_indices:
            doc = all_docs[i]
            doc["score"] = float(similarities[i])
            results.append(doc)
        return results

    except Exception as e:
        st.error(f"検索エラー: {e}")
        return []


# ==========================================
# 🎨 UI
# ==========================================

def main():
    # --- APIキー ---
    api_key = None
    try:
        api_key = st.secrets.get("CLAUDE_API_KEY")
    except Exception:
        pass

    # --- サイドバー ---
    st.sidebar.title("⚙️ 設定")

    if not api_key:
        api_key = st.sidebar.text_input(
            "Anthropic API Key",
            type="password",
            help="ローカル実行時はここに入力。デプロイ時は Secrets に設定。",
        )

    # Teacher / Coaching 切替
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 学習スタイル")

    if "chat_style" not in st.session_state:
        st.session_state.chat_style = "Teacher"

    style_mapping = {
        "Teacher": "👨‍🏫 Teacher Mode（解説・知識重視）",
        "Coaching": "🧘 Coaching Mode（気づき・思考重視）",
    }
    selected_style = st.sidebar.radio(
        "AIの振る舞いを選んでください:",
        options=["Teacher", "Coaching"],
        format_func=lambda x: style_mapping[x],
        key="style_radio",
        index=0 if st.session_state.chat_style == "Teacher" else 1,
    )
    if st.session_state.chat_style != selected_style:
        st.session_state.chat_style = selected_style
        st.session_state.messages.append({
            "role": "system",
            "content": (
                f"【システム通知】これよりAIのモードが"
                f"『{selected_style}』に変更されました。"
                f"以前の振る舞いを捨て、新しい役割に徹してください。"
            ),
        })

    # ソクラテス・レベル切替
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧘 Socratic Tuning")

    if "socratic_level" not in st.session_state:
        st.session_state.socratic_level = 1

    st.session_state.socratic_level = st.sidebar.select_slider(
        "ソクラテス・レベル",
        options=[1, 2, 3],
        value=st.session_state.socratic_level,
        help="レベルが上がるほど、AIは答えを教えなくなり、思考を強います。",
    )
    level_labels = {
        1: "🟢 L1: 伴走（ヒントあり）",
        2: "🟡 L2: 沈黙（比喩・問い）",
        3: "🔴 L3: 鉄仮面（問いのみ）",
    }
    st.sidebar.caption(
        f"現在の設定: **{level_labels[st.session_state.socratic_level]}**"
    )

    # 履歴クリア
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ 会話履歴をクリア", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # --- メインエリア ---
    st.title("🏛️ Socratic RAG: Phase 1")
    st.caption(
        "NewRAG（最小構成）による知識検索 + ソクラテス対話の検証"
    )

    # セッション初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 会話履歴の表示
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue  # システムメッセージはUIに表示しない
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 参照した知識"):
                    for src in msg["sources"]:
                        st.caption(
                            f"**{src['title']}** (Score: {src['score']:.4f})"
                        )

    # --- チャット入力 ---
    if not api_key:
        st.info("👈 サイドバーに Anthropic API Key を入力してください。")
        return

    prompt = st.chat_input("質問を入力してください（例: テストの本当の目的は？）")

    if not prompt:
        return

    # ==============================================
    # 🛡️ 1. Pre-hook: Scope Guard
    # ==============================================
    is_valid, error_msg = pre_hook_scope_guard(prompt)

    if not is_valid:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append(
            {"role": "assistant", "content": error_msg}
        )
        with st.chat_message("assistant"):
            st.markdown(error_msg)
        st.stop()

    # ユーザー入力を表示・保存（ガード通過後）
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ==============================================
    # 🔍 2. RAG検索
    # ==============================================
    with st.chat_message("assistant"):
        results = search_documents(prompt, top_k=3)

        if not results:
            msg = "⚠️ データベースから知識を取得できませんでした。しばらく待ってから再試行してください。"
            st.error(msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": msg}
            )
            st.stop()

        # コンテキスト構築
        context = "\n\n---\n\n".join(
            [
                f"■ {r.get('display_title', r.get('title', '不明'))}\n{r.get('content', '')}"
                for r in results
            ]
        )
        sources = [
            {
                "title": r.get("display_title", r.get("title", "不明")),
                "score": r.get("score", 0.0),
            }
            for r in results
        ]

        # ==============================================
        # 🤖 3. プロンプト構築 + LLM呼び出し
        # ==============================================
        if st.session_state.chat_style == "Teacher":
            style_instruction = """
あなたは「解説好きな先生」です。
ユーザーの質問に対し、以下の【参照データ】を元に、論理的かつ分かりやすく解説してください。
専門用語には噛み砕いた説明を加え、学習をサポートしてください。
"""
        else:
            style_instruction = """
あなたは「ソクラテス式の厳しいコーチ」です。
ユーザーは生徒であり、あなたの目的は「知識を与えること」ではなく「自ら考えさせること」です。

【対話の絶対ルール】
1. **解説禁止**: ユーザーが「〜とは？」「教えて」と聞いても、絶対に定義や解説を述べないでください。
2. **逆質問**: 「あなたはどう思いますか？」「その言葉の定義は何だと考えますか？」と質問で返してください。
3. **短文回答**: 1回の回答は「100文字以内」に抑えてください。長々と語るのは「解説」とみなされます。
4. **ヒントの制限**: ユーザーが「降参」や「わからん」と言った時だけ、例外的に【参照データ】からヒントを1つだけ出してください。
"""

        system_prompt = f"""
{style_instruction}

【参照データ】
{context}
"""

        # API送信用メッセージ（system ロールを除外）
        messages_for_api = [
            m
            for m in st.session_state.messages[-10:]
            if m.get("role") != "system"
        ]

        # ストリーミング表示
        stream_container = st.empty()
        full_response = ""

        try:
            client = anthropic.Anthropic(api_key=api_key)
            with client.messages.stream(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                system=system_prompt,
                messages=messages_for_api,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    stream_container.markdown(full_response + "▌")

            # ==============================================
            # 🛡️ 4. Post-hook: Socratic Validation
            #    （Coaching モード時のみ発動）
            # ==============================================
            if st.session_state.chat_style == "Coaching":
                is_valid, feedback = socratic_validation(
                    full_response,
                    level=st.session_state.socratic_level,
                )

                if not is_valid:
                    # One-shot Retry: レベル別の制約で再生成
                    stream_container.empty()

                    level_constraints = {
                        1: (
                            "【指示】先生として振る舞ってください。"
                            "結論を教える前に、ヒントを出してユーザーを"
                            "誘導する質問をしてください。"
                        ),
                        2: (
                            "【指示】解説を控えてください。"
                            "「それは〜のようなものです」といった比喩や、"
                            "ユーザーの常識を疑うような質問を投げかけてください。"
                        ),
                        3: (
                            "【絶対命令】あなたは冷徹なソクラテスです。"
                            "回答、解説、共感は一切不要です。"
                            "ユーザーが持っている『思い込み』を刺すような、"
                            "短い逆質問を1つだけ返してください。"
                        ),
                    }
                    constraint = level_constraints.get(
                        st.session_state.socratic_level, ""
                    )

                    with st.status(
                        f"🤔 Level {st.session_state.socratic_level} "
                        f"の思考を執行中..."
                    ):
                        retry_response = client.messages.create(
                            model="claude-3-haiku-20240307",
                            max_tokens=300,
                            system=(
                                system_prompt
                                + f"\n\n【再生成指示】\n{feedback}\n{constraint}\n"
                                + "※文章が途中で切れないよう、必ず完結させてください。"
                            ),
                            messages=messages_for_api,
                        )
                        full_response = retry_response.content[0].text

            # 最終表示
            stream_container.markdown(full_response)

            # メッセージ保存（ソース情報を含む）
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            })

            # ソース表示
            with st.expander("📚 参照した知識"):
                for src in sources:
                    st.caption(
                        f"**{src['title']}** (Score: {src['score']:.4f})"
                    )

        except anthropic.AuthenticationError:
            st.error("❌ API Key が無効です。正しいキーを入力してください。")
        except anthropic.RateLimitError:
            st.error("⚠️ API レート制限に達しました。しばらく待ってから再試行してください。")
        except Exception as e:
            st.error(f"❌ エラーが発生しました: {e}")


if __name__ == "__main__":
    main()

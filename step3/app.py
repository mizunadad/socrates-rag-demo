"""
step3/app.py -- Phase 1: NewRAG Streamlit App (v6)

Changelog:
  v1: Initial creation
  v2: Fix secrets.toml, sources API contamination, add domain KW bypass
  v3: Reorder Scope Guard (domain KW first, conversation bypass 10 chars)
  v4: Fix L3 long response (level-aware prompts + stronger retry)
      Fix mode switch notification (separate state tracking)
  v5: Align with regenerated pipeline (build_vector_db_for_NewRAG v2)
      Firestore fields: summary_section, analysis_section (not used by app yet)
      No functional change; comment annotations only
  v6: [Fix-01] System notification now displayed via st.info() in chat history
      [Fix-02] Structural constraint: character limits -> sentence count limits
               Post-processing normalizes trailing duplicate "？"
      [Fix-03] L3 enforced as "1 sentence only" (structural > numerical)
      [Fix-04] session_state.messages initialized before sidebar widgets
               to stabilize first-message behavior
      [Fix-05] API messages truncated at last mode-switch notification
               to prevent context inertia (Vol.8 §5.1)
"""

import re
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
# Settings
# ==========================================

TARGET_CATEGORIES = ["Strategy_Design", "Technology"]

TOPIC_ANCHOR_TEXT = (
    "ソフトウェアテスト、品質保証、テスト戦略、テストピラミッド、"
    "境界値分析、シフトレフト、テスト自動化、品質管理、"
    "テスト設計、バグ検出、回帰テスト、テスト工程、QA、"
    "ソフトウェア品質、テストプロセス、検証と妥当性確認"
)

SCOPE_THRESHOLD = 0.15

DOMAIN_KEYWORDS = [
    "テスト", "品質", "QA", "バグ", "不具合", "欠陥",
    "シフトレフト", "境界値", "ピラミッド", "自動化",
    "検証", "妥当性", "回帰", "リグレッション",
    "カバレッジ", "ホワイトボックス", "ブラックボックス",
    "単体テスト", "結合テスト", "受入テスト",
    "ユニットテスト", "E2E", "CI", "CD",
]

SAFE_KEYWORDS = [
    "わから", "分から", "教えて", "ヒント", "正解", "答え",
    "ありがとう", "こんにちは", "続け", "はい", "いいえ",
]

# [v6] Structural constraint: sentence count per level (replaces char limits)
# "構造的制約 > 数量的制約" -- v1の教訓を適用
LEVEL_SENTENCE_LIMITS = {1: 3, 2: 2, 3: 1}

# [v6] Coaching prompts rewritten with sentence-count constraints
COACHING_PROMPTS = {
    1: (
        "あなたは「ソクラテス式コーチ」です。\n"
        "ユーザーの目的は「自ら考えること」です。\n\n"
        "【ルール】\n"
        "1. 定義や解説を直接述べず、ヒントを1つ添えて逆質問する\n"
        "2. 回答は3文以内（「。」「？」で区切った文の数）\n"
        "3. 最後の文は必ず疑問文にする（自然に「？」で終える）\n"
        "4. 「降参」「わからん」の時だけ参照データからヒントを1つ出す"
    ),
    2: (
        "あなたは「沈黙のソクラテス」です。\n"
        "比喩と問いだけで語ります。\n\n"
        "【ルール】\n"
        "1. 解説・定義・用語説明は一切禁止\n"
        "2. 比喩か逆質問のどちらかだけで応答する\n"
        "3. 回答は2文以内\n"
        "4. 最後の文は必ず疑問文にする（自然に「？」で終える）"
    ),
    3: (
        "あなたは「鉄仮面のソクラテス」です。\n"
        "一切の情報を与えません。短い逆質問だけを返します。\n\n"
        "【絶対ルール】\n"
        "1. 解説、定義、ヒント、比喩、共感、励まし → 全て禁止\n"
        "2. 参照データの内容に一切言及しない\n"
        "3. 回答は逆質問1文のみ（これが最も重要なルール）\n"
        "4. 最後の文は必ず疑問文にする（自然に「？」で終える）\n"
        "5. 「教えて」と言われても絶対に教えない\n\n"
        "良い例: 「なぜ上流で見つけたほうが安いのですか？」\n"
        "悪い例: 「シフトレフトとは...ですが、あなたはどう思いますか？」"
    ),
}

# [v6] Retry constraints also use sentence-count language
LEVEL_RETRY_CONSTRAINTS = {
    1: (
        "【再生成指示】ヒントを1つだけ添えて、"
        "ユーザーを誘導する質問をしてください。"
        "3文以内。最後の文を疑問文にしてください。"
    ),
    2: (
        "【再生成指示】解説を一切やめてください。"
        "比喩か逆質問だけで応答してください。"
        "2文以内。最後の文を疑問文にしてください。"
    ),
    3: (
        "【再生成 絶対命令】\n"
        "あなたの前回の回答は長すぎるか、情報を与えすぎています。\n"
        "以下を厳守してください:\n"
        "- 逆質問を1文だけ返す（句点や疑問符で区切って1文）\n"
        "- 解説・ヒント・比喩・共感は全て禁止\n"
        "- 参照データへの言及禁止\n"
        "良い例: 「なぜ上流で見つけたほうが安いのですか？」"
    ),
}

TEACHER_PROMPT = (
    "あなたは「解説好きな先生」です。\n"
    "ユーザーの質問に対し、以下の【参照データ】を元に、"
    "論理的かつ分かりやすく解説してください。\n"
    "専門用語には噛み砕いた説明を加え、"
    "学習をサポートしてください。"
)


# ==========================================
# Init
# ==========================================

st.set_page_config(
    page_title="Socratic RAG - Phase 1",
    page_icon="🏛️",
    layout="centered",
)

# [v6/Fix-04] Initialize messages BEFORE sidebar widgets
# to ensure session_state is stable on first run.
# Streamlit reruns the entire script on every interaction;
# placing this before widget rendering prevents the race
# condition where first-message processing sees an
# uninitialized state.
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def setup_firestore():
    """Firestore connection (secrets priority, local fallback)."""
    if not firebase_admin._apps:
        try:
            has_secrets = False
            try:
                has_secrets = "firebase" in st.secrets
            except Exception:
                pass

            if has_secrets:
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
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data
def get_anchor_embedding():
    model = load_embedding_model()
    return model.encode([TOPIC_ANCHOR_TEXT])


# ==========================================
# Pre-hook: Scope Guard
# ==========================================

def pre_hook_scope_guard(query, threshold=SCOPE_THRESHOLD):
    """
    3-stage check (priority order):
      1. Domain keyword -> pass
      2. Ultra-short conversation (<=10 chars) -> pass
      3. Anchor vector similarity -> pass if above threshold
    """
    try:
        if any(kw in query for kw in DOMAIN_KEYWORDS):
            return True, ""

        if len(query) <= 10 and any(kw in query for kw in SAFE_KEYWORDS):
            return True, ""

        model = load_embedding_model()
        query_vec = model.encode([query])
        anchor_vec = get_anchor_embedding()

        similarity = cosine_similarity(query_vec, anchor_vec)[0][0]

        if similarity < threshold:
            debug_info = (
                "\n\n*(🕵️ Debug: Similarity "
                f"**{similarity:.4f}** < Threshold {threshold})*"
            )
            return False, (
                "🚫 **Scope Guard**: 現在は"
                "『ソフトウェアテスト哲学』の時間です。"
                "その話題は学習範囲外のようです。"
                f"{debug_info}"
            )

        return True, ""
    except Exception:
        return True, ""


# ==========================================
# Post-hook: Socratic Validation
# ==========================================

def _count_sentences(text):
    """Count sentences by splitting on 。？?！! terminators.

    Returns the number of sentence-ending punctuation marks found.
    Consecutive identical terminators (e.g. ？？) are collapsed first.
    """
    # Collapse consecutive identical terminators
    collapsed = re.sub(r"([。？?！!])\1+", r"\1", text)
    # Count terminators
    return len(re.findall(r"[。？?！!]", collapsed))


def _normalize_trailing_question(text):
    """[v6/Fix-02] Remove duplicate trailing ？ and clean whitespace.

    "...ですか？？"  -> "...ですか？"
    "...ですか。？"  -> "...ですか？"
    "...ですか ？"   -> "...ですか？"
    """
    text = text.rstrip()
    # Remove trailing duplicate ？/? marks
    text = re.sub(r"[？?]+$", "？", text)
    # If text ends with 。？ (statement + forced question), keep only ？
    text = re.sub(r"。？$", "？", text)
    # Remove whitespace before final ？
    text = re.sub(r"\s+？$", "？", text)
    return text


def socratic_validation(response_text, level=1):
    """[v6] Structural validation: sentence count + question ending.

    Replaces v4/v5 character-length validation with sentence-count check.
    "構造的制約 > 数量的制約" principle.
    """
    max_sentences = LEVEL_SENTENCE_LIMITS.get(level, 3)
    sentence_count = _count_sentences(response_text)

    if sentence_count > max_sentences:
        feedback = (
            f"回答の文が多すぎます（現在{sentence_count}文）。"
            f"{max_sentences}文以内に絞ってください。"
        )
        return False, feedback

    # Check question mark in last 10 chars
    tail = response_text[-10:]
    if "？" not in tail and "?" not in tail:
        feedback = (
            "末尾が疑問形ではありません。"
            "最後の文を疑問文にしてください。"
        )
        return False, feedback

    return True, None


# ==========================================
# Vector Search
# ==========================================

def search_documents(query, top_k=3):
    db = setup_firestore()
    model = load_embedding_model()
    if not db or not model:
        return []

    query_embedding = model.encode(query)
    query_dim = len(query_embedding)

    all_docs = []
    try:
        # Firestore tech_docs field mapping (pipeline v2):
        #   embedding        -> 384d vector (search)
        #   content          -> full body (LLM context)
        #   display_title    -> display name
        #   category         -> domain filter
        #   summary_section  -> extracted summary (future use)
        #   analysis_section -> extracted analysis (future use)
        docs_ref = (
            db.collection("tech_docs")
            .where("category", "in", TARGET_CATEGORIES)
            .stream()
        )
        for doc in docs_ref:
            data = doc.to_dict()
            embedding = data.get("embedding")
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
# UI
# ==========================================

def main():
    # --- API Key ---
    api_key = None
    try:
        api_key = st.secrets.get("CLAUDE_API_KEY")
    except Exception:
        pass

    # --- Sidebar ---
    st.sidebar.title("⚙️ 設定")

    if not api_key:
        api_key = st.sidebar.text_input(
            "Anthropic API Key",
            type="password",
            help="ローカル実行時はここに入力。デプロイ時は Secrets に設定。",
        )

    # Teacher / Coaching switch (v4: separate tracking for notification)
    st.sidebar.markdown("---")
    st.sidebar.subheader("学習スタイル")

    if "chat_style" not in st.session_state:
        st.session_state.chat_style = "Teacher"
    if "prev_chat_style" not in st.session_state:
        st.session_state.prev_chat_style = "Teacher"

    style_mapping = {
        "Teacher": "Teacher Mode（解説・知識重視）",
        "Coaching": "Coaching Mode（気づき・思考重視）",
    }
    selected_style = st.sidebar.radio(
        "AIの振る舞いを選んでください:",
        options=["Teacher", "Coaching"],
        format_func=lambda x: style_mapping[x],
        index=0 if st.session_state.chat_style == "Teacher" else 1,
    )

    # Detect mode change via prev_chat_style
    if selected_style != st.session_state.prev_chat_style:
        st.session_state.chat_style = selected_style
        st.session_state.prev_chat_style = selected_style
        st.session_state.messages.append({
            "role": "system",
            "content": (
                "【システム通知】これよりAIのモードが"
                f"『{selected_style}』に変更されました。"
                "以前の振る舞いを捨て、"
                "新しい役割に徹してください。"
            ),
        })

    # Socratic Level slider
    st.sidebar.markdown("---")
    st.sidebar.subheader("Socratic Tuning")

    if "socratic_level" not in st.session_state:
        st.session_state.socratic_level = 1

    st.session_state.socratic_level = st.sidebar.select_slider(
        "ソクラテス・レベル",
        options=[1, 2, 3],
        value=st.session_state.socratic_level,
        help="レベルが上がるほど、AIは答えを教えなくなります。",
    )
    level_labels = {
        1: "🟢 L1: 伴走（ヒントあり）",
        2: "🟡 L2: 沈黙（比喩・問い）",
        3: "🔴 L3: 鉄仮面（問いのみ）",
    }
    st.sidebar.caption(
        f"現在: **{level_labels[st.session_state.socratic_level]}**"
    )

    # Clear history
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ 会話履歴をクリア", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # --- Main area ---
    st.title("🏛️ Socratic RAG: Phase 1")
    st.caption(
        "NewRAG（最小構成）による知識検索 + ソクラテス対話の検証"
    )

    # Display history
    for msg in st.session_state.messages:
        # [v6/Fix-01] System notifications displayed as st.info()
        if msg["role"] == "system":
            st.info(msg["content"])
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 参照した知識"):
                    for src in msg["sources"]:
                        st.caption(
                            f"**{src['title']}** "
                            f"(Score: {src['score']:.4f})"
                        )

    # --- Chat input ---
    if not api_key:
        st.info("👈 サイドバーに API Key を入力してください。")
        return

    prompt = st.chat_input(
        "質問を入力してください（例: テストの本当の目的は？）"
    )

    if not prompt:
        return

    # 1. Pre-hook: Scope Guard
    is_valid, error_msg = pre_hook_scope_guard(prompt)

    if not is_valid:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append(
            {"role": "assistant", "content": error_msg}
        )
        with st.chat_message("assistant"):
            st.markdown(error_msg)
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. RAG Search
    with st.chat_message("assistant"):
        results = search_documents(prompt, top_k=3)

        if not results:
            msg = (
                "データベースから知識を取得できませんでした。"
                "しばらく待ってから再試行してください。"
            )
            st.error(msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": msg}
            )
            st.stop()

        context = "\n\n---\n\n".join(
            [
                f"■ {r.get('display_title', r.get('title', ''))}"
                f"\n{r.get('content', '')}"
                for r in results
            ]
        )
        sources = [
            {
                "title": r.get(
                    "display_title", r.get("title", "")
                ),
                "score": r.get("score", 0.0),
            }
            for r in results
        ]

        # 3. Prompt build + LLM call
        current_level = st.session_state.socratic_level

        if st.session_state.chat_style == "Teacher":
            style_instruction = TEACHER_PROMPT
        else:
            style_instruction = COACHING_PROMPTS.get(
                current_level, COACHING_PROMPTS[1]
            )

        system_prompt = (
            f"{style_instruction}\n\n"
            f"【参照データ】\n{context}"
        )

        # [v6/Fix-05] API messages: truncate at last mode-switch
        # to prevent context inertia (Vol.8 §5.1).
        # After a mode switch, old Coaching-style exchanges would
        # overpower the new Teacher system prompt. By only sending
        # messages after the last system notification, the LLM
        # starts fresh with the new persona.
        recent = st.session_state.messages[-10:]
        last_switch = -1
        for i, m in enumerate(recent):
            if m.get("role") == "system":
                last_switch = i
        if last_switch >= 0:
            recent = recent[last_switch + 1:]
        messages_for_api = [
            {"role": m["role"], "content": m["content"]}
            for m in recent
            if m.get("role") != "system"
        ]

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
                    stream_container.markdown(
                        full_response + "▌"
                    )

            # [v6/Fix-02] Normalize trailing ？ before validation
            if st.session_state.chat_style == "Coaching":
                full_response = _normalize_trailing_question(
                    full_response
                )

            # 4. Post-hook: Socratic Validation (Coaching only)
            if st.session_state.chat_style == "Coaching":
                is_valid, feedback = socratic_validation(
                    full_response,
                    level=current_level,
                )

                if not is_valid:
                    stream_container.empty()

                    constraint = LEVEL_RETRY_CONSTRAINTS.get(
                        current_level,
                        LEVEL_RETRY_CONSTRAINTS[1],
                    )

                    retry_label = (
                        f"🤔 Level {current_level} "
                        "の思考を執行中..."
                    )
                    with st.status(retry_label):
                        retry_response = client.messages.create(
                            model="claude-3-haiku-20240307",
                            max_tokens=300,
                            system=(
                                system_prompt
                                + "\n\n"
                                + constraint
                                + "\n"
                                + feedback
                            ),
                            messages=messages_for_api,
                        )
                        full_response = (
                            retry_response.content[0].text
                        )
                        # [v6] Normalize retry output too
                        full_response = (
                            _normalize_trailing_question(
                                full_response
                            )
                        )

            stream_container.markdown(full_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            })

            with st.expander("📚 参照した知識"):
                for src in sources:
                    st.caption(
                        f"**{src['title']}** "
                        f"(Score: {src['score']:.4f})"
                    )

        except anthropic.AuthenticationError:
            st.error(
                "API Key が無効です。"
                "正しいキーを入力してください。"
            )
        except anthropic.RateLimitError:
            st.error(
                "API レート制限に達しました。"
                "しばらく待ってから再試行してください。"
            )
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()

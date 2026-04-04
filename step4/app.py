"""
step4/app.py -- Phase 3: MixedRAG Streamlit App (Multi-Domain)

[FILE] step4/app.py
NOTE: このファイルは Step 4（MixedRAG / Phase 3）のマルチドメイン対応版です。
      最小構成版（NewRAG / Phase 1）は step3/app.py を参照してください。

step3/app.py からの差分（4箇所のみ）:
  1. DOMAIN_CONFIG: ドメインごとの設定を辞書で管理
  2. TOPIC_ANCHORS: ドメインごとのアンカーテキスト
  3. サイドバー: ドメイン選択UIを追加
  4. Scope Guard: 選択ドメインに応じてアンカー・キーワードを動的切替

設計のハイライト:
  コードの変更は最小限。ドメインを追加するにはDOMAIN_CONFIGに
  エントリを足すだけ。ロジックは触らない。
  「コードを変えなくても、データとカテゴリ設定の追加だけで
  マルチドメイン対応できる」ことを示す。

Base: step3/app.py v6 (646 lines, Phase 1 verified)
Model: claude-haiku-4-5-20251001 (migrated from retired claude-3-haiku-20240307)

Changelog:
  v1: Initial creation (multi-domain extension of step3 v6)
  v1.1: MODEL_NAME constant + FORMAT_CONSTRAINT added
        Haiku 4.5 produces longer, heavily Markdown-formatted responses.
        FORMAT_CONSTRAINT addresses this at the prompt layer (preventive safety).
  v1.2: Scope Guard threshold per-domain (DOMAIN_CONFIG) + sidebar slider
        Top-K configurable via sidebar slider
        Information panels (search source, original docs, category guide,
        project overview, dev workflow) as sidebar expanders
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
# [DIFF-1] Domain Configuration
# ==========================================
# step3: 単一ドメイン（SWTest固定）
# step4: 辞書でドメインを管理。追加は設定を足すだけ。

DOMAIN_CONFIG = {
    "SWTest": {
        "label": "🧪 ソフトウェアテスト哲学",
        "categories": ["Strategy_Design", "Technology"],
        "threshold": 0.30,
        "anchor": (
            "ソフトウェアテスト、品質保証、テスト戦略、テストピラミッド、"
            "境界値分析、シフトレフト、テスト自動化、品質管理、"
            "テスト設計、バグ検出、回帰テスト、テスト工程、QA、"
            "ソフトウェア品質、テストプロセス、検証と妥当性確認"
        ),
        "keywords": [
            "テスト", "品質", "QA", "バグ", "不具合", "欠陥",
            "シフトレフト", "境界値", "ピラミッド", "自動化",
            "検証", "妥当性", "回帰", "リグレッション",
            "カバレッジ", "ホワイトボックス", "ブラックボックス",
            "単体テスト", "結合テスト", "受入テスト",
            "ユニットテスト", "E2E", "CI", "CD",
        ],
        "scope_msg": "現在は『ソフトウェアテスト哲学』の時間です。",
    },
    "Singularity": {
        "label": "🚀 Tech Singularity",
        "categories": ["TECH_research"],
        "threshold": 0.20,
        "anchor": (
            "コンピュータ科学、人工知能(AI)、半導体工学、物理学、"
            "ハードウェア、ソフトウェア開発、プログラミング、アルゴリズム、"
            "シンギュラリティ、GPU、データセンター、エネルギー技術、"
            "L0(物理層)、L1(知能層)、L2、L3、L4、アーキテクチャ、統合技術"
        ),
        "keywords": [
            "AI", "人工知能", "半導体", "GPU", "シンギュラリティ",
            "アーキテクチャ", "データセンター", "アルゴリズム",
            "プログラミング", "ソフトウェア", "ハードウェア",
            "物理層", "知能層", "L0", "L1", "L2", "L3", "L4",
            "エネルギー", "コンピュータ", "機械学習", "深層学習",
            "ニューラル", "トランスフォーマー", "推論", "学習",
        ],
        "scope_msg": "現在は『Tech Singularity』の時間です。",
    },
}

# Default domain
DEFAULT_DOMAIN = "SWTest"

SCOPE_THRESHOLD_DEFAULT = 0.30

# Model: single-point constant for future migration
MODEL_NAME = "claude-haiku-4-5-20251001"

# Format constraint: Haiku 4.5 tends to produce long, Markdown-heavy responses.
# This prompt-level instruction is "preventive safety" (collision safety = max_tokens).
FORMAT_CONSTRAINT = (
    "対話形式で応答してください。Markdownの見出し、箇条書き、"
    "コードブロックは使わず、自然な文章で説明してください。"
    "回答は400字程度を目安にしてください。"
)

SAFE_KEYWORDS = [
    "わから", "分から", "教えて", "ヒント", "正解", "答え",
    "ありがとう", "こんにちは", "続け", "はい", "いいえ",
]

# Structural constraint: sentence count per level
LEVEL_SENTENCE_LIMITS = {1: 3, 2: 2, 3: 1}

# Coaching prompts (identical to step3 v6)
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
    page_title="Socratic RAG - Multi-Domain",
    page_icon="🏛️",
    layout="centered",
)

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


# [DIFF-2] Anchor embedding is now per-domain (not cached globally)
@st.cache_data
def get_anchor_embedding(anchor_text):
    model = load_embedding_model()
    return model.encode([anchor_text])


# ==========================================
# Pre-hook: Scope Guard (domain-aware)
# ==========================================

def pre_hook_scope_guard(query, domain_key, threshold=None):
    """
    [DIFF-3] Domain-aware Scope Guard.
    Uses the selected domain's keywords and anchor for validation.
    Threshold priority: explicit arg > domain config > global default.

    3-stage check (priority order):
      1. Domain keyword -> pass
      2. Ultra-short conversation (<=10 chars) -> pass
      3. Anchor vector similarity -> pass if above threshold
    """
    config = DOMAIN_CONFIG.get(domain_key, DOMAIN_CONFIG[DEFAULT_DOMAIN])

    # Threshold: slider override > domain config > global default
    if threshold is None:
        threshold = config.get("threshold", SCOPE_THRESHOLD_DEFAULT)

    try:
        # Stage 1: Domain keywords (per-domain)
        if any(kw in query for kw in config["keywords"]):
            return True, ""

        # Stage 2: Safe conversation bypass
        if len(query) <= 10 and any(kw in query for kw in SAFE_KEYWORDS):
            return True, ""

        # Stage 3: Vector similarity (per-domain anchor)
        model = load_embedding_model()
        query_vec = model.encode([query])
        anchor_vec = get_anchor_embedding(config["anchor"])

        similarity = cosine_similarity(query_vec, anchor_vec)[0][0]

        if similarity < threshold:
            debug_info = (
                "\n\n*(🕵️ Debug: Similarity "
                f"**{similarity:.4f}** < Threshold {threshold})*"
            )
            return False, (
                f"🚫 **Scope Guard**: {config['scope_msg']}"
                "その話題は学習範囲外のようです。"
                f"{debug_info}"
            )

        return True, ""
    except Exception:
        return True, ""


# ==========================================
# Post-hook: Socratic Validation (unchanged from step3 v6)
# ==========================================

def _count_sentences(text):
    collapsed = re.sub(r"([。？?！!])\1+", r"\1", text)
    return len(re.findall(r"[。？?！!]", collapsed))


def _normalize_trailing_question(text):
    text = text.rstrip()
    text = re.sub(r"[？?]+$", "？", text)
    text = re.sub(r"。？$", "？", text)
    text = re.sub(r"\s+？$", "？", text)
    return text


def socratic_validation(response_text, level=1):
    max_sentences = LEVEL_SENTENCE_LIMITS.get(level, 3)
    sentence_count = _count_sentences(response_text)

    if sentence_count > max_sentences:
        feedback = (
            f"回答の文が多すぎます（現在{sentence_count}文）。"
            f"{max_sentences}文以内に絞ってください。"
        )
        return False, feedback

    tail = response_text[-10:]
    if "？" not in tail and "?" not in tail:
        feedback = (
            "末尾が疑問形ではありません。"
            "最後の文を疑問文にしてください。"
        )
        return False, feedback

    return True, None


# ==========================================
# Vector Search (domain-aware categories)
# ==========================================

def search_documents(query, target_categories, top_k=3):
    """[DIFF-4] target_categories is now a parameter, not a global."""
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
            .where("category", "in", target_categories)
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

    # [DIFF: Domain selection UI]
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 学習ドメイン")

    if "current_domain" not in st.session_state:
        st.session_state.current_domain = DEFAULT_DOMAIN

    domain_options = list(DOMAIN_CONFIG.keys())
    domain_labels = {k: v["label"] for k, v in DOMAIN_CONFIG.items()}

    selected_domain = st.sidebar.radio(
        "ドメインを選択:",
        options=domain_options,
        format_func=lambda x: domain_labels[x],
        index=domain_options.index(st.session_state.current_domain),
    )

    # Domain change notification
    if selected_domain != st.session_state.current_domain:
        st.session_state.current_domain = selected_domain
        config = DOMAIN_CONFIG[selected_domain]
        # Reset threshold to new domain's recommended value
        st.session_state.scope_threshold = config.get(
            "threshold", SCOPE_THRESHOLD_DEFAULT
        )
        st.session_state.messages.append({
            "role": "system",
            "content": (
                f"【システム通知】学習ドメインが"
                f"『{config['label']}』に変更されました。"
                "検索対象とScope Guardが切り替わります。"
            ),
        })

    # Teacher / Coaching switch
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

    # RAG Tuning (Threshold + Top-K)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 検索チューニング")

    current_config = DOMAIN_CONFIG[st.session_state.current_domain]
    domain_default_th = current_config.get(
        "threshold", SCOPE_THRESHOLD_DEFAULT
    )

    if "scope_threshold" not in st.session_state:
        st.session_state.scope_threshold = domain_default_th
    if "top_k" not in st.session_state:
        st.session_state.top_k = 3

    st.session_state.scope_threshold = st.sidebar.slider(
        "Scope Guard 閾値",
        min_value=0.10,
        max_value=0.40,
        value=st.session_state.scope_threshold,
        step=0.05,
        help=(
            "低い→通りやすい／高い→厳しくフィルタ。"
            f"ドメイン推奨値: {domain_default_th:.2f}"
        ),
    )

    st.session_state.top_k = st.sidebar.slider(
        "参照ドキュメント数 (Top-K)",
        min_value=1,
        max_value=5,
        value=st.session_state.top_k,
        step=1,
        help="少ない→深く考える／多い→広く俯瞰する",
    )

    # Information Panels
    st.sidebar.markdown("---")
    st.sidebar.subheader("📖 ガイド")

    with st.sidebar.expander("🔍 検索ソース選択（将来機能）"):
        st.caption(
            "教育アイテムの検索範囲を絞ったり広げたりすることで、"
            "参加型の学習体験を実現します。\n\n"
            "例: 特定の文書だけに絞って深掘りしたり、"
            "全文書を横断して俯瞰的に学ぶことができます。"
        )

    with st.sidebar.expander("📄 原文参照（将来機能）"):
        st.caption(
            "AIの回答が参照した原文を直接読む機能です。\n\n"
            "一次資料に当たる習慣は、学びの質を大きく高めます。"
            "「答え」ではなく「根拠」を自分の目で確かめましょう。"
        )

    with st.sidebar.expander("🗺️ カテゴリ別ガイド"):
        st.caption(
            "**docs/knowledge_map.md** を参照すると、"
            "登録された文書群の全体像と各文書の関連性が分かります。\n\n"
            "どの文書がどのテーマをカバーしているか、"
            "学習の道筋を把握するための地図です。"
        )

    with st.sidebar.expander("📋 プロジェクト概要"):
        st.caption(
            "**specs/spec_knowledge_map_routing.md** に"
            "システム設計文書の履歴と構成が記載されています。\n\n"
            "このRAGシステムがどのような文書群で構成され、"
            "どのように進化してきたかを確認できます。"
        )

    with st.sidebar.expander("🔄 開発ワークフロー"):
        st.caption(
            "システム全体のアーキテクチャと処理フローです。\n\n"
            "Pre-hook（Scope Guard）→ RAG検索 → LLM生成 → "
            "Post-hook（Socratic Validation）の流れを理解すると、"
            "AI nativeな学び方に近づけます。"
        )

    # Clear history
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ 会話履歴をクリア", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # --- Main area ---
    current_domain = st.session_state.current_domain
    config = DOMAIN_CONFIG[current_domain]

    st.title("🏛️ Socratic RAG: Multi-Domain")
    st.caption(
        f"MixedRAG ── 現在のドメイン: {config['label']}"
    )

    # Display history
    for msg in st.session_state.messages:
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

    prompt = st.chat_input("質問を入力してください")

    if not prompt:
        return

    # 1. Pre-hook: Scope Guard (domain-aware)
    is_valid, error_msg = pre_hook_scope_guard(
        prompt, current_domain,
        threshold=st.session_state.scope_threshold,
    )

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

    # 2. RAG Search (domain-aware categories)
    with st.chat_message("assistant"):
        results = search_documents(
            prompt,
            target_categories=config["categories"],
            top_k=st.session_state.top_k,
        )

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
            f"【出力形式】\n{FORMAT_CONSTRAINT}\n\n"
            f"【参照データ】\n{context}"
        )

        messages_for_api = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-10:]
            if m.get("role") != "system"
        ]

        stream_container = st.empty()
        full_response = ""

        try:
            client = anthropic.Anthropic(api_key=api_key)
            with client.messages.stream(
                model=MODEL_NAME,
                max_tokens=1000,
                system=system_prompt,
                messages=messages_for_api,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    stream_container.markdown(
                        full_response + "▌"
                    )

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
                            model=MODEL_NAME,
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

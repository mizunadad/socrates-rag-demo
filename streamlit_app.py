import streamlit as st
import os 
import json
import firebase_admin
from firebase_admin import credentials, firestore
from sentence_transformers import SentenceTransformer
import anthropic
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import base64
from streamlit_agraph import agraph, Node, Edge, Config
import streamlit.components.v1 as components
import re
import datetime # 👈 保存機能に必須

# ==========================================
# ⚙️ 1. 初期設定・認証・DB接続
# ==========================================

# --- Firestore接続 ---
@st.cache_resource
def setup_firestore():
    if not firebase_admin._apps:
        try:
            if "firebase" in st.secrets:
                cert_json_string = st.secrets["firebase"]["cert_json"] 
                cert_dict = json.loads(cert_json_string) 
                cred = credentials.Certificate(cert_dict)
                firebase_admin.initialize_app(cred)
            else:
                cred = credentials.Certificate("serviceAccountKey.json")
                firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Firestore接続エラー: {e}")
            return None
    return firestore.client()

# --- Embeddingモデル ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- 共通検索関数 (キャッシュ & バッチ処理対応版) ---
@st.cache_data(ttl=600)  # 👈 10分間キャッシュして、無駄な再検索を防ぐ
def search_documents(query, target_categories, top_k=5):
    """
    クエリに基づいてFirestoreからドキュメントを検索し、類似度順に返す共通関数
    """
    # キャッシュ関数内では st.cache_resource の関数を直接呼ぶとエラーになることがあるため再取得
    #db = firestore.client() 
    db = setup_firestore()
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    if not db or not model: return []

    # 1. クエリのベクトル化
    query_embedding = model.encode(query)
    query_dim = len(query_embedding) # 384
    
    # 2. 候補ドキュメントの取得
    all_docs = []
    try:
        # 🔥  カテゴリが多い場合の「全件取得」をやめ、10個ずつに分けて検索する (Read削減)
        if len(target_categories) > 10:
            # 10個ずつの塊（チャンク）に分割
            chunks = [target_categories[i:i + 10] for i in range(0, len(target_categories), 10)]
            
            for chunk in chunks:
                # チャンクごとに検索を実行
                chunk_ref = db.collection("tech_docs").where("category", "in", chunk).stream()
                
                for doc in chunk_ref:
                    data = doc.to_dict()
                    embedding = data.get('embedding')
                    
                    # 次元チェック
                    if embedding and len(embedding) == query_dim:
                        data['doc_id'] = doc.id
                        all_docs.append(data)
        else:
            # 10個以下の場合は通常通り検索
            query_ref = db.collection("tech_docs").where("category", "in", target_categories).stream()
            
            for doc in query_ref:
                data = doc.to_dict()
                embedding = data.get('embedding')
                
                if embedding and len(embedding) == query_dim:
                    data['doc_id'] = doc.id
                    all_docs.append(data)
                
        if not all_docs:
            return []

        # 3. コサイン類似度計算
        doc_embeddings = np.array([doc['embedding'] for doc in all_docs])
        similarities = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings).flatten()
        
        # 4. 上位抽出
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [all_docs[i] for i in top_indices]
        
        return results

    except Exception as e:
        # キャッシュ内でst.errorを使うとUIが崩れることがあるためprint推奨だが、一旦そのまま
        # st.error(f"検索エラー: {e}") 
        print(f"Search Error: {e}")
        return []

# --- 既存のRAG検索ロジック (単発モード用) ---
def run_rag_search(query, selected_categories, top_k=5):
    top_docs = search_documents(query, selected_categories, top_k=top_k)
    
    if not top_docs:
        return {"answer": "データが見つかりません。", "sources": [], "context": "", "meta_context": "", "raw_docs": []}

    context_text = "\n\n---\n\n".join([doc.get('content', '') for doc in top_docs])
    
    meta_context_list = []
    for doc in top_docs:
        title = doc.get('title', 'No Title')
        summary = doc.get('summary_section', '(要約なし)')
        analysis = doc.get('analysis_section', '(分析なし)')
        meta_context_list.append(f"■事例: {title}\n[要約]\n{summary}\n[分析]\n{analysis}")
    meta_context = "\n\n".join(meta_context_list)
    
    try:
        client = anthropic.Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])
        prompt = f"""
        あなたは家族向け技術トレンド相談エキスパートです。以下の技術情報を参考に、質問に回答してください。
        【技術情報】
        {context_text}
        【質問】
        {query}

        【回答形式】
        - 簡潔で分かりやすく
        - 必ず具体的な技術名と出典（文書タイトル）を挙げる
        """
        
        response = client.messages.create(
            #model="claude-3-haiku-20240307",  # 👈 旧version 2026/4/20まで
            model="claude-haiku-4-5-20251001", 
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        sources = [doc.get('title', '不明') for doc in top_docs]
        
        return {
            "answer": response.content[0].text,
            "sources": sources,
            "context": context_text,
            "meta_context": meta_context,
            "raw_docs": top_docs
        }
    except Exception as e:
        return {"answer": f"エラー: {e}", "sources": [], "context": "", "meta_context": "", "raw_docs": []}

    # --- その他のヘルパー関数 ---
def get_document_by_id(doc_id):
    db = setup_firestore()
    if not db: return None
    try:
        clean_id = doc_id.replace(".md", "")
        doc_ref = db.collection("tech_docs").document(clean_id)
        doc = doc_ref.get()
        if doc.exists: return doc.to_dict()
        else: return None
    except Exception as e:
        st.error(f"ドキュメント取得エラー: {e}")
        return None

def call_claude_json(prompt):
    client = anthropic.Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])
    try:
        response = client.messages.create(
            #model="claude-3-haiku-20240307",  # 👈 旧version 2026/4/20まで
            model="claude-haiku-4-5-20251001", 
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text
        s_idx = content.find("{")
        e_idx = content.rfind("}")
        if s_idx != -1 and e_idx != -1:
            return json.loads(content[s_idx:e_idx+1], strict=False)
        else: return None
    except Exception as e:
        st.error(f"AI生成エラー: {e}")
        return None

def render_mermaid(graph_code):
    graphbytes = graph_code.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    url = f"https://mermaid.ink/img/{base64_string}"
    st.image(url, use_container_width=True)

def render_mermaid_html(code):
    html_code = f"""
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true }});
    </script>
    <div class="mermaid">{code}</div>
    """
    components.html(html_code, height=600, scrolling=True)
# ==========================================
# 🛡️ HOOK機構 (Scope Guard / Deadlock / Exit)
# ==========================================
# カテゴリごとの「概念定義（アンカーテキスト）」
# 具体的な単語を網羅する必要はありません。「どういう分野か」を文章で書けばOKです。
TOPIC_ANCHORS = {
    "Strategy_Mgmt": """
    企業経営、ビジネス戦略、組織マネジメント、リーダーシップ、
    マーケティング、意思決定、心理学、コーチング、チームビルディング、
    SWOT分析、KPI設定、キャリア開発に関する話題、
    現代社会（SNS、デジタル環境）における精神性、マインドセット、価値観の在り方、
    思想哲学のビジネス・実生活への応用。
    """,
    
    "Tech_Singularity": """
    コンピュータ科学、人工知能(AI)、半導体工学、物理学、
    ハードウェア、ソフトウェア開発、プログラミング、アルゴリズム、
    シンギュラリティ、GPU、データセンター、エネルギー技術,
    L0(物理層)、L1(知能層)、L2、L3、L4、アーキテクチャ、統合技術に関する話題。
    """,
    
    "Life_Scaling": """
    個人の健康管理、メンタルヘルス、睡眠、食事、運動、
    人間関係の悩み、コミュニケーション、感情のコントロール、
    ライフハック、習慣形成、自己管理に関する話題。
    """
}

@st.cache_data
def get_topic_anchor_embedding(mode_key):
    """指定されたモードの概念定義をベクトル化"""
    model = load_embedding_model()
    # 定義文を取得（なければTechをデフォルトに）
    anchor_text = TOPIC_ANCHORS.get(mode_key, TOPIC_ANCHORS["Tech_Singularity"])
    return model.encode([anchor_text]) # リスト形式で渡す

def pre_hook_scope_guard(query, current_category_group="Tech_Singularity", threshold=0.15):
    """
    【Pre-hook】概念ベクトル判定版
    """
    try:
        # 🛑 0. Bypass Check (会話・救済用)
        # 「わからん」「教えて」などの短い会話コマンドは、ベクトル判定せずに通す。
        # これがないと、話題自体が含まれていないため弾かれてしまう。
        safe_keywords = [
            "わから", "分から", "教えて", "ヒント", "正解", "答え",
            "ありがとう", "こんにちは", "続き", "はい", "いいえ"
        ]
        # 条件: 「キーワードが含まれている」かつ「短文（30文字以下）」なら通す
        # (長文で「鶏肉の作り方を教えて」と言われた場合は弾きたいので、文字数制限をつける)
        if len(query) < 30 and any(kw in query for kw in safe_keywords):
            return True, ""

        model = load_embedding_model()
        
        # 1. ユーザーの質問をベクトル化
        query_vec = model.encode([query])
        
        # 2. 現在のモードの「概念（アンカー）」ベクトルを取得
        anchor_vec = get_topic_anchor_embedding(current_category_group)
        
        # 3. 類似度判定
        # 「質問」と「分野の定義文」がどれくらい似ているか？
        similarity = cosine_similarity(query_vec, anchor_vec)[0][0]
        
        # デバッグ用（調整時に便利）
        # st.toast(f"🕵️ Scope Match: {similarity:.4f} (Threshold: {threshold})")

        if similarity < threshold:
            msg_map = {
                "Strategy_Mgmt": "現在は『戦略・マネジメント』の時間です。",
                "Tech_Singularity": "現在は『技術・科学』の時間です。",
                "Life_Scaling": "現在は『ライフスケーリング』の時間です。"
            }
            prefix = msg_map.get(current_category_group, "")
            # 👇 追加: 将来のために「なぜ弾かれたか（数値）」を残す
            debug_info = f"\n\n*(🕵️ Debug: Similarity **{similarity:.4f}** < Threshold {threshold} | Mode: {current_category_group})*"
            
            return False, f"🚫 **Scope Guard**: {prefix} その話題は学習範囲外のようです。{debug_info}"

        return True, ""
    except Exception as e:
        return True, ""

def check_deadlock_breaker(history, current_prompt):
    """
    【Exception Handler】無限質問ループ（デッドロック）を検知する
    ★最強安全版: エラーが起きてもアプリを落とさない
    """
    try:
        import re  # ここで読み込むので import忘れエラーが起きない

        # 履歴が少なすぎる、または空の場合は何もしない
        if not history or len(history) < 2:
            return False
        
        # 履歴の最後がAIでない場合（万が一のズレ）もスルー
        last_msg = history[-1]
        if last_msg.get("role") != "assistant":
            return False

        last_ai_msg = last_msg.get("content", "")
        
        # 条件1: AIが直前で質問している（? or ？で終わる）
        is_ai_questioning = last_ai_msg.strip().endswith(("?", "？"))
        
        if not is_ai_questioning:
            return False

        # 条件2: 「あきらめ」または「思考停止」の検知
        # パターンA: 明示的な「あきらめ/ヒント要求」キーワード
        give_up_keywords = [
            "わから", "知ら", "無理", "むり", "教えて", "答え", 
            "ギブ", "パス", "ヒント", "help", "give up"
        ]
        is_give_up = any(kw in current_prompt for kw in give_up_keywords)

        # パターンB: 意味のない短文（記号のみ）
        is_meaningless = bool(re.match(r"^[\s\.。、\?？!！]+$", current_prompt))

        # パターンC: 連投（前回と同じ回答を繰り返している）
        is_repeating = False
        if len(history) >= 3:
            last_user_msg = history[-2].get("content", "")
            if current_prompt.strip() == last_user_msg.strip():
                is_repeating = True

        if is_give_up or is_meaningless or is_repeating:
            return True

    except Exception as e:
        # 万が一エラーが起きても、アプリを止めずに「検知なし」として進める
        # print(f"Deadlock Check Error: {e}") # ログ用
        return False

    return False

def post_hook_exit_trigger(response_text):
    """
    【Post-hook】正解・合格を検知して演出を入れる
    修正: 「素晴らしい質問」などで誤爆しないよう、キーワードを厳格化
    """
    success_keywords = [
        # 明確な正解
        "正解です", "その通りです", "合格です", "完璧です",
        "合っています", "合ってます", "正しいです",

        # 理解の確認
        "よく理解", "理解されています", "正しい理解",
        "間違いありません", "正確です",

        # 完了の合図
        "次の単元", "次のステップ", "核心を突いて"
    ]
    
    # "素晴らしい" 単体だと "素晴らしい質問" に反応するので削除
    # "良い視点" も "良い視点ですが..." と続く場合があるので削除
    
    for kw in success_keywords:
        if kw in response_text:
            # 念のための否定形チェック（「正しくないです」「合っていません」を除外）
            if "正しくない" not in response_text and "合っていません" not in response_text:
                return True
    return False

def socratic_validation(response_text, level=1):
    """
    レベルに応じた検閲：レベルが上がるほど短く、鋭い問いを強いる
    """
    # レベルごとの最大文字数設定
    limits = {1: 160, 2: 120, 3: 80}
    max_len = limits.get(level, 160)

    # 1. 文字数検閲
    if len(response_text) > max_len:
        feedback = f"回答が長すぎます（現在{len(response_text)}文字）。情報の贅肉を削ぎ落とし、{max_len}文字以内で本質的な『問い』に絞ってください。" 
        return False, feedback
        
    # 2. 構文検閲
    if "？" not in response_text[-10:] and "?" not in response_text[-10:]:
        feedback = "教えっぱなしは禁止です。最後は必ずユーザーへの『問いかけ』で締めてください。" 
        return False, feedback
        
    return True, None


# --- 拡張機能 (名刺、日記、思考展開、マップ) ---
def generate_future_career(topic):
    prompt = f"""
    あなたは2035年のキャリアコンサルタントです。トピック: '{topic}' に基づいて、未来的でかっこいい架空の職業プロフィールを作成してください。
    【重要】日本語で出力。
    Output format (JSON): {{ "job_title": "英語名 / 日本語名", "estimated_salary": "15,000,000 JPY", "required_skills": ["スキル1", "スキル2"], "mission": "ミッション" }}
    Only output the JSON.
    """
    return call_claude_json(prompt)

def generate_future_diary(topic):
    prompt = f"""
    あなたは小説家です。2035年を舞台に、'{topic}' が日常になった世界のショートショート日記を書いてください。
    【重要】日本語で出力。
    Output format (JSON): {{ "date": "2035年X月X日", "title": "タイトル", "author_profile": "属性", "content": "本文..." }}
    Only output the JSON.
    """
    return call_claude_json(prompt)

def generate_thought_expansion(topic, mode, meta_context=""):
    base_inst = f"以下は検索された技術資料の要約と分析データです。\n【参照データ】\n{meta_context}" if meta_context else ""
    instructions = {
        "abstract": "参照データから、共通する成功法則や構造的な強みを抽出し、上位概念を言語化してください。",
        "concrete": "参照データの強みを活かして、2030年の社会課題解決に向けた具体的なアプローチを提案してください。",
        "analogous": "参照データの成功ロジックを、全く異なる異分野に転用するアイデアを出してください。"
    }
    titles = {"abstract": "⬆️ 抽象化", "concrete": "⬇️ 具体化", "analogous": "↔️ 横展開"}
    prompt = f"""
    あなたは技術ストラテジストです。トピック: '{topic}' を分析してください。
    {base_inst}
    指示: {instructions.get(mode, "")}
    【重要】日本語で出力。Output format (JSON): {{ "title": "{titles.get(mode)}", "items": ["洞察1", "洞察2", "洞察3", "洞察4", "洞察5"] }}
    Only output the JSON.
    """
    return call_claude_json(prompt)

def generate_tech_hierarchy(topic):
    client = anthropic.Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])
    prompt = f"Create a hierarchical technology map for: '{topic}'. Extract main keywords. Labels MUST be in Japanese. Output ONLY valid Graphviz DOT code."
    try:
        response = client.messages.create(
            #model="claude-3-haiku-20240307",  # 👈 旧version 2026/4/20まで
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.replace("```graphviz", "").replace("```", "").strip()
    except: return None

# --- ナレッジグラフ ---
@st.cache_data(ttl=3600)
def build_knowledge_graph(target_categories):
    db = setup_firestore()
    if not db: return [], []
    nodes, edges, existing_nodes = [], [], set()
    docs = db.collection("tech_docs").select(['title', 'category', 'tags']).stream()
    for doc in docs:
        d = doc.to_dict()
        if d.get('category') in target_categories:
            doc_id = d.get('title', 'No Title')
            tags = d.get('tags', [])
            if doc_id not in existing_nodes:
                nodes.append(Node(id=doc_id, label=doc_id, size=15, color="#4F8BF9", shape="dot"))
                existing_nodes.add(doc_id)
            for tag in tags:
                tag_id = f"tag_{tag}"
                if tag_id not in existing_nodes:
                    nodes.append(Node(id=tag_id, label=tag, size=10, color="#FF6B6B", shape="diamond"))
                    existing_nodes.add(tag_id)
                edges.append(Edge(source=doc_id, target=tag_id, color="#DDDDDD"))
    return nodes, edges

def build_search_graph(doc_list):
    nodes, edges, existing_nodes = [], [], set()
    for d in doc_list:
        doc_id = d.get('title', 'No Title')
        tags = d.get('tags', [])
        if doc_id not in existing_nodes:
            nodes.append(Node(id=doc_id, label=doc_id, size=20, color="#4F8BF9", shape="dot"))
            existing_nodes.add(doc_id)
        for tag in tags:
            tag_id = f"tag_{tag}"
            if tag_id not in existing_nodes:
                nodes.append(Node(id=tag_id, label=tag, size=12, color="#FF6B6B", shape="diamond"))
                existing_nodes.add(tag_id)
            edges.append(Edge(source=doc_id, target=tag_id, color="#999999"))
    return nodes, edges

@st.cache_data(ttl=600)
def get_all_data_as_df():
    db = setup_firestore()
    if not db: return pd.DataFrame()
    docs_list = []
    for doc in db.collection("tech_docs").stream():
        d = doc.to_dict()
        docs_list.append({"Title": d.get('title', ''), "Category": d.get('category', '')})
    return pd.DataFrame(docs_list)

# ==========================================
# 🔐 認証 & サイドバー設定
# ==========================================

def check_password():
    input_pass = st.session_state.get("password_input")
    authorized_users = st.secrets.get("user_passwords", {})
    for username, password in authorized_users.items():
        if input_pass == password:
            del st.session_state["password_input"]
            st.session_state["current_user"] = username
            return True
    if input_pass == st.secrets.get("APP_PASSWORD"):
        del st.session_state["password_input"]
        st.session_state["current_user"] = "Family Member"
        return True
    return False

if "password_correct" not in st.session_state: st.session_state["password_correct"] = False
if "current_user" not in st.session_state: st.session_state["current_user"] = "Guest"

if not st.session_state["password_correct"]:
    st.title("⚔️ CAREER DATA VAULT: AUTH")
    st.markdown("##### 次世代戦略AIへアクセスするには、認証が必要です。")
    with st.form("login_form"):
        st.text_input("パスワード", type="password", key="password_input")
        if st.form_submit_button("Login"):
            if check_password():
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error('パスワードが間違っています。')
    st.stop()

# --- サイドバー構成 ---
st.sidebar.title("🔧 Control Panel")
user_name = st.session_state.get("current_user", "Guest")
st.sidebar.caption(f"Login as: **{user_name}**")

if st.sidebar.button("ログアウト", key='logout_top'):
    st.session_state["password_correct"] = False
    st.session_state["current_user"] = None
    st.rerun()

# モード選択
app_mode = st.sidebar.radio(
    "モード選択",
    [
        "🔍 AI検索 (単発RAG)",
        "💬 AIチャット (対話モード)",
        "🌍 シラバス図書館 (Community)",
        "🎚️ Life Scaling",
        "🗺️ カテゴリ別ガイド",
        "📚 データカタログ一覧",
        "🕸️ ナレッジグラフ",
        "📖 プロジェクト概要",
        "👨‍💻 開発ワークフロー"
    ],
    key="navigation"
)
# --- サイドバー：ソクラテス・レベル設定 ---
if app_mode == "💬 AIチャット (対話モード)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧘 Socratic Tuning")
    st.session_state.socratic_level = st.sidebar.select_slider(
        "ソクラテス・レベル",
        options=[1, 2, 3],
        value=st.session_state.get("socratic_level", 1),
        help="レベルが上がるほど、AIは答えを教えなくなり、あなたに思考を強います。"
    )
    # レベルに応じたラベル表示
    level_labels = {1: "🟢 L1: 伴走 (ヒントあり)", 2: "🟡 L2: 沈黙 (比喩・問い)", 3: "🔴 L3: 鉄仮面 (問いのみ)"}
    st.sidebar.caption(f"現在の設定: **{level_labels[st.session_state.socratic_level]}**")

CATEGORY_GUIDES = {
    "Management (体系)": "Management_Start_Guide",
    "Management (学習)": "learning_path_guide",
    "Strategy Design (体系/学習)": "Strategy_Design_Guide",
    "Strategy Design (学習Navigation)": "Strategy_Design_Navigation",
    "Singularity (体系)": "singularity_master_navi",
    "Singularity (学習)": "learning_path_singularity",
    }

CATEGORY_HIERARCHY = {
    "🧠 Strategy & Management": {
        "Strategy Design (戦略デザイン)": "Strategy_Design",
        "[Mgmt] マネジメント(全層統合)": "Management",
        "Articles: 心理・コーチング・脳科学": "Psychology",
    },
    "📊 Market & Future Trends": {
        "次世代発電技術": "次世代発電",
        "Singularity (技術的特異点)": "Singularity",
    },
    "💻 Engineering & Tech": {
        "Articles: AI Info": "AIinfo",
        "Articles: Python & Web": "Python_Web",
        "Articles: Quality & Security": "Quality_Security",
        "Articles: Semiconductor": "Semiconductor",
        "Articles: Tips": "Tips",
    }
}

st.sidebar.markdown("---")

# --- フィルタリングUI ---
selected_categories = []
if app_mode not in ["📖 プロジェクト概要", "👨‍💻 開発ワークフロー", "🗺️ カテゴリ別ガイド"]:
    header_text = "🔍 検索ソース選択" if "AI" in app_mode else "📂 表示カテゴリ選択"
    st.sidebar.subheader(header_text)

    all_cat_ids = [cid for items in CATEGORY_HIERARCHY.values() for cid in items.values()]
    col_all, col_none = st.sidebar.columns(2)
    if col_all.button("全選択", key="all_cat", use_container_width=True):
        for cid in all_cat_ids: st.session_state[f"cat_{cid}"] = True
    if col_none.button("全解除", key="no_cat", use_container_width=True):
        for cid in all_cat_ids: st.session_state[f"cat_{cid}"] = False

    if "cat_states" not in st.session_state: st.session_state.cat_states = {}

    for group_name, items in CATEGORY_HIERARCHY.items():
        # ★追加: 「Strategy & Management」のグループだけ初期値をTrueにする 20260325
        is_default = (group_name == "🧠 Strategy & Management")

        with st.sidebar.expander(f"📁 {group_name}", expanded=False):
            # valueを is_default に変更
            if st.checkbox(f"全 {group_name} を選択", value=is_default, key=f"group_{group_name}"):
                for label, category_id in items.items():
                    if st.checkbox(label, value=True, key=f"cat_{category_id}"):
                        selected_categories.append(category_id)
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ 検索パラメータ設定")
    search_top_k = st.sidebar.slider(
        "参照する文書数 (Top-K)",
        min_value=3,
        max_value=30,
        value=10,
        help="AIが回答を作成する際に、データベースから取得する関連文書の数です。"
    )

# ==========================================
# 🚀 1. 🔍 AI検索 (単発RAG)
# ==========================================
if app_mode == "🔍 AI検索 (単発RAG)":
    st.title("🧬 NEXT-GEN CAREER BRAIN")
    try:
        st.image("tech-trend-rag-family.jpg", caption="Concept: The Future Career Exploring System", use_container_width=True)
    except FileNotFoundError:
        pass
    st.markdown("---")
    st.markdown("##### **[ACCESS GRANTED]** KNOWLEDGE SYSTEM READY FOR QUERY.")

    st.markdown("#### 🔌 System Architecture")
    render_mermaid("""
    graph LR
        User((User/Query)) -->|Search| DB[(Vector DB)]
        DB -->|Retrieval| AI[[Gen-AI Claude]]
        User -->|Context| AI
        AI -->|Generation| Output>Output Result]
        style User fill:#e8f0fe,stroke:#333
        style DB fill:#e6f3ff,stroke:#00f
        style AI fill:#ffebee,stroke:#f00
        style Output fill:#d4edda,stroke:#333
    """)
    st.markdown("---")

    if "rag_result" not in st.session_state: st.session_state.rag_result = None
    if "last_query" not in st.session_state: st.session_state.last_query = ""
    if "thought_expansion" not in st.session_state: st.session_state.thought_expansion = None
    if "career_card" not in st.session_state: st.session_state.career_card = None
    if "future_diary" not in st.session_state: st.session_state.future_diary = None

    query = st.text_area("Enter Your Question ...🤣日本語でええよ🤣", height=100)

    if st.button("🔍 Research Techs ", type="primary", key='rag_search_button'):
        if not selected_categories:
            st.error("⚠️ 検索対象ソースが選択されていません。")
        elif query:
            st.session_state.thought_expansion = None
            st.session_state.career_card = None
            st.session_state.future_diary = None
            with st.spinner("Analyzing 700 Data Feeds... Standby for Analysis."):
                st.session_state.rag_result = run_rag_search(query, selected_categories, top_k=search_top_k)
                st.session_state.last_query = query
        else:
            st.error("質問を入力してください。")

    if st.session_state.rag_result:
        result = st.session_state.rag_result
        if isinstance(result, dict):
            st.markdown(f"**💡 回答**\n\n{result['answer']}")
            st.markdown("---")
            sources_str = ', '.join(result['sources']) if result['sources'] else "なし"
            st.markdown(f"**📚 参照された資料:** {sources_str}")
            with st.expander("📄 参照データ（原文・抽出メタデータ）を確認する"):
                st.code(result['context'], language="markdown")
                if result.get('meta_context'):
                    st.caption("▼ 思考エレベーター用抽出データ")
                    st.code(result['meta_context'], language="markdown")

            st.markdown("---")
            st.subheader("🕸️ Search Result Knowledge Graph")
            with st.expander("🔍 検索結果の関係性を可視化する (Dynamic Graph)", expanded=True):
                raw_docs = result.get('raw_docs', [])
                if raw_docs:
                    s_nodes, s_edges = build_search_graph(raw_docs)
                    if len(s_nodes) > 0:
                        st.info(f"Visualizing: {len(raw_docs)} Docs | {len(s_nodes)} Nodes")
                        s_config = Config(width="100%", height=400, directed=False, physics=True, collapsible=True)
                        agraph(nodes=s_nodes, edges=s_edges, config=s_config)
                    else: st.warning("グラフデータを作成できませんでした。")
                else: st.warning("可視化できるデータがありません。")

            st.markdown("---")
            st.subheader("💡 Deep Dive & Expansion")
            c1, c2, c3 = st.columns(3)
            meta_context = result.get('meta_context', '')
            with c1:
                if st.button("⬆️ 抽象化", key="btn_abs", use_container_width=True):
                    with st.spinner("Thinking Macro..."):
                        st.session_state.thought_expansion = generate_thought_expansion(st.session_state.last_query, "abstract", meta_context)
            with c2:
                if st.button("⬇️ 具体化", key="btn_con", use_container_width=True):
                    with st.spinner("Thinking Micro..."):
                        st.session_state.thought_expansion = generate_thought_expansion(st.session_state.last_query, "concrete", meta_context)
            with c3:
                if st.button("↔️ 横展開", key="btn_ana", use_container_width=True):
                    with st.spinner("Connecting Dots..."):
                        st.session_state.thought_expansion = generate_thought_expansion(st.session_state.last_query, "analogous", meta_context)

            if st.session_state.thought_expansion:
                d = st.session_state.thought_expansion
                st.markdown(f"#### {d.get('title', 'Analysis')}")
                for item in d.get('items', []): st.write(f"• {item}")

            st.markdown("")
            if st.button("🕸️ 技術体系マップを表示する", key="btn_map", use_container_width=True):
                with st.spinner("Mapping..."):
                    dot = generate_tech_hierarchy(st.session_state.last_query)
                    if dot: st.graphviz_chart(dot)

            st.markdown("---")
            st.subheader("🚀 2035 Vision Simulation")
            ec1, ec2 = st.columns(2)
            with ec1:
                if st.button("🃏 未来の名刺", key="btn_card", use_container_width=True):
                    with st.spinner("Designing..."):
                        st.session_state.career_card = generate_future_career(st.session_state.last_query)
            with ec2:
                if st.button("📖 未来の日記", key="btn_diary", use_container_width=True):
                    with st.spinner("Writing..."):
                        st.session_state.future_diary = generate_future_diary(st.session_state.last_query)

            if st.session_state.career_card:
                c = st.session_state.career_card
                st.success("✅ 2035 Career Prediction")
                with st.container(border=True):
                    col_img, col_txt = st.columns([1, 3])
                    with col_img: st.image("https://img.icons8.com/fluency/96/future.png", width=80)
                    with col_txt:
                        st.markdown(f"### {c.get('job_title', 'Unknown Job')}")
                        st.metric(label="想定年収 (2035)", value=c.get('estimated_salary', '---'))
                    st.write(f"**Mission:** {c.get('mission', '')}")
                    st.write(f"**Skills:** {', '.join(c.get('required_skills', []))}")

            if st.session_state.future_diary:
                d = st.session_state.future_diary
                st.info("✅ 2035 Daily Log")
                with st.container(border=True):
                    st.markdown(f"### 📖 {d.get('title', 'Diary')}")
                    st.caption(f"📅 {d.get('date', '')} | ✍️ {d.get('author_profile', '')}")
                    st.write(d.get('content', ''))

# ==========================================
# 💬 2. AIチャット (対話モード) - Multi-Modal (Teaching & Coaching)
# ==========================================
elif app_mode == "💬 AIチャット (対話モード)":
    # タイトルと画像の配置
    col_cat, col_title = st.columns([1, 5])
    with col_cat:
        st.image("https://img.icons8.com/fluency/96/cat.png", width=70)
    with col_title:
        st.title("AI Knowledge Chat")

    # --- 履歴クリアボタン ---
    col_empty, col_clear = st.columns([3, 1])
    with col_clear:
        if st.button("🗑️ 会話履歴をクリア", key="clear_chat_history", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_syllabus = None
            st.session_state.summary_context = "" 
            # chat_styleはクリアせず維持する（ユーザーの好みなので）
            st.rerun()

    # --- セッション情報の初期化 ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "こんにちは！戦略デザインや技術トレンドについて、何でも聞いてください🐱"}]
    if "current_syllabus" not in st.session_state:
        st.session_state.current_syllabus = None
    if "summary_context" not in st.session_state:
        st.session_state.summary_context = "" 
    # 🔥 AIのスタンス（チャットスタイル）の初期化
    if "chat_style" not in st.session_state:
        st.session_state.chat_style = "Teacher"

    # -------------------------------------------
    # 🎚️ AIスタンス選択 (Teaching vs Coaching)
    # -------------------------------------------
    # ここで「解説役」か「コーチ役」かを選ばせます
    st.markdown("##### 🤖 AIのスタンス選択")
    style_mapping = {
        "Teacher": "👨‍🏫 Teacher Mode (解説・知識重視)",
        "Coaching": "🧘 Coaching Mode (気づき・思考重視)"
    }
    
    selected_style_key = st.radio(
        "学習の進め方を選んでください:　(注意）チャット途中で変更すると人格が豹変します！ コンテキスト慣性という文脈から生じる人格の迷いをここでは断ち切っています。",
        options=["Teacher", "Coaching"],
        format_func=lambda x: style_mapping[x],
        horizontal=True,
        key="style_radio",
        index=0 if st.session_state.chat_style == "Teacher" else 1
    )

    #  モードが変わったら、AIに「キャラ変」を意識させるシステムメッセージを注入
    # (既に chat_style が設定されており、かつ今回変更された場合)
    if "chat_style" in st.session_state and st.session_state.chat_style != selected_style_key:
        st.session_state.chat_style = selected_style_key
        # 会話履歴にシステム通知を強制挿入して、以前のキャラを引きずらせないようにする
        st.session_state.messages.append({
            "role": "system",
            "content": f"【システム通知】これよりAIのモードが『{selected_style_key}』に変更されました。以前の振る舞いを捨て、新しい役割に徹してください。"
        })
    # (まだ設定されていない場合 = 初回)
    elif "chat_style" not in st.session_state:
        st.session_state.chat_style = selected_style_key

    # -------------------------------------------
    # 🛠️ Helper: 会話履歴の圧縮関数
    # -------------------------------------------
    def compress_history(messages):
        KEEP_COUNT = 10 
        if len(messages) <= KEEP_COUNT: return messages, ""
        to_summarize = messages[:-KEEP_COUNT]
        to_keep = messages[-KEEP_COUNT:]
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in to_summarize])
        try:
            client = anthropic.Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])
            resp = client.messages.create(
                #model="claude-3-haiku-20240307",  # 👈 旧version 2026/4/20まで
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                system="以下の会話ログを、文脈がわかるように要約してください。",
                messages=[{"role": "user", "content": history_text}]
            )
            return to_keep, resp.content[0].text
        except: return messages, ""

    # -------------------------------------------
    # 🧰 Memory & Study Tools (保存・ロード・学習計画・共有)
    # -------------------------------------------
    with st.expander("🧰 Memory & Study Tools (保存・共有・学習計画)", expanded=False):
        # 🔥 タブを1つ追加: "🌍 共有"
        tab_guide, tab_study, tab_save, tab_load, tab_share = st.tabs(["ℹ️ 使い方", "📝 勉強プラン作成", "💾 セーブ", "📂 ロード", "🌍 共有"])

        # --- Tab 1: 使い方ガイド ---
        with tab_guide:
            st.markdown("""
            ### 📖 Memory & Study Tools 使用ガイド

            #### 🔰 1. 基本的な使い方 (単発利用)
            学習計画を作らず、その場限りの壁打ちをする場合。
            1. **チャット**: 自由に会話します。
            2. **終了**: サイドバーの `🗑️ 会話履歴をクリア` を押してリセットします。

            ---

            #### 🎓 2. 学習モード (継続学習)
            テーマを決めて体系的に学びたい場合に使います。

            **[Step 1: 学習開始]**
            1. 会話で学びたいテーマをAIに伝えます。
            2. `📝 勉強プラン作成` タブで **「🔄 シラバスを生成/更新する」** を押します。
            👉 「🎓 Study Mode」バナーが表示され、シラバスモードになります。

            **[Step 2: スタンス（AIの接し方）を選択]**
            画面上のラジオボタンで、今の気分に合わせてモードを選んでください。
            * **👨‍🏫 Teacher**: 分からないことを解説してほしい時。
            * **🧘 Coaching**: 自分の考えを整理したい時。

            **[Step 3: 共有 (New!)]**
            素晴らしいシラバスができたら、`🌍 共有` タブからライブラリに登録しましょう。
            あなたのナレッジが、他の誰かの役に立ちます。

            **[Step 4: 中断と保存]**
            1. `💾 セーブ` タブで保存します。
            2. サイドバーの `🗑️ 会話履歴をクリア` で終了します。

            **[Step 5: 再開 (Load)]**
            `📂 ロード` タブでJSONファイルを上げ **「🔄 復元を実行」** します。
            """)

        # --- Tab 2: 勉強プラン作成 (根拠付き生成版) ---
        with tab_study:
            st.caption("会話履歴と登録文書を分析し、学習カリキュラムを生成します。")

            # Step 1: シラバスの生成・更新
            if st.button("🔄 シラバスを生成/更新する"):
                if len(st.session_state.messages) < 2:
                    st.warning("まずはチャットで「〜について学びたい」と相談してください！")
                elif not selected_categories:
                    st.error("⚠️ サイドバーで検索対象のカテゴリを選択してください（全選択推奨）。")
                else:
                    with st.spinner("1. Analyzing Topic..."):
                        # A. 会話から検索クエリを抽出
                        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-10:]])
                        topic_resp = call_claude_json(f"""
                        以下の会話から「ユーザーが学びたい中心テーマ」を単語で抽出してください。
                        会話: {history_text}
                        Format JSON: {{ "query": "検索キーワード" }}
                        """)
                        search_query = topic_resp.get("query", "") if topic_resp else ""
                    
                    with st.spinner(f"2. Searching Knowledge DB ('{search_query}')..."):
                        # B. データベース検索 (RAG)
                        ref_docs = []
                        if search_query:
                            ref_docs = search_documents(search_query, selected_categories, top_k=10)
                        
                        # 参照コンテキストの作成
                        doc_context = ""
                        if ref_docs:
                            doc_context = "\n".join([f"ID: {d.get('title')}\nContent: {d.get('content')[:1000]}" for d in ref_docs])
                        else:
                            doc_context = "(関連文書なし)"

                    with st.spinner("3. Architecting Syllabus..."):
                        # C. 根拠付きシラバス生成
                        try:
                            client = anthropic.Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])
                            
                            sys_msg = f"""
                            あなたは「根拠に基づく学習カリキュラム生成エンジン」です。
                            会話履歴と、ユーザーのデータベース(Reference Docs)を突き合わせ、最適なシラバスを作成してください。

                            【重要: ソース明示ルール】
                            各章やトピックについて、情報の出処を以下の記号で明記してください。
                            - 登録文書に記述がある場合: タイトルを付記する
                              例: 「第1章: ソクラテス問答法の基礎 (📚Ref: ソクラテス問答法による思考変革)」
                            - 登録文書になく、あなたの一般知識で補完した場合:
                              例: 「第2章: 実践ワーク (🤖General Knowledge - 文書未登録)」

                            【出力形式】
                            - 余計な会話文は省略し、Markdownのシラバスのみを出力してください。
                            - タイトルから書き始めてください。
                            """
                            
                            user_prompt = f"""
                            【会話履歴】
                            {history_text}
                            
                            【Reference Docs (検索結果)】
                            {doc_context}
                            
                            上記を統合し、全5章程度のシラバスを作成してください。
                            特に「Reference Docs」に含まれる概念を優先的に取り入れてください。
                            """

                            resp = client.messages.create(
                                #model="claude-3-haiku-20240307", max_tokens=2000, system=sys_msg,  # 👈 旧version 2026/4/20まで
                                model="claude-haiku-4-5-20251001", max_tokens=2000, system=sys_msg,
                                messages=[{"role": "user", "content": user_prompt}]
                            )
                            st.session_state.current_syllabus = resp.content[0].text
                            st.success("✅ 根拠情報を付与してシラバスを生成しました！")
                            
                        except Exception as e:
                            st.error(f"Error: {e}")

            # Step 2: 確定して学習開始
            if st.session_state.current_syllabus:
                st.markdown("---")
                st.info("📝 シラバスがセットされています。内容を確認してください。")
                st.markdown(st.session_state.current_syllabus) # プレビュー表示
                
                col_start_btn, col_dummy = st.columns([2, 1])
                with col_start_btn:
                    if st.button("🧹 履歴を消して学習を始める", type="primary", use_container_width=True):
                        st.session_state.messages = []
                        st.session_state.summary_context = "" 
                        welcome_msg = "✅ **学習モードを開始します**\n\nシラバスに基づき学習を進めます。\n(📚Ref)がついている項目は詳細な解説が可能です。\n(🤖General)がついている項目は、必要に応じて文書を追加登録してください。"
                        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
                        st.rerun()

        # --- Tab 3: セーブ ---
        with tab_save:
            st.caption("「会話履歴」と「シラバス」を保存します。")
            save_data = {
                "timestamp": str(datetime.datetime.now()),
                "chat_style": st.session_state.chat_style, 
                "syllabus": st.session_state.current_syllabus,
                "summary_context": st.session_state.summary_context,
                "history": st.session_state.messages
            }
            save_json = json.dumps(save_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 JSONファイルで保存",
                data=save_json,
                file_name=f"chat_backup_{datetime.date.today()}.json",
                mime="application/json"
            )

        # --- Tab 4: ロード ---
        with tab_load:
            st.caption("バックアップを復元します。")
            uploaded_file = st.file_uploader("JSONファイルをアップロード", type=["json"])
            compress_on_load = st.checkbox("🔄 ロード時に履歴を圧縮する (トークン節約)", value=True)

            if uploaded_file is not None and st.button("🔄 復元を実行"):
                try:
                    loaded_data = json.load(uploaded_file)
                    msgs = loaded_data.get("history", [])
                    st.session_state.current_syllabus = loaded_data.get("syllabus", None)
                    st.session_state.summary_context = loaded_data.get("summary_context", "")
                    st.session_state.chat_style = loaded_data.get("chat_style", "Teacher")
                    
                    if compress_on_load and len(msgs) > 10:
                        with st.spinner("履歴が長いため圧縮しています..."):
                            kept_msgs, new_summary = compress_history(msgs)
                            st.session_state.messages = kept_msgs
                            if st.session_state.summary_context:
                                st.session_state.summary_context += f"\n(追記) {new_summary}"
                            else:
                                st.session_state.summary_context = new_summary
                            st.success(f"圧縮して復元しました (元の件数: {len(msgs)} -> {len(kept_msgs)})")
                    else:
                        st.session_state.messages = msgs
                        st.success("復元完了")
                    st.rerun()
                except Exception as e:
                    st.error(f"Load Error: {e}")

        # --- Tab 5: 🌍 シラバス共有 (New!) ---
        with tab_share:
            st.markdown("### 🌍 ナレッジを共有する")
            st.caption("あなたの作成したシラバスを公開し、他のメンバーが利用できるようにします。")

            if not st.session_state.current_syllabus:
                st.warning("⚠️ 共有するには、まず「📝 勉強プラン作成」でシラバスを作成してください。")
            else:
                st.info("現在アクティブなシラバスをライブラリに登録します。")
                
                # シラバスの内容プレビュー
                with st.expander("登録する内容を確認", expanded=False):
                    st.markdown(st.session_state.current_syllabus)

                # 登録フォーム
                with st.form("share_syllabus_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        # タイトルの自動抽出を試みる（# の行を取得）
                        default_title = "無題のシラバス"
                        for line in st.session_state.current_syllabus.split('\n'):
                            if line.startswith("# "):
                                default_title = line.replace("# ", "").strip()
                                break
                        
                        input_title = st.text_input("タイトル", value=default_title)
                        input_author = st.text_input("作成者名 (任意)", value="Anonymous")
                    
                    with col2:
                        input_category = st.selectbox("カテゴリ", ["技術・RAG", "戦略・デザイン", "マネジメント", "マーケティング", "その他"])
                        input_style = st.selectbox("推奨モード", ["Teacher", "Coaching"], index=0 if st.session_state.chat_style=="Teacher" else 1)

                    input_desc = st.text_area("紹介文 (このシラバスで何が学べる？)", placeholder="例: RAGの基礎から応用まで、PM視点で学べます。", height=80)

                    submit_share = st.form_submit_button("🚀 ライブラリに公開する")

                    if submit_share:
                        if not input_title or not input_desc:
                            st.error("タイトルと紹介文は必須です。")
                        else:
                            # Firestoreへの保存処理
                            try:
                                # 🔥 ここで接続！
                                db = setup_firestore()
                                if db:
                                    doc_ref = db.collection("shared_syllabi").document()
                                    doc_ref.set({
                                        "title": input_title,
                                        "author": input_author,
                                        "category": input_category,
                                        "content": st.session_state.current_syllabus,
                                        "description": input_desc,
                                        "chat_style": input_style, # ロードしやすいキー名に統一
                                        "created_at": firestore.SERVER_TIMESTAMP,
                                        "likes": 0
                                    })
                                    st.success(f"✅ 公開しました！")
                                    st.balloons()
                                else:
                                    st.error("データベースに接続できませんでした。")
                            except Exception as e:
                                st.error(f"保存エラー: {e}")

    # --- メインチャット画面 ---
    
    if st.session_state.current_syllabus:
        with st.status("🎓 Study Mode: Active Syllabus", expanded=False):
            st.markdown(st.session_state.current_syllabus)
            if st.button("学習モード終了"):
                st.session_state.current_syllabus = None
                st.rerun()

    if st.session_state.summary_context:
        with st.expander("🗜️ 過去の会話の要約 (圧縮済みコンテキスト)"):
            st.info(st.session_state.summary_context)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 参照"):
                    for src in msg["sources"]:
                        st.caption(f"{src.get('title')}")

    if prompt := st.chat_input("Ask a question about tech trends..."):
        
        # -------------------------------------------
        # 🛑 1. Pre-hook: Scope Guard (スコープガード)
        # -------------------------------------------
        # ▼▼▼ 追加: 現在のモード判定ロジック ▼▼▼
        current_scope_mode = "Tech_Singularity" # デフォルト

        # selected_categories にマネジメント系が含まれていたらモード変更
        # (簡易判定: IDリストの中にキーワードが含まれるかチェック)
        # ※CATEGORY_HIERARCHYのID定義に基づいています
        for cat_id in selected_categories:
            if cat_id in ["Strategy_Design", "Management", "Psychology"]:
                current_scope_mode = "Strategy_Mgmt"
                break
        # Life Scalingモードなら（アプリモード自体がScalingの場合の保険）
        if app_mode == "🎚️ Life Scaling":
            current_scope_mode = "Life_Scaling"
            
        # ▼▼▼ 変更: 引数にモードを渡す ▼▼▼
        is_valid, error_msg = pre_hook_scope_guard(prompt, current_scope_mode)
        
        if not is_valid:
            # ガード発動時: ユーザー入力は表示するが、AI生成はせず即終了
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.markdown(error_msg)
            st.stop() # ここで処理を中断

        # -------------------------------------------
        # ユーザー入力を表示・保存 (ガード通過後)
        # -------------------------------------------
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # -------------------------------------------
        # ⚠️ 2. Exception Handler: Deadlock Breaker
        # -------------------------------------------
        is_deadlock = False # 初期値をFalseに固定（安全策）
        
        try:
            # 履歴スライスの作成（今回の入力を除く）
            current_history = st.session_state.messages[:-1]
            
            # 関数が存在するかチェックしてから呼び出す
            if 'check_deadlock_breaker' in globals():
                is_deadlock = check_deadlock_breaker(current_history, prompt)
            else:
                # 関数が見つからない場合（定義場所の間違いなど）
                # st.warning("⚠️ Debug: check_deadlock_breaker 関数が見つかりません。")
                is_deadlock = False

        except Exception as e:
            # どんなエラーが起きても、アプリを止めずにエラー内容だけ表示してスルーする
            st.error(f"⚠️ Debug: Deadlock判定でエラーが発生しました: {e}")
            is_deadlock = False

        deadlock_instruction = ""
        if is_deadlock:
            deadlock_instruction = """
            【緊急システム指示】
            現在、ユーザーとの会話が膠着状態（デッドロック）にあります。
            これ以上質問を重ねるのをやめ、検索結果（特に 'requires' や 'dependency'）の情報を使い、
            「答えの核心的なヒント」または「部分的な正解」を提示して、ユーザーを助けてください。
            """
            # デバッグ用
            # st.toast("⚠️ Deadlock Breaker Activated")

        # -------------------------------------------
        # RAG検索 & Prompt構築
        # -------------------------------------------
        with st.chat_message("assistant"):
            # 全カテゴリ検索
            selected_categories_ids = [cid for items in CATEGORY_HIERARCHY.values() for cid in items.values()]
            rag_data = run_rag_search(prompt, selected_categories_ids, top_k=5)
            # 🛑 追加: 検索データが空（エラー含む）の場合は、正直に伝えて中断する
            if not rag_data.get("context"):
                st.error("⚠️ データベースへのアクセス制限（Quota Exceeded）により、資料を検索できませんでした。しばらく時間を置いてから再試行してください。")
                st.stop()  # ここで処理を強制終了

            context = rag_data.get("context", "")
            sources = rag_data.get("sources", [])

            # チャットスタイル指示
            style_instruction = ""
            if st.session_state.chat_style == "Teacher":
                style_instruction = "あなたは解説好きな先生です。論理的に、かつ分かりやすく専門用語を解説してください。"
            else:
                style_instruction = "あなたはソクラテス式のコーチです。すぐに答えを教えず、問いかけを通じてユーザーに気づきを与えてください。"
            # -------------------------------------------
            # プロンプトの組み立て (モード別に指示を明確化)
            # -------------------------------------------
            if st.session_state.chat_style == "Teacher":
                # 👨‍🏫 Teacher: 答えを教えるモード
                style_instruction = """
                あなたは「解説好きな先生」です。
                ユーザーの質問に対し、以下の【参照データ】を元に、論理的かつ分かりやすく解説してください。
                専門用語には噛み砕いた説明を加え、学習をサポートしてください。
                """
            else:
                # 🧘 Coaching: 答えを教えないモード（ここを最強版に修正！）
                style_instruction = """
                あなたは「ソクラテス式の厳しいコーチ」です。
                ユーザーは生徒であり、あなたの目的は「知識を与えること」ではなく「自ら考えさせること」です。

                【対話の絶対ルール】
                1. **解説禁止**: ユーザーが「〜とは？」「教えて」と聞いても、絶対に定義や解説を述べないでください。
                2. **逆質問**: 「あなたはどう思いますか？」「その言葉の定義は何だと考えますか？」と質問で返してください。
                3. **短文回答**: 1回の回答は「100文字以内」に抑えてください。長々と語るのは「解説」とみなされます。
                4. **ヒントの制限**: ユーザーが「降参」や「わからん」と言った時だけ、例外的に【参照データ】からヒントを1つだけ出してください。

                【悪い例】
                User: L0とは？
                You: L0は物理層のことです。具体的には...(長文解説)...

                【良い例】
                User: L0とは？
                You: いきなり正解を求めるとは感心しませんね。
                シンギュラリティ（L4）を実現するために、一番下で支えている「土台」は何だと思いますか？
                """ 
            # ★ここを修正: 過去の要約コンテキストがあれば、プロンプトに含める
            past_context_prompt = ""
            if "summary_context" in st.session_state and st.session_state.summary_context:
                 past_context_prompt = f"""
                 【これまでの会話の要約】
                 以下の文脈を踏まえて回答してください：
                 {st.session_state.summary_context}
                 """
            # 出力形式の制約（Haiku 4.5対応）
            format_constraint = "対話形式で応答してください。Markdownの見出し、箇条書き、コードブロックは使わず、自然な文章で説明してください。回答は400字程度を目安にしてください。"

            # システムプロンプト統合 
            full_system_prompt = f"""
            {style_instruction}

            【出力形式】
            {format_constraint}

            {past_context_prompt}
            
            【参照データ】
            {context}

            {deadlock_instruction}  # 👈 ここにDeadlock時の指示が挿入されます
            """
            # 履歴の圧縮と送信（最新10件のみ）
            messages_for_api = []
            for m in st.session_state.messages[-10:]:
                # 修正: ここで 'system' ロールのメッセージを除外する！
                if m.get("role") != "system":
                    messages_for_api.append(m)

            # LLM呼び出し
            stream_container = st.empty()
            full_response = ""
            
            try:
                client = anthropic.Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])
                with client.messages.stream(
                    #model="claude-3-haiku-20240307",  # 👈 旧version 2026/4/20まで
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1000,
                    system=full_system_prompt,
                    messages=messages_for_api
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        stream_container.markdown(full_response + "▌")
                # 🛡️ 追加: Coachingモード時のみ、長文解説を「再生成」で叩き直す
                if st.session_state.chat_style == "Coaching":
                    is_valid, feedback = socratic_validation(full_response, level=st.session_state.socratic_level)
    
                    if not is_valid:
                        stream_container.empty()
        
                        # レベル別：再生成プロンプト（文字数ではなく「役割」を強調）
                        level_constraints = {
                            1: "【指示】先生として振る舞ってください。結論を教える前に、ヒントを出してユーザーを誘導する質問をしてください。",
            
                            2: "【指示】解説を控えてください。「それは〜のようなものです」といった比喩や、ユーザーの常識を疑うような質問を投げかけてください。",
            
                            3: "【絶対命令】あなたは冷徹なソクラテスです。回答、解説、共感は一切不要です。ユーザーが持っている『思い込み』を刺すような、短い逆質問を1つだけ返してください。"
                        }
                        constraint = level_constraints.get(st.session_state.socratic_level, "")

                        with st.status(f"🤔 Level {st.session_state.socratic_level} の思考を執行中..."):
                            retry_response = client.messages.create(
                                #model="claude-3-haiku-20240307",  # 👈 旧version 2026/4/20まで
                                model="claude-haiku-4-5-20251001",
                                max_tokens=300, # 物理切れを防ぐため余裕を持たせる
                                system=full_system_prompt + f"\n\n【再生成指示】\n{feedback}\n{constraint}\n※文章が途中で切れないよう、必ず完結させてください。",
                                messages=messages_for_api
                            )
                            full_response = retry_response.content[0].text

                # -------------------------------------------
                # 🎉 3. Post-hook: Exit Trigger
                # -------------------------------------------
                if post_hook_exit_trigger(full_response):
                    st.balloons()
                    full_response += "\n\n(✅ **Excellent!** 次の単元へ進みましょう)"
                
                # 最終表示と保存
                stream_container.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # 出典の表示
                if sources:
                    unique_sources = list(set(sources))
                    st.caption(f"📚 参照: {', '.join(unique_sources)}")

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
# ==========================================
# 🗺️ 3. カテゴリ別ガイド
# ==========================================
elif app_mode == "🗺️ カテゴリ別ガイド":
    st.title("🗺️ Knowledge Guide")
    guide_names = list(CATEGORY_GUIDES.keys())
    selected_guide_label = st.selectbox("閲覧したいガイドを選択してください", guide_names)
    target_doc_id = CATEGORY_GUIDES[selected_guide_label]

    with st.spinner("Loading Guide..."):
        doc_data = get_document_by_id(target_doc_id)

    if doc_data:
        st.caption(f"📅 Last Updated: {doc_data.get('date', 'Unknown')}")
        content = doc_data.get('content', '')

        parts = re.split(r'(```mermaid[\s\S]*?```)', content)
        for part in parts:
            if part.strip().startswith("```mermaid"):
                mermaid_code = part.replace("```mermaid", "").replace("```", "").strip()
                st.write("---")
                st.caption("Architecture Diagram")
                render_mermaid_html(mermaid_code)
                st.write("---")
            else:
                if part.strip():
                    st.markdown(part)

        st.markdown("---")
        referenced_docs = re.findall(r'\[\[(.*?)\]\]', content)
        referenced_docs = sorted(list(set(referenced_docs)))

        if referenced_docs:
            st.subheader("📚 関連ドキュメント (Deep Dive)")
            cols = st.columns(3)
            for i, ref_doc_id in enumerate(referenced_docs):
                clean_id = ref_doc_id.replace(".md", "")
                if cols[i % 3].button(f"📄 {clean_id}", key=f"btn_{clean_id}"):
                    st.session_state['selected_ref_doc'] = clean_id

            if 'selected_ref_doc' in st.session_state:
                ref_id = st.session_state['selected_ref_doc']
                with st.spinner(f"Loading {ref_id}..."):
                    ref_data = get_document_by_id(ref_id)
                if ref_data:
                    st.write("---")
                    st.success(f"📖 {ref_id}")
                    with st.container(border=True):
                        st.caption(f"Category: {ref_data.get('category','')} | Layer: {ref_data.get('layer','')}")
                        st.markdown(ref_data.get('content', 'No content'))
    else:
        st.warning(f"ガイドドキュメント '{target_doc_id}' が見つかりませんでした。")

# ==========================================
# 📚 4. データカタログ一覧
# ==========================================
elif app_mode == "📚 データカタログ一覧":
    st.title("📚 Data Catalog")
    df = get_all_data_as_df()
    if not df.empty:
        df_filtered = df[df['Category'].isin(selected_categories)]
        st.info(f"全データ数: {len(df)} 件 / 表示中: {len(df_filtered)} 件")
        st.dataframe(df_filtered, use_container_width=True, hide_index=True)
    else:
        st.warning("データが見つかりません。")

# ==========================================
# 🕸️ 5. ナレッジグラフ
# ==========================================
elif app_mode == "🕸️ ナレッジグラフ":
    st.title("🕸️ Knowledge Graph")
    if not selected_categories:
        st.warning("👈 左側のサイドバーで、可視化したいカテゴリにチェックを入れてください。")
    else:
        with st.spinner("グラフを構築中..."):
            nodes, edges = build_knowledge_graph(selected_categories)
            if len(nodes) > 700:
                st.error(f"⚠️ ノード数が多すぎます ({len(nodes)}件)。ブラウザのフリーズを防ぐため、カテゴリを絞ってください。（推奨: 700件以下）")
            elif not nodes:
                st.warning("選択されたカテゴリにデータが見つかりません。")
            else:
                st.info(f"Nodes: {len(nodes)} | Edges: {len(edges)}")
                config = Config(width="100%", height=600, directed=False, physics=True, hierarchical=False, collapsible=True)
                agraph(nodes=nodes, edges=edges, config=config)

# ==========================================
# 📖 6. プロジェクト概要 & マニュアル
# ==========================================
elif app_mode == "📖 プロジェクト概要":
    st.title("📖 Project Overview & Manual")
    
    st.markdown("""
    ### 🧬 NEXT-GEN CAREER BRAIN とは
    技術トレンド、組織戦略、そして個人のキャリアデザインを統合した、次世代のナレッジベースシステムです。
    **「調べる（Search）」** と **「考える（Chat）」** を使い分け、未来を設計するためのツールです。

    ---
    
    ### 🚀 2つのメインモードの使い分け

    #### 1. 🔍 AI検索 (単発RAG) ＝ 「コックピット / 生成ラボ」
    **「1つのテーマを多角的に分析・シミュレーションしたい時」** に使います。
    
    * **特徴**: 1回の検索で、「要約」「関連グラフ」「思考の拡張」を一画面に展開します。
    * **✨ ここだけの機能**: 
        * **未来シミュレーション**: 検索結果を基に、「2035年の未来日記」や「未来の職業名刺」をワンクリックで生成できます。
        * **思考エレベーター**: 「具体化」「抽象化」ボタンで、思考の解像度を調整できます。
    * **推奨シーン**: 
        * 「この技術、将来どうなる？」と未来を想像したい時。
        * アイデア出しのために、ランダムな刺激が欲しい時。
    
    #### 2. 💬 AIチャット (対話モード) ＝ 「作戦会議室 / 壁打ち」
    **「文脈を踏まえて、深く相談したい時」** に使います。
    
    * **特徴**: 会話のキャッチボールができるため、曖昧な悩みからスタートできます。
    * **✨ ここだけの機能**: 
        * **文脈理解**: 「さっきの話だけど…」「具体的には？」といった指示語が通じます。
        * **コーチング**: AIがあなたの理解度に合わせて解説してくれます。
    * **推奨シーン**: 
        * 「チームがうまくいかない…」といった複雑な悩み相談。
        * 学習した概念について、「自分の理解で合ってる？」と確認したい時。

    ---

    ### 📚 その他の機能

    #### 🗺️ カテゴリ別ガイド
    **「教科書」** です。「戦略デザイン」などのフレームワークを、基礎から体系的に学ぶことができます。
    * まずはここを一通り読んで、全体像を掴むのがおすすめです。

    #### 🕸️ ナレッジグラフ
    **「地図」** です。ドキュメント同士がどう繋がっているか（例：「サバイバルモード」と「スラック」の関係）を可視化します。
    
    ---

    ### 🏗️ System Architecture
    """)
    
    render_mermaid("""
    graph TD
        User[User] --> FE[Streamlit App]
        
        subgraph "Purpose Selection"
            FE --> Search["🔍 Search Mode"]
            FE --> Chat["💬 Chat Mode"]
            FE --> Guide["🗺️ Guide Mode"]
        end

        subgraph "AI Core"
            Chat --> LLM[Claude 3 Haiku]
            Search --> LLM
            LLM <--> VDB[(Firestore Vector DB)]
        end
    """)

    st.info("Powered by: Streamlit, Firebase Firestore, Anthropic Claude 3")

# ==========================================
# 👨‍💻 7. 開発ワークフロー & システムアーキテクチャ
# ==========================================
elif app_mode == "👨‍💻 開発ワークフロー":
    st.title("👨‍💻 System Architecture & Workflow")
    st.caption("本システムの内部ロジックと、開発・運用フローの解説")

    # --- Tab構成 ---
    tab_arch, tab_flow = st.tabs(["🧠 システム設計図 (Logic)", "🛠️ 開発・運用ガイド"])

    with tab_arch:
        st.subheader("🏛️ RAG Architecture: Dense Retrieval System")
        st.markdown("""
        このシステムは、あなたの個人的な脳（Knowledge）を拡張するために設計されています。
        一般的な検索システムとは異なり、**「文脈」と「意味」を重視した3段階のプロセス** で思考をサポートします。
        """)

        st.markdown("#### 📐 全体フロー図")
        
        # 修正: コードを変数に入れ、インデントを左に寄せて記述し、ラベルを""で囲んで安定化
        mermaid_code = """
graph TD
    subgraph Phase1 ["Phase 1: 知識の植え付け (Ingestion)"]
        MD[Markdown Files] -->|"読込 & 解析"| Text[本文・要約抽出]
        MD -->|Metadata| Meta[カテゴリ・タグ]
        Text & Meta -->|結合| VectorText["ベクトル化用テキスト"]
        VectorText -->|"Embedding Model<br>(MiniLM-L12)"| Vector[384次元ベクトル]
        Vector & Meta & Text -->|保存| DB[(Firestore DB)]
    end

    subgraph Phase2 ["Phase 2: 知識の引き出し (Retrieval)"]
        User[User Query] -->|Embedding| Q_Vector[クエリベクトル]
        User -->|UI操作| Filter[カテゴリ選択]
        
        Filter -->|"1. 絞り込み"| DB
        DB -->|"2. 候補取得"| Candidates[候補文書]
        
        Q_Vector & Candidates -->|"3. 類似度計算"| Ranked[類似度順リスト]
        
        Ranked -->|"Top-K カット"| TopDocs[上位文書]
    end

    subgraph Phase3 ["Phase 3: 回答生成 (Generation)"]
        TopDocs -->|全文結合| Context[プロンプトコンテキスト]
        Context & User -->|"LLMへ送信"| Claude[Claude 3 Haiku]
        Claude -->|"文脈理解 & 生成"| Answer[最終回答]
    end
    
    style DB fill:#f9f,stroke:#333,stroke-width:2px
    style Claude fill:#ff9,stroke:#333,stroke-width:2px
"""
        render_mermaid(mermaid_code)

        st.markdown("---")
        
        # 2. 詳細解説 (以下は変更なし)
        st.subheader("🔍 処理プロセスの詳細解説")
        
        with st.expander("Phase 1: 知識の植え付け (Ingestion)", expanded=True):
            st.markdown("""
            **「AIに教科書を渡して、暗記させるフェーズ」です。**
            
            1. **情報の統合**: Markdownファイルの「本文」だけでなく、「タイトル」「タグ」「要約」をすべて結合し、ひとつの意味の塊を作ります。
            2. **ベクトル化 (Embedding)**: 文章を「384次元の数値の羅列」に変換します。これにより、AIはキーワードの一致だけでなく、「意味の近さ」で検索できるようになります。
            3. **保存**: ベクトルデータと元のテキストをセットにして、Firestore（データベース）に格納します。
            """)
        
        # ... (Phase 2, Phase 3 の解説コードはそのまま維持) ...
        with st.expander("Phase 2: 知識の引き出し (Retrieval)", expanded=True):
             st.markdown("""
            **「質問に合わせて、最も参考になる教科書のページを探すフェーズ」です。**
            
            1. **フィルタリング**: ユーザーが選んだカテゴリ（例: 心理学）以外のデータは最初から除外します。これにより、無関係なノイズが混ざるのを防ぎます。
            2. **総当たり類似度計算**: あなたの質問と、データベース内の全文書の「意味の距離（コサイン類似度）」を計算します。
            3. **Top-K抽出**: 最も意味が近い上位（Top-K）の文書だけをピックアップします。このシステムでは、断片的な切り抜きではなく「文書丸ごと」を取得するのが特徴です。
            """)

        with st.expander("Phase 3: 回答生成 (Generation)", expanded=True):
            st.markdown("""
            **「見つけたページを読み込んで、あなたの質問に答えるフェーズ」です。**
            
            1. **コンテキスト構築**: ピックアップした文書の全文を繋ぎ合わせ、AIへの命令文（プロンプト）に組み込みます。
            2. **AIによる読解**: Claude 3 (AI) が、渡された資料をその場で読み込みます。
            3. **回答生成**: 資料の内容を根拠にしつつ、あなたの質問に対する答えを作成します。ここで「ティーチング」や「コーチング」といった人格の調整も行われます。
            """)

        st.markdown("---")
        st.subheader("📊 システムの特性比較")
        
        # 比較テーブル (変更なし)
        df_arch = pd.DataFrame({
            "項目": ["検索手法", "データ単位", "強み", "弱み"],
            "本システム (Tech-Trend-RAG)": [
                "Dense Vector Search (意味検索)",
                "ドキュメント単位 (全文)",
                "文脈が切れない。AIが深い考察を行える。",
                "超長文だとベクトルの特徴が薄まる。"
            ],
            "一般的な検索システム": [
                "Keyword + Vector (Hybrid)",
                "チャンク単位 (500文字程度)",
                "ピンポイントな事実検索に強い。",
                "文脈が分断され、全体像を見失いやすい。"
            ]
        })
        st.table(df_arch)

    with tab_flow:
        st.subheader("🛠️ Developer Workflow")
        st.markdown("コンテンツの追加から反映までの手順です。")

        st.markdown("#### 1. コンテンツの追加・修正")
        st.code("""
# Markdownファイルのヘッダー例 (Frontmatter)
---
title: "新しい技術トレンド"
category: "Market & Future Trends"
tags: ["AI", "2030"]
date: 2025-12-24
---
本文...
        """, language="yaml")

        st.markdown("#### 2. データベース更新 (ベクトル化)")
        st.markdown("ローカル環境でスクリプトを実行し、Firestoreを更新します。")
        st.code("""
# 特定フォルダのみ更新（高速）
python scripts/update_strategy_only.py

# 心理学・コーチング領域の更新
python scripts/update_psychology.py

# 特定ファイル(交渉術など)のみ更新
python scripts/update_negotiation_files.py
        """, language="bash")

        st.markdown("#### 3. アプリへの反映 (Git Deploy)")
        st.code("""
# 1. 変更をステージング
git add .

# 2. コミット
git commit -m "feat: Add new content and update architecture"

# 3. Cloudへデプロイ (Develop)
git push origin develop

# 4. 本番反映 (Main)
git checkout main
git merge develop
git push origin main
git checkout develop
        """, language="bash")

# ==========================================
# 🎚️ 8. Life Scaling
# ==========================================
elif app_mode == "🎚️ Life Scaling":
    st.title("🎚️ Life Scaling Console")
    st.caption("Everything is scaling. Tune your reality.")

    st.markdown("""
    <style>
    section[data-testid="stMain"] div[role="radiogroup"] label {
        font-size: 20px !important;
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #ddd;
    }
    section[data-testid="stMain"] div[role="radiogroup"] label[data-baseweb="radio"]:hover {
        background-color: #e9ecef;
    }
    </style>
    """, unsafe_allow_html=True)

    MODE_MAP = {
        "⚾ Physical Tuning (身体・感覚)": "⚾ Physical Tuning",
        "❤️ Emotional Sync (感情・人間関係)": "❤️ Emotional Sync",
        "🏗️ Structural Design (仕組み・学習)": "🏗️ Structural Design"
    }
    
    selected_label = st.radio("Select Dimension:", list(MODE_MAP.keys()))
    selected_key = MODE_MAP[selected_label]

    settings = {
        "⚾ Physical Tuning": {
            "label_min": "😫 Heavy (ドロドロ)", "label_max": "⚡ Sharp (キレキレ)",
            "persona": "Top Athlete Coach & Biomechanics Expert",
            "context": "スポーツ科学、フロー理論、超回復、自律神経調整",
            "analogy_target": "ビジネスにおける『生産性と休息の管理』",
            "advice_tone": "冷静で簡潔、身体感覚に訴える"
        },
        "❤️ Emotional Sync": {
            "label_min": "🧊 Disconnect (冷戦)", "label_max": "🤝 Resonance (共鳴)",
            "persona": "Professional Mediator & NVC Specialist",
            "context": "心理学、NVC（非暴力コミュニケーション）、愛着理論、マインドフルネス",
            "analogy_target": "顧客やチームとの『エンゲージメント構築』",
            "advice_tone": "受容的で温かい、感情のニーズに焦点を当てる"
        },
        "🏗️ Structural Design": {
            "label_min": "🌪️ Chaos (混乱)", "label_max": "🏛️ Order (整然)",
            "persona": "Strategic Architect & System Thinker",
            "context": "TOC（制約理論）、ボトルネック分析、GTD、断捨離",
            "analogy_target": "肉体の『トレーニング計画と負荷調整』",
            "advice_tone": "論理的で構造的、仕組みに焦点を当てる"
        }
    }
    
    current_setting = settings[selected_key]

    st.markdown("---")
    st.markdown(f"## Current State: <br> **{selected_key}**", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1,1])
    c1.markdown(f"### 0: {current_setting['label_min']}")
    c2.markdown(f"<div style='text-align: right;'><h3>10: {current_setting['label_max']}</h3></div>", unsafe_allow_html=True)

    score = st.slider(
        "今の状態を直感で選んでください",
        min_value=0, max_value=10, value=5,
        format="%d"
    )

    user_input = st.text_input(
        "Memo (Optional)", 
        placeholder="具体的なテーマがあれば入力（例: バッティングの調子、彼女と喧嘩、テスト勉強）- 空欄でもOK"
    )

    if st.button("Analyze & Tune", type="primary", key="scaling_btn", use_container_width=True):
        
        context_str = user_input if user_input else "General context (No specific details provided)"
        
        system_prompt = f"""
        Act as a {current_setting['persona']}.
        The user selected the mode "{selected_key}" and rated their current state as {score}/10.
        Labels: 0={current_setting['label_min']}, 10={current_setting['label_max']}.
        User's specific context: "{context_str}".
        
        Base Knowledge: {current_setting['context']}.
        
        # Task
        1. **Assessment**: Briefly interpret what a score of {score} means in this domain. (Empathize with the state).
        2. **Micro-Step**: Provide ONE simple, actionable "Micro-Step" to raise the score by +1. (Do not aim for 10 immediately).
        3. **👁️ Meta Insight (Crucial)**: 
           - Explain how this advice structurally connects to "{current_setting['analogy_target']}".
           - Start with "💡 Insight: This is actually the same mechanism as..."
           - Show that "Scaling and structure are universal" across sports, feelings, and business.
        
        Output Language: Japanese (Use English for Key Concepts).
        Tone: {current_setting['advice_tone']}.
        """
        
        with st.spinner("Tuning your reality..."):
            try:
                client = anthropic.Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])
                response = client.messages.create(
                    #model="claude-3-haiku-20240307",  # 👈 旧version 2026/4/20まで
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": "Analyze my scale."}]
                )
                
                result_text = response.content[0].text
                st.success("✅ Tuning Complete")
                st.markdown(result_text)
                
            except Exception as e:
                st.error(f"Error: {e}")


# ==========================================
# 🌍 9. シラバス図書館 (Community Library)
# ==========================================
elif app_mode == "🌍 シラバス図書館 (Community)":
    st.title("🌍 Syllabus Library")
    st.caption("組織の集合知にアクセスし、学習プランをインストールします。")

    # --- Req ①: 更新ガイド ---
    with st.expander("ℹ️ シラバスの更新・活用ガイド (How to Update)"):
        st.markdown("""
        **Q. シラバスの内容が古くなったら？**
        1. 対象のシラバスをロードする。
        2. チャットで **「このシラバスを、現在の登録文書を元に最新情報を用いて更新・再構築してください」** と指示。
        3. 生成された新しいシラバスを「🌍 共有」タブから新規登録。
        4. 古いシラバスは、メンテナンスモードから削除。
        """)

    # 1. データ取得
    db = setup_firestore()
    docs = []
    if db:
        try:
            refs = db.collection("shared_syllabi").order_by("created_at", direction=firestore.Query.DESCENDING).stream()
            for r in refs:
                d = r.to_dict()
                d["id"] = r.id
                docs.append(d)
        except Exception as e:
            st.error(f"データ取得エラー: {e}")

    # 2. ツールバー
    col_filter, col_sort = st.columns([3, 1])
    with col_filter:
        cat_filter = st.multiselect("カテゴリ絞り込み", ["技術・RAG", "戦略・デザイン", "マネジメント", "マーケティング", "その他"])
    with col_sort:
        is_maintenance = st.toggle("🛠️ 管理モード", value=False)
    
    st.markdown("---")

    # 3. リスト表示 (Compact Layout)
    if not docs:
        st.info("まだ共有されたシラバスはありません。")
    else:
        count = 0
        for doc in docs:
            if cat_filter and doc.get("category") not in cat_filter:
                continue
            
            count += 1
            doc_id = doc["id"]
            
            # --- カードコンテナ ---
            with st.container(border=True):
                # 上段: タイトルとアクションボタンを横並びに
                c_main, c_btn = st.columns([3, 1])
                
                with c_main:
                    # タイトル & カテゴリ
                    st.subheader(f"📑 {doc.get('title', 'No Title')}")
                    
                    # メタ情報 (1行にまとめる)
                    style = doc.get("chat_style", "Teacher")
                    icon = "🧘" if style == "Coaching" else "👨‍🏫"
                    likes = doc.get("likes", 0)
                    auth = doc.get("author", "Anonymous")
                    date_str = doc.get("created_at").date() if doc.get("created_at") else "-"
                    
                    st.caption(f"**{doc.get('category')}** | {icon} {style} | ❤️ {likes} | 👤 {auth} | 📅 {date_str}")

                with c_btn:
                    # ロードボタン (右側に寄せる)
                    if not is_maintenance:
                        def on_load_click(content, style, title):
                             st.session_state.current_syllabus = content
                             st.session_state.chat_style = style
                             st.session_state.messages = []
                             st.session_state.summary_context = ""
                             welcome_msg = f"✅ **シラバスをロードしました: {title}**\n\nライブラリから共有プランを取り込みました。\nさっそく学習を始めましょう！"
                             st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
                             st.session_state.navigation = "💬 AIチャット (対話モード)"

                        st.button(
                            "📥 学ぶ", 
                            key=f"load_{doc_id}", 
                            type="primary",
                            use_container_width=True,
                            on_click=on_load_click,
                            args=(doc.get("content"), doc.get("chat_style", "Teacher"), doc.get("title"))
                        )

                # 下段: 詳細情報 (折りたたみ)
                # 紹介文もここに収納してスッキリさせる
                with st.expander("ℹ️ 概要・シラバス内容を見る"):
                    st.markdown("**【Introduction】**")
                    st.write(doc.get("description", "紹介文なし"))
                    
                    # いいねボタン (Expanderの中に配置してスペース節約)
                    def on_like_click(did, current_likes):
                        db.collection("shared_syllabi").document(did).update({"likes": current_likes + 1})
                    
                    col_lk, col_dummy = st.columns([1, 4])
                    with col_lk:
                        st.button(f"❤️ いいね ({likes})", key=f"like_{doc_id}", on_click=on_like_click, args=(doc_id, likes))
                    
                    st.markdown("---")
                    st.markdown(doc.get("content", ""))

                # --- メンテナンス機能 ---
                if is_maintenance:
                    st.caption("🔧 管理メニュー")
                    with st.form(key=f"edit_{doc_id}"):
                        new_desc = st.text_area("紹介文修正", value=doc.get("description", ""))
                        if st.form_submit_button("保存"):
                            db.collection("shared_syllabi").document(doc_id).update({"description": new_desc})
                            st.rerun()
                    
                    if st.button("🗑️ 削除", key=f"del_{doc_id}"):
                         st.session_state[f"confirm_{doc_id}"] = True
                    
                    if st.session_state.get(f"confirm_{doc_id}"):
                        st.warning("本当に削除しますか？")
                        if st.button("はい、削除", key=f"yes_{doc_id}"):
                            db.collection("shared_syllabi").document(doc_id).delete()
                            del st.session_state[f"confirm_{doc_id}"]
                            st.rerun()

        if count == 0:
            st.warning("条件に一致するシラバスが見つかりませんでした。")

    # ==========================================
    # 📢 Req ⑤: 要望・改善ボード (BBS)
    # ==========================================
    st.markdown("---")
    st.subheader("📢 シラバス要望・改善ボード")
    st.caption("「こんなシラバスが欲しい」「ここは直して」など、自由に投稿してください。")

    # 投稿フォーム
    with st.form("request_form", clear_on_submit=True):
        req_col1, req_col2 = st.columns([4, 1])
        with req_col1:
            req_content = st.text_input("要望を入力...", placeholder="例: 生成AIのデザインパターンについてのシラバスが欲しいです！")
        with req_col2:
            req_submit = st.form_submit_button("送信 📨")

        if req_submit and req_content:
            if db:
                db.collection("syllabus_requests").add({
                    "content": req_content,
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "likes": 0  # ✨ 初期値を追加
                })
                st.success("投稿しました！")
                st.rerun()

    # 掲示板表示 (最新20件)
    if db:
        try:
            req_refs = db.collection("syllabus_requests").order_by("created_at", direction=firestore.Query.DESCENDING).limit(20).stream()
            req_docs = []
            for r in req_refs:
                d = r.to_dict()
                d["id"] = r.id
                req_docs.append(d)
            
            if req_docs:
                with st.container(border=True):
                    for req in req_docs:
                        req_id = req["id"]
                        msg = req.get("content", "")
                        likes = req.get("likes", 0)
                        ts = req.get("created_at")
                        date_label = ts.date() if ts else ""

                        # レイアウト: [メッセージ (日付)] -- [いいね] -- [削除(管理モードのみ)]
                        # カラム比率調整
                        if is_maintenance:
                            c_msg, c_like, c_del = st.columns([6, 1.5, 1])
                        else:
                            c_msg, c_like = st.columns([7, 1.5])

                        with c_msg:
                            st.text(f"・{msg} ({date_label})")

                        with c_like:
                            # ❤️ いいねボタン
                            def on_bbs_like(rid, current_likes):
                                db.collection("syllabus_requests").document(rid).update({"likes": current_likes + 1})
                            
                            st.button(f"❤️ {likes}", key=f"req_like_{req_id}", on_click=on_bbs_like, args=(req_id, likes), help="この要望に投票する")

                        # 🛠️ 削除ボタン (管理モードがONの時だけ表示)
                        if is_maintenance:
                            with c_del:
                                def on_bbs_delete(rid):
                                    db.collection("syllabus_requests").document(rid).delete()
                                
                                # 誤操作防止のため、ポップオーバーの中に削除実行ボタンを置くスタイル
                                with st.popover("🗑️"):
                                    st.write("削除しますか？")
                                    st.button("はい", key=f"req_del_exec_{req_id}", on_click=on_bbs_delete, args=(req_id,), type="primary")

            else:
                st.caption("まだ投稿はありません。")
        except Exception as e:
            st.caption(f"読み込みエラー: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("ログアウト", key='logout_bottom'):
    st.session_state["password_correct"] = False
    st.session_state["current_user"] = None
    st.rerun()



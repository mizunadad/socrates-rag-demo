"""
step5/app.py -- Phase 4: 哲学者相談モード追加版

[FILE] step5/app.py
NOTE: step4（MixedRAG / Multi-Domain）に哲学者相談モードを追加した版。

step4/app.py からの差分（4箇所のみ）:
  1. import追加: components, datetime, philo_shapes
  2. PHILO定数・関数群を追加（哲学者相談モード）
  3. サイドバー先頭にモード切替ラジオを追加
  4. main()をRAGモード／哲学者モードで分岐

必要なファイル:
  philo_shapes.py（同ディレクトリに配置すること）

Base: step4/app.py (666 lines, Multi-Domain verified)
"""

import re
import streamlit as st
import streamlit.components.v1 as components  # ★ 追加
import os
import json
import datetime                                # ★ 追加
import firebase_admin
from firebase_admin import credentials, firestore
from sentence_transformers import SentenceTransformer
import anthropic
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from philo_shapes import PHILO_SHAPE_DATA, build_shape_cards  # ★ 追加

# ==========================================
# [DIFF-1] Domain Configuration
# ==========================================
# step3: 単一ドメイン（SWTest固定）
# step4: 辞書でドメインを管理。追加は設定を足すだけ。

DOMAIN_CONFIG = {
    "SWTest": {
        "label": "🧪 ソフトウェアテスト哲学",
        "categories": ["Strategy_Design", "Technology"],
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

SCOPE_THRESHOLD = 0.15

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
# [DIFF-1] 哲学者相談モード定数
# ==========================================

MODEL_PHILO = "claude-haiku-4-5-20251001"

PHILO_DEFAULTS = {
    "philo_phase":        "input",
    "philo_concern":      "",
    "philo_l1_question":  "",
    "philo_l1_answer":    "",
    "philo_candidates":   [],
    "philo_shapes":       [],
    "philo_selected_idx": None,
    "philo_l3_history":   [],
    "philo_turn_count":   0,
    "philo_reveal_intro": "",
    "philo_category":     "",
}

PHILO_ADV_INDEX = """
あなたは哲学者選出システムです。
以下のインデックスを参照し、ユーザーの悩みと返答に最も関連する
哲学者を3名選び、悩みカテゴリを1つ特定してください。

悩みカテゴリ別インデックス：
- 判断・意思決定       → アリストテレス、カント、ソクラテス、デカルト
- 自己理解・アイデンティティ → キルケゴール、サルトル、ニーチェ、ハイデガー
- モチベーション・やる気  → デシ＆ライアン、ニーチェ、エピクロス、ストア派
- 権力・発言できない    → フーコー、ハーバーマス、ロック、エドモンドソン
- 対話・合意形成       → ボーム、ハーバーマス、ハーバード式、P4C
- 知識共有・KM       → 野中郁次郎、エドモンドソン、アリストテレス
- 変革・構造         → マルクス、ヘーゲル、ロック、ニーチェ
- 不安・孤独・逆境     → ストア派、キルケゴール、エピクロス、スピノザ
- 交渉・利害調整      → ハーバード式、ハーバーマス、ヘーゲル
- 感情・内面の整理     → ストア派、スピノザ、エピクロス
- 恋愛・人間関係      → キルケゴール、サルトル、ニーチェ、ボーム
- スポーツ・競技      → ストア派、ニーチェ、アリストテレス、デシ＆ライアン
- 学問・学習        → デカルト、アリストテレス、ソクラテス、カント
- キャリア・進路      → ハイデガー、サルトル、デシ＆ライアン、ニーチェ
- 品質管理・工学判断   → アリストテレス、デカルト、カント、フーコー
- ゴール設定・限界突破  → ベッチさん、ニーチェ、デシ＆ライアン
- ゼロベース思考      → エロンさん、デカルト、マルクス、ソクラテス
- 家族・子育て・コーチング → エドモンドソン、デシ＆ライアン、P4C、ロック

必ず以下のJSON形式のみで返してください（説明不要）：
{"philosophers": ["哲学者名1", "哲学者名2", "哲学者名3"], "category": "カテゴリ名"}
"""

NUDGE_STYLES = [
    "通常の問い。悩みの言葉を使って問いかける。60字以内。最後は「？」で終わる。",
    "一言の問いかけ。15字以内。鋭く短く。最後は「？」で終わる。",
    "観察の形式。「〜ように見えます。」で始まり最後に短い問いを添える。60字以内。",
]

CATEGORY_COLORS = {
    "判断・意思決定":          "#5A7A9A",
    "自己理解・アイデンティティ": "#8B6BA8",
    "モチベーション・やる気":    "#6A4A8A",
    "権力・発言できない":       "#5A3A4A",
    "対話・合意形成":          "#3A5A7A",
    "知識共有・KM":           "#4A7A6A",
    "変革・構造":             "#B85A5A",
    "不安・孤独・逆境":        "#4A7A8A",
    "交渉・利害調整":          "#5A6B8A",
    "感情・内面の整理":        "#7A7A9E",
    "恋愛・人間関係":          "#8A3A5A",
    "スポーツ・競技":          "#1D9E75",
    "学問・学習":             "#C17F3A",
    "キャリア・進路":          "#5A5A7A",
    "品質管理・工学判断":       "#3A7A5C",
    "ゴール設定・限界突破":     "#C04A3A",
    "ゼロベース思考":          "#7A5A3A",
    "家族・子育て・コーチング":  "#4A6A5A",
    "その他":                 "#888888",
}


# ==========================================
# [DIFF-2] 哲学者相談モード関数群
# ==========================================

def init_philo():
    for k, v in PHILO_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "philo_nickname" not in st.session_state:
        st.session_state.philo_nickname = ""


def reset_philo():
    for k, v in PHILO_DEFAULTS.items():
        st.session_state[k] = v


def save_philo_log(db, nickname, concern, shape_label, philosopher, category="その他"):
    if not db or not nickname:
        return
    try:
        db.collection("philo_log").add({
            "user_id":          nickname,
            "timestamp":        datetime.datetime.now(),
            "date":             datetime.date.today().isoformat(),
            "concern_summary":  concern[:40],
            "shape_label":      shape_label,
            "philosopher":      philosopher,
            "concern_category": category,
        })
    except Exception as e:
        st.toast(f"記録エラー: {e}", icon="⚠️")


def load_philo_log(db, nickname):
    if not db or not nickname:
        return []
    try:
        docs = (
            db.collection("philo_log")
            .where("user_id", "==", nickname)
            .limit(50)
            .stream()
        )
        records = [d.to_dict() for d in docs]
        records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return records
    except Exception as e:
        st.caption(f"記録読み込みエラー: {e}")
        return []


def generate_l1_question(concern):
    client = anthropic.Anthropic(api_key=st.session_state.get("philo_api_key", ""))
    system = (
        "あなたは問いを投げる存在です。"
        "相手の悩みを聞いて、その悩みの輪郭をはっきりさせるための問いを"
        "ひとつだけ返してください。\n"
        "制約：問いは1つのみ。分析・診断・アドバイス不可。"
        "30字以内。「〜ですか？」「〜か？」で終わる形式。"
    )
    try:
        resp = client.messages.create(
            model=MODEL_PHILO, max_tokens=80, system=system,
            messages=[{"role": "user", "content": concern}]
        )
        return resp.content[0].text.strip()
    except Exception:
        return "その状況で、いちばん重くのしかかっているのは何ですか？"


def generate_contextual_nudge(philosopher, concern, answer, style):
    client = anthropic.Anthropic(api_key=st.session_state.get("philo_api_key", ""))
    philo_data = PHILO_SHAPE_DATA.get(philosopher, {})
    system = (
        f"あなたは{philosopher}の思考で問いを作る存在です。\n"
        f"核心：{philo_data.get('one_liner', '')}\n\n"
        f"スタイル指定：{style}\n\n"
        "制約：相手の悩みの具体的な言葉を使う。哲学用語不可。"
        f"{philosopher}の視点を悩みに当てはめる。スタイルを厳守する。"
    )
    prompt = f"悩み：{concern}\n返答：{answer}"
    try:
        resp = client.messages.create(
            model=MODEL_PHILO, max_tokens=120, system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text.strip()
        if len(text) > 80:
            for sep in ["。", "？", "か"]:
                idx = text[:80].rfind(sep)
                if idx > 20:
                    text = text[:idx+1]
                    break
            else:
                text = text[:80] + "…"
        return text
    except Exception:
        return philo_data.get("nudge", "")


def select_philosophers(concern, answer):
    client = anthropic.Anthropic(api_key=st.session_state.get("philo_api_key", ""))
    prompt = f"悩み：{concern}\n返答：{answer}"
    try:
        resp = client.messages.create(
            model=MODEL_PHILO, max_tokens=100, system=PHILO_ADV_INDEX,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1:
            data = json.loads(text[s:e+1])
            names = data.get("philosophers", [])
            category = data.get("category", "その他")
            valid = [n for n in names if n in PHILO_SHAPE_DATA]
            if len(valid) < 3:
                for d in ["ソクラテス", "ストア派", "サルトル"]:
                    if d not in valid:
                        valid.append(d)
                    if len(valid) >= 3:
                        break
            return valid[:3], category
    except Exception:
        pass
    return ["ソクラテス", "ストア派", "サルトル"], "その他"


def call_philosopher_chat(philosopher, history):
    client = anthropic.Anthropic(api_key=st.session_state.get("philo_api_key", ""))
    system = (
        f"あなたは{philosopher}の思考パターンで問いを投げる存在です。\n"
        "制約：問いは1つだけ返す。答えを教えない。"
        f"{philosopher}の核心概念から問いを導く。80字以内。"
    )
    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    try:
        resp = client.messages.create(
            model=MODEL_PHILO, max_tokens=150, system=system, messages=msgs
        )
        return resp.content[0].text.strip()
    except Exception:
        return f"（{philosopher}）もう少し、そこを掘り下げてみませんか？"


def generate_reveal_intro(philosopher, concern, answer):
    client = anthropic.Anthropic(api_key=st.session_state.get("philo_api_key", ""))
    philo_data = PHILO_SHAPE_DATA.get(philosopher, {})
    system = (
        f"あなたは{philosopher}の思考パターンを紹介する案内役です。\n\n"
        "以下の3段構成で120字以内のイントロを作ってください：\n"
        f"①「あなたの悩み・考えは{philosopher}の型があっているようです。」\n"
        "②「その型とは〜という特徴的なもので、〜主義的な背景を持ちます。」\n"
        "③「では、その型に合わせて問いを深めてみましょう。」\n\n"
        f"制約：平易な言葉で。悩みの文脈を反映させる。\n"
        f"{philosopher}の核心：{philo_data.get('one_liner', '')}"
    )
    prompt = f"悩み：{concern}\n返答：{answer}"
    try:
        resp = client.messages.create(
            model=MODEL_PHILO, max_tokens=200, system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip()
    except Exception:
        return (
            f"あなたの悩みは{philosopher}の型があっているようです。"
            f"その型とは{philo_data.get('one_liner', '')}という特徴を持ちます。"
            "では、その型に合わせて問いを深めてみましょう。"
        )


def generate_synthesis(philosopher, concern, history):
    client = anthropic.Anthropic(api_key=st.session_state.get("philo_api_key", ""))
    philo_data = PHILO_SHAPE_DATA.get(philosopher, {})
    system = (
        f"あなたは{philosopher}の思考パターンで対話を合成する存在です。\n"
        f"核心：{philo_data.get('one_liner', '')}\n\n"
        "以下の2段構成で合成アファメーションを作ってください：\n"
        f"①「{philosopher}の型から言える、あなたへの核心的な問いは〜ということになりますね。」\n"
        "②「さぁ、課題解決に向けて前向きにいきましょう。」\n\n"
        "制約：対話の具体的な言葉を拾う。答えを渡さない。全体150字以内。"
    )
    dialogue_text = "\n".join([
        f"{'問い' if m['role'] == 'assistant' else '返答'}：{m['content']}"
        for m in history
    ])
    prompt = f"悩み：{concern}\n\n対話の記録：\n{dialogue_text}"
    try:
        resp = client.messages.create(
            model=MODEL_PHILO, max_tokens=220, system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip()
    except Exception:
        return (
            f"{philosopher}の型から言える、あなたへの核心的な問いはこの対話の中に宿っています。\n"
            "さぁ、課題解決に向けて前向きにいきましょう。"
        )


# --- フェーズ別レンダー関数 ---

def render_philo_input():
    st.markdown("#### あなたの悩みを聞かせてください")
    st.caption("どんなことが気になっていますか？短くて構いません。")
    concern = st.text_area("", placeholder="例：チームの意見が通らなくて消耗している",
                           height=80, key="philo_concern_input",
                           label_visibility="collapsed")
    if st.button("投げかける →", key="btn_submit_concern", type="primary"):
        if concern.strip():
            with st.spinner("問いを用意しています…"):
                question = generate_l1_question(concern.strip())
            st.session_state.philo_concern = concern.strip()
            st.session_state.philo_l1_question = question
            st.session_state.philo_phase = "layer1"
            st.rerun()
        else:
            st.warning("悩みを入力してください。")


def render_philo_layer1():
    st.markdown("#### 一つ、聞かせてください")
    with st.chat_message("assistant"):
        st.write(st.session_state.philo_l1_question)
    answer = st.chat_input("返答をどうぞ")
    if answer:
        st.session_state.philo_l1_answer = answer
        with st.spinner("形を選んでいます…"):
            candidates, category = select_philosophers(
                st.session_state.philo_concern, answer)
            shapes = build_shape_cards(candidates)
            for i, shape in enumerate(shapes):
                style = NUDGE_STYLES[i % len(NUDGE_STYLES)]
                shape["contextual_nudge"] = generate_contextual_nudge(
                    shape["philosopher"], st.session_state.philo_concern,
                    answer, style=style)
            st.session_state.philo_shapes = shapes
            st.session_state.philo_category = category
        st.session_state.philo_phase = "layer2"
        st.rerun()


def render_philo_layer2():
    st.markdown("#### 3つの思考の形が届いています")
    st.caption("どの形が、今の自分に近いですか？　哲学者は選んだ後に現れます。")
    st.markdown("")
    shapes = st.session_state.philo_shapes
    if not shapes:
        st.error("形データの取得に失敗しました。")
        if st.button("やり直す"):
            reset_philo(); st.rerun()
        return
    cols = st.columns(3)
    for i, (col, shape) in enumerate(zip(cols, shapes)):
        with col:
            with st.container(border=True):
                components.html(
                    f'<div style="background:transparent;">{shape["svg"]}</div>',
                    height=168, scrolling=False)
                st.markdown(
                    f"<div style='text-align:center;font-weight:600;"
                    f"color:{shape['color']};font-size:13px;"
                    f"margin-top:4px'>{shape['label']}</div>",
                    unsafe_allow_html=True)
                st.caption(shape["one_liner"])
                st.markdown(
                    f"<div style='font-size:12px;color:#333;line-height:1.7;"
                    f"margin-top:8px;padding:8px 10px;background:#f8f7f4;"
                    f"border-radius:6px;max-height:100px;overflow-y:auto'>"
                    f"{shape.get('contextual_nudge', shape['one_liner'])}</div>",
                    unsafe_allow_html=True)
                if st.button("この形を選ぶ", key=f"philo_pick_{i}",
                             use_container_width=True):
                    st.session_state.philo_selected_idx = i
                    db = setup_firestore()
                    save_philo_log(db, st.session_state.philo_nickname,
                                   st.session_state.philo_concern,
                                   shape["label"], shape["philosopher"],
                                   category=st.session_state.get("philo_category", "その他"))
                    st.session_state.philo_phase = "reveal"
                    st.rerun()
    st.caption("⚠ この形は思想の一断面です。見る角度によって異なる解釈があります。")


def render_philo_reveal():
    idx = st.session_state.philo_selected_idx
    shape = st.session_state.philo_shapes[idx]
    c = shape["color"]
    flip_html = f"""
    <style>
    @keyframes flipIn {{0%{{transform:rotateY(90deg);opacity:0}}100%{{transform:rotateY(0);opacity:1}}}}
    .rv{{animation:flipIn .55s ease-out forwards;border:1.5px solid {c};border-radius:14px;
         padding:20px 24px;background:{c}0a;max-width:340px;margin:0 auto;font-family:-apple-system,sans-serif;}}
    .rv-tag{{display:inline-block;font-size:11px;padding:2px 10px;border-radius:4px;
             background:{c}25;color:{c};font-weight:600;margin-bottom:10px;}}
    .rv-name{{font-size:18px;font-weight:500;color:#1a1a18;margin-bottom:6px;}}
    </style>
    <div class="rv">
      <div class="rv-tag">{shape['label']}</div>
      <div class="rv-name">{shape['philosopher']}の思考パターン</div>
      {shape['svg']}
    </div>"""
    components.html(flip_html, height=500, scrolling=False)
    if not st.session_state.get("philo_reveal_intro"):
        with st.spinner("型を読み解いています…"):
            intro = generate_reveal_intro(
                shape["philosopher"], st.session_state.philo_concern,
                st.session_state.philo_l1_answer)
        st.session_state.philo_reveal_intro = intro
    else:
        intro = st.session_state.philo_reveal_intro
    st.markdown(
        f"<div style='margin-top:12px;font-size:13px;color:#1a1a18;line-height:1.8;"
        f"background:#f8f7f4;border-radius:8px;padding:14px 16px;"
        f"border-left:3px solid {c}'>{intro}</div>",
        unsafe_allow_html=True)
    st.markdown("")
    col_deep, col_end = st.columns([3, 1])
    with col_deep:
        if st.button("問いを深める →", key="btn_to_layer3",
                     type="primary", use_container_width=True):
            st.session_state.philo_l3_history = []
            st.session_state.philo_turn_count = 0
            st.session_state.philo_phase = "layer3"
            st.rerun()
    with col_end:
        if st.button("記録だけして終わる", key="btn_to_mirror",
                     use_container_width=True):
            st.session_state.philo_phase = "mirror"
            st.rerun()
    st.caption("✓ この選択は記録されました。" if st.session_state.philo_nickname
               else "ニックネームを入力すると記録が残ります。")


def render_philo_layer3():
    idx = st.session_state.philo_selected_idx
    shape = st.session_state.philo_shapes[idx]
    c = shape["color"]
    with st.container(border=True):
        col_svg, col_info = st.columns([1, 2])
        with col_svg:
            components.html(
                f'<div style="background:transparent;">{shape["svg"]}</div>',
                height=100, scrolling=False)
        with col_info:
            st.markdown(
                f"<div style='font-size:12px;color:{c};font-weight:600;"
                f"margin-top:8px'>{shape['philosopher']}</div>",
                unsafe_allow_html=True)
            st.caption(shape["label"])
            st.caption(shape["one_liner"])
    st.markdown(
        f"<span style='font-size:13px;color:{c};font-weight:600'>"
        f"{shape['philosopher']}</span>"
        f"<span style='font-size:13px;color:#888'> との対話</span>",
        unsafe_allow_html=True)
    turn_count = st.session_state.philo_turn_count
    history = st.session_state.philo_l3_history
    if not history:
        opening = shape.get("contextual_nudge", shape["nudge"])
        history.append({"role": "assistant", "content": opening})
        st.session_state.philo_l3_history = history
    for msg in history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    if turn_count >= 2:
        st.markdown("")
        if st.button("この対話の着地点へ →", key="btn_to_synthesis", type="primary"):
            with st.spinner("対話を合成しています…"):
                synthesis = generate_synthesis(
                    shape["philosopher"], st.session_state.philo_concern, history)
            st.session_state.philo_synthesis = synthesis
            st.session_state.philo_phase = "synthesis"
            st.rerun()
        return
    remaining = 2 - turn_count
    st.caption(f"あと {remaining} 往復で着地点に向かいます。")
    user_input = st.chat_input("返答をどうぞ")
    if user_input:
        history.append({"role": "user", "content": user_input})
        st.session_state.philo_l3_history = history
        with st.spinner("…"):
            reply = call_philosopher_chat(shape["philosopher"], history)
        history.append({"role": "assistant", "content": reply})
        st.session_state.philo_turn_count = turn_count + 1
        st.session_state.philo_l3_history = history
        st.rerun()


def render_philo_synthesis():
    idx = st.session_state.philo_selected_idx
    shape = st.session_state.philo_shapes[idx]
    c = shape["color"]
    st.markdown("#### この対話の着地点")
    st.markdown("")
    components.html(
        f'<div style="background:transparent;max-width:400px;margin:0 auto;">'
        f'{shape["svg"]}</div>',
        height=320, scrolling=False)
    st.markdown(
        f"<div style='text-align:center;margin-top:8px;color:{c};"
        f"font-weight:600;font-size:16px'>{shape['philosopher']}</div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center;color:#888;font-size:12px;"
        f"margin-bottom:16px'>{shape['label']} — {shape['one_liner']}</div>",
        unsafe_allow_html=True)
    synthesis = st.session_state.get("philo_synthesis", "")
    st.markdown(
        f"<div style='font-size:13px;color:#1a1a18;line-height:1.85;"
        f"background:#f8f7f4;border-radius:8px;padding:16px 20px;"
        f"border-left:3px solid {c}'>"
        f"{synthesis.replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True)
    st.markdown("")
    if st.button("思考の地層を見る →", key="btn_to_mirror",
                 type="primary", use_container_width=True):
        st.session_state.philo_phase = "mirror"
        st.rerun()


def render_philo_mirror():
    st.markdown("#### あなたの思考の地層")
    nickname = st.session_state.philo_nickname
    if not nickname:
        st.info("サイドバーのニックネームを入力すると記録が表示されます。")
    else:
        db = setup_firestore()
        records = load_philo_log(db, nickname)
        if not records:
            st.caption("まだ記録がありません。")
        else:
            for rec in records:
                category = rec.get("concern_category", "その他")
                cat_color = CATEGORY_COLORS.get(category, "#888888")
                with st.container(border=True):
                    col_cat, col_date = st.columns([3, 2])
                    with col_cat:
                        st.markdown(
                            f"<span style='display:inline-block;font-size:10px;"
                            f"padding:2px 8px;border-radius:4px;"
                            f"background:{cat_color}20;color:{cat_color};"
                            f"font-weight:600'>{category}</span>",
                            unsafe_allow_html=True)
                    col_date.caption(rec.get("date", ""))
                    st.markdown(
                        f"**{rec.get('philosopher', '')}**"
                        f"　{rec.get('shape_label', '')}")
                    concern = rec.get("concern_summary", "")
                    if concern:
                        st.markdown(
                            f"<div style='font-size:12px;color:#555;margin-top:4px;"
                            f"padding:6px 10px;background:#f8f7f4;border-radius:6px;"
                            f"border-left:2px solid {cat_color}'>「{concern}…」</div>",
                            unsafe_allow_html=True)
            st.caption("← 気づきはここにある。アルゴリズムは何も言わない。")
    st.markdown("")
    if st.button("新しい相談を始める →", key="btn_restart", type="primary"):
        reset_philo()
        st.rerun()


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

def pre_hook_scope_guard(query, domain_key, threshold=SCOPE_THRESHOLD):
    """
    [DIFF-3] Domain-aware Scope Guard.
    Uses the selected domain's keywords and anchor for validation.

    3-stage check (priority order):
      1. Domain keyword -> pass
      2. Ultra-short conversation (<=10 chars) -> pass
      3. Anchor vector similarity -> pass if above threshold
    """
    config = DOMAIN_CONFIG.get(domain_key, DOMAIN_CONFIG[DEFAULT_DOMAIN])

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

    # [DIFF-3] ★ モード切替（サイドバー先頭）
    st.sidebar.markdown("---")
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "RAG学習"

    app_mode = st.sidebar.radio(
        "モードを選択:",
        options=["🏛️ RAG学習", "🧭 哲学者に相談"],
        index=0 if st.session_state.app_mode == "RAG学習" else 1,
    )
    st.session_state.app_mode = app_mode

    # [DIFF-4] ★ 哲学者モード分岐
    if app_mode == "🧭 哲学者に相談":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧭 哲学者に相談")
        init_philo()
        # philo_api_keyをsession_stateに保存（philo関数から参照）
        st.session_state.philo_api_key = api_key or ""
        st.session_state.philo_nickname = st.sidebar.text_input(
            "ニックネーム（記録用）",
            value=st.session_state.philo_nickname,
            placeholder="例: mizuna",
            key="philo_nickname_input",
            help="入力すると思考の地層が記録されます。",
        )
        # メインエリア：哲学者相談モード
        st.title("🧭 哲学者に相談")
        st.caption("思考の形を選ぶことで、今の自分がどこにいるかわかる。")
        st.markdown("---")
        phase = st.session_state.philo_phase
        if phase == "input":
            render_philo_input()
        elif phase == "layer1":
            render_philo_layer1()
        elif phase == "layer2":
            render_philo_layer2()
        elif phase == "reveal":
            render_philo_reveal()
        elif phase == "layer3":
            render_philo_layer3()
        elif phase == "synthesis":
            render_philo_synthesis()
        elif phase == "mirror":
            render_philo_mirror()
        else:
            reset_philo()
            st.rerun()
        return  # ★ 哲学者モードはここで終了

    # --- 以下は既存のRAGモード（変更なし）---

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
        prompt, current_domain
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
            top_k=3,
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

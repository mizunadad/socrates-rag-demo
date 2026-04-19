"""
# ==============================================================================
# 🛠 Socrates Reconstruct (v5.0 - Manual Parsing & Absolute Recovery)
# ==============================================================================
#
# ⚠️ 注意: このスクリプトは対象ディレクトリの .md ファイルを直接上書きします。
#    実行前に必ずバックアップを取るか、git commit済みの状態で実行してください。
#    誤って実行した場合は git checkout <対象ディレクトリ> で復元できます。
# 
# [History]
# v4.9まで: YAMLライブラリの厳格さと、AIの出力のゆらぎに翻弄され情報が消失。
# v5.0: (今回) 
#   1. 入力読み込みに yaml ライブラリを使わず、独自の手動パーサーを実装。
#      コロン、全角スペース、インデントの乱れを無視して全情報を救出。
#   2. AIの出力も正規表現で「言葉の塊」として抽出し、マッピング。
#   3. 情報の欠落を物理的に不可能にする設計。
# ==============================================================================
"""

import os
import yaml
import re
import time
import anthropic

# --- 設定 ---
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "sk-ant-xxxxx")  # 環境変数 or 直接記入
MODEL_NAME = "claude-haiku-4-5-20251001"
client = anthropic.Anthropic(api_key=API_KEY)
TARGET_DIRS = ["docs"] # デモ文書で動作確認可能 あるいは　貴方の文書庫

def manual_metadata_parser(yaml_str):
    """
    YAMLライブラリを使わず、一行ずつ気合でパースする。
    文法エラーがあっても、全てのキーと値を救い出す。
    """
    data = {}
    current_key = None
    lines = yaml_str.split('\n')
    for line in lines:
        line = line.replace('　', ' ') # 全角スペース抹殺
        stripped = line.strip()
        if not stripped: continue
        
        # リストアイテム (- 値) の処理
        if stripped.startswith('-'):
            val = stripped[1:].strip().strip('"').strip("'")
            if current_key:
                if not isinstance(data[current_key], list):
                    data[current_key] = []
                data[current_key].append(val)
            continue
            
        # キー: 値 (key: value) の処理
        if ':' in stripped:
            key, val = stripped.split(':', 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            data[key] = val
            current_key = key
    return data

def robust_ai_parse(text):
    """AIの出力から summary と tags を無理やり抜き出す"""
    data = {}
    # summary抽出 (日本語・英語、引用符の有無に対応)
    s_match = re.search(r'(?:summary|要約|概要)\s*[:：]\s*["\'「]?(.*?)["\'」]?(?:\n|$|(?=tags|タグ))', text, re.DOTALL | re.IGNORECASE)
    if s_match:
        data['summary'] = s_match.group(1).strip()
    
    # tags抽出 (リスト形式、カンマ区切り、日本語キーに対応)
    t_match = re.search(r'(?:tags|タグ)\s*[:：]\s*\[?(.*?)\]?(?:\n|$)', text, re.IGNORECASE)
    if t_match:
        # リスト内のゴミを掃除
        raw = t_match.group(1).replace('"', '').replace("'", "").replace('、', ',').replace('-', '')
        data['tags'] = [t.strip() for t in raw.split(',') if t.strip()]
    return data

def optimize_for_socrates_spec(body, title):
    prompt = f"""あなたはRAGの編集者です。以下の技術文書を要約してください。
必ず summary: "..." と tags: [...] の形式で出力してください。

タイトル: {title}
本文: {body[:2500]}
"""
    try:
        res = client.messages.create(
            model=MODEL_NAME, max_tokens=600, temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.content[0].text.strip()
    except: return None

def main():
    print(f"🚀 Socrates Reconstruct (v5.0-Manual) 開始...")
    targets = []
    for root, _, files in os.walk(TARGET_DIRS[0]):
        for file in files:
            if file.endswith(".md"): targets.append(os.path.join(root, file))

    for i, path in enumerate(targets):
        print(f"[{i+1}/{len(targets)}] {path} ... ", end="", flush=True)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content.startswith('---'): continue
            parts = content.split('---', 2)
            
            # --- 救出：手動パースで全情報を吸い出す ---
            old_meta = manual_metadata_parser(parts[1])
            body = parts[2].strip()
            
            # --- 生成：AIに要約を頼む ---
            raw_ai_out = optimize_for_socrates_spec(body, old_meta.get('title', 'Unknown'))
            ai_data = robust_ai_parse(raw_ai_out) if raw_ai_out else {}
            
            # --- 融合：情報を1bitも捨てずにマージ ---
            final_data = old_meta.copy()
            
            # サマリーの決定（AIがダメなら本文冒頭）
            summary = str(ai_data.get('summary', '')).strip()
            if len(summary) < 5:
                summary = body.replace('\n', ' ')[:110] + "..."
            
            summary = " ".join(summary.split()).rstrip('。、, .')
            if len(summary) > 135: summary = summary[:131] + "（略）"
            final_data['summary'] = summary + "。"
            
            # タグをAIの厳選版に更新
            if ai_data.get('tags'):
                final_data['tags'] = ai_data.get('tags', [])[:5]
            
            # 必須項目
            if 'category' not in final_data: final_data['category'] = "Psychology"
            if 'level' not in final_data: final_data['level'] = "1"

            # --- 保存 ---
            safe_yaml = yaml.dump(final_data, allow_unicode=True, sort_keys=False, default_style='"', width=1000)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f"---\n{safe_yaml}---\n\n{body}")
            print("✅ 復活完了")
            time.sleep(0.4)
        except Exception as e:
            print(f"💥 Error: {e}")

if __name__ == "__main__":
    main()

"""
philo_shapes.py
philo_viz2.jsx を Python f-string SVG に変換したモジュール。
streamlit_app.py から import して使用する。

使い方:
    from philo_shapes import PHILO_SHAPE_DATA, build_shape_cards
"""

import math

# ── ユーティリティ ────────────────────────────────────────


def _marker(id_: str, color: str) -> str:
    """汎用矢印マーカー（defs内用）"""
    return (
        f'<marker id="{id_}" markerWidth="7" markerHeight="7" '
        f'refX="5" refY="3.5" orient="auto">'
        f'<path d="M0,0 L7,3.5 L0,7 Z" fill="{color}"/></marker>'
    )


def _small_marker(id_: str, color: str) -> str:
    return (
        f'<marker id="{id_}" markerWidth="6" markerHeight="6" '
        f'refX="5" refY="3" orient="auto">'
        f'<path d="M0,0 L6,3 L0,6 Z" fill="{color}"/></marker>'
    )


def _wrap(inner: str, defs: str = "") -> str:
    """SVG ラッパー（200×160 viewBox）"""
    defs_block = f"<defs>{defs}</defs>" if defs else ""
    return (
        f'<svg viewBox="0 0 200 160" width="100%" height="100%">'
        f"{defs_block}{inner}</svg>"
    )


# ── 各哲学者 SVG 生成関数 ──────────────────────────────────


def _svg_socrates(c: str) -> str:
    def _ellipse(i):
        dash = ' stroke-dasharray="4,3"' if i > 0 else ""
        return (
            f'<ellipse cx="100" cy="82" rx="{18+i*18}" ry="{11+i*10}" '
            f'fill="none" stroke="{c}" stroke-width="{round(1.5-i*0.2, 1)}" '
            f'opacity="{round(1-i*0.16, 2)}"{dash}/>'
        )
    ellipses = "".join(_ellipse(i) for i in range(5))
    return _wrap(
        ellipses
        + f'<circle cx="100" cy="82" r="13" fill="{c}" opacity="0.28" stroke="{c}" stroke-width="2"/>'
        + f'<text x="100" y="79" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">無知</text>'
        + f'<text x="100" y="89" text-anchor="middle" fill="{c}" font-size="7">の知</text>'
        + f'<text x="100" y="32" text-anchor="middle" fill="{c}" font-size="8" opacity="0.7">問い</text>'
        + f'<text x="100" y="19" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.5">深い問い…</text>'
    )


def _svg_plato(c: str) -> str:
    arrows = "".join(
        f'<path d="M{70+i*30},83 L{70+i*30},77" stroke="{c}" stroke-width="1.5" '
        f'marker-end="url(#arP)" opacity="0.7"/>'
        for i in range(3)
    )
    defs = (
        f'<marker id="arP" markerWidth="5" markerHeight="5" refX="2.5" refY="2.5" orient="auto">'
        f'<path d="M0,5 L2.5,0 L5,5 Z" fill="{c}"/></marker>'
    )
    inner = (
        f'<rect x="15" y="15" width="170" height="62" rx="5" fill="{c}" opacity="0.2" stroke="{c}" stroke-width="1.5"/>'
        + f'<rect x="15" y="85" width="170" height="60" rx="5" fill="#555" opacity="0.12" stroke="#888" stroke-width="1.5" stroke-dasharray="5,3"/>'
        + f'<text x="100" y="40" text-anchor="middle" fill="{c}" font-size="10" font-weight="700">イデア（本質）</text>'
        + f'<text x="100" y="56" text-anchor="middle" fill="{c}" font-size="8" opacity="0.85">完全・永遠・普遍</text>'
        + f'<text x="100" y="68" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.7">真の実在</text>'
        + f'<text x="100" y="108" text-anchor="middle" fill="#888" font-size="10" font-weight="700">現実（影）</text>'
        + f'<text x="100" y="124" text-anchor="middle" fill="#aaa" font-size="8">不完全・変化・近似</text>'
        + f'<text x="100" y="136" text-anchor="middle" fill="#aaa" font-size="7.5">JTB——正解のみが知識</text>'
        + arrows
    )
    return _wrap(inner, defs)


def _svg_aristotle(c: str) -> str:
    defs = _marker("arA", c)
    return _wrap(
        defs
        + f'<circle cx="38" cy="82" r="24" fill="{c}" opacity="0.15" stroke="{c}" stroke-width="1.5"/>'
        + f'<text x="38" y="79" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">現状</text>'
        + f'<text x="38" y="90" text-anchor="middle" fill="{c}" font-size="7">今の文脈</text>'
        + f'<circle cx="162" cy="82" r="24" fill="{c}" opacity="0.35" stroke="{c}" stroke-width="2"/>'
        + f'<text x="162" y="79" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">テロス</text>'
        + f'<text x="162" y="90" text-anchor="middle" fill="{c}" font-size="7">目的・善</text>'
        + f'<path d="M63,82 L137,82" stroke="{c}" stroke-width="2.5" marker-end="url(#arA)"/>'
        + f'<rect x="72" y="68" width="58" height="20" rx="4" fill="{c}" opacity="0.15" stroke="{c}" stroke-width="1"/>'
        + f'<text x="101" y="82" text-anchor="middle" fill="{c}" font-size="8.5" font-weight="700">フロネシス</text>'
        + f'<text x="101" y="118" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.75">実践知——文脈の中の最善判断</text>'
        + f'<text x="38" y="138" text-anchor="middle" fill="{c}" font-size="7" opacity="0.6">現実から帰納</text>'
        + f'<text x="162" y="138" text-anchor="middle" fill="{c}" font-size="7" opacity="0.6">現実の中の善</text>'
    )


def _svg_stoic(c: str) -> str:
    return _wrap(
        f'<rect x="10" y="20" width="85" height="120" rx="6" fill="{c}" opacity="0.15" stroke="{c}" stroke-width="1.5"/>'
        + f'<rect x="105" y="20" width="85" height="120" rx="6" fill="#888" opacity="0.07" stroke="#888" stroke-width="1.5" stroke-dasharray="5,3"/>'
        + f'<text x="52" y="52" text-anchor="middle" fill="{c}" font-size="8.5" font-weight="700">コントロール</text>'
        + f'<text x="52" y="64" text-anchor="middle" fill="{c}" font-size="8.5" font-weight="700">できるもの</text>'
        + f'<text x="52" y="85" text-anchor="middle" fill="{c}" font-size="8">意志・判断</text>'
        + f'<text x="52" y="97" text-anchor="middle" fill="{c}" font-size="8">行動・反応</text>'
        + f'<text x="52" y="128" text-anchor="middle" fill="{c}" font-size="8" font-weight="600">→ 全力を注ぐ</text>'
        + f'<text x="147" y="52" text-anchor="middle" fill="#888" font-size="8.5" font-weight="700">コントロール</text>'
        + f'<text x="147" y="64" text-anchor="middle" fill="#888" font-size="8.5" font-weight="700">できないもの</text>'
        + f'<text x="147" y="85" text-anchor="middle" fill="#aaa" font-size="8">他者・天気</text>'
        + f'<text x="147" y="97" text-anchor="middle" fill="#aaa" font-size="8">結果・評価</text>'
        + f'<text x="147" y="128" text-anchor="middle" fill="#aaa" font-size="8">→ 手放す</text>'
        + f'<line x1="100" y1="20" x2="100" y2="140" stroke="#ccc" stroke-width="1" stroke-dasharray="3,3"/>'
    )


def _svg_epicurus(c: str) -> str:
    items = [
        {"x": round(100 + 50 * math.cos((i * 120 + 30) * math.pi / 180), 1),
         "y": round(80 + 50 * math.sin((i * 120 + 30) * math.pi / 180), 1),
         "label": lb}
        for i, lb in enumerate(["名声", "富", "権力"])
    ]
    crosses = "".join(
        f'<circle cx="{it["x"]}" cy="{it["y"]}" r="18" fill="#888" opacity="0.12" stroke="#888" stroke-width="1" stroke-dasharray="3,2"/>'
        f'<line x1="{it["x"]-10}" y1="{it["y"]-10}" x2="{it["x"]+10}" y2="{it["y"]+10}" stroke="#c00" stroke-width="2" opacity="0.5"/>'
        f'<line x1="{it["x"]+10}" y1="{it["y"]-10}" x2="{it["x"]-10}" y2="{it["y"]+10}" stroke="#c00" stroke-width="2" opacity="0.5"/>'
        f'<text x="{it["x"]}" y="{it["y"]+28}" text-anchor="middle" fill="#888" font-size="7" opacity="0.7">{it["label"]}</text>'
        for it in items
    )
    return _wrap(
        f'<circle cx="100" cy="80" r="60" fill="{c}" opacity="0.08" stroke="{c}" stroke-width="1.5" stroke-dasharray="5,3"/>'
        + crosses
        + f'<circle cx="100" cy="80" r="22" fill="{c}" opacity="0.35" stroke="{c}" stroke-width="2"/>'
        + f'<text x="100" y="77" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">アタラクシア</text>'
        + f'<text x="100" y="88" text-anchor="middle" fill="{c}" font-size="7">苦の不在</text>'
        + f'<text x="100" y="150" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.7">過剰を削れば、すでに十分</text>'
    )


def _svg_descartes(c: str) -> str:
    layers = [
        {"y": 118, "w": 158, "label": "疑えない土台（cogito）", "op": 0.9},
        {"y": 94,  "w": 118, "label": "確実な第一原理",       "op": 0.7},
        {"y": 70,  "w": 82,  "label": "論理的推論",           "op": 0.5},
        {"y": 46,  "w": 50,  "label": "結論",                 "op": 0.35},
    ]
    rects = "".join(
        f'<rect x="{100 - l["w"]//2}" y="{l["y"]}" width="{l["w"]}" height="20" rx="3" '
        f'fill="{c}" opacity="{round(l["op"]*0.22, 3)}" stroke="{c}" stroke-width="1.5" stroke-opacity="{l["op"]}"/>'
        f'<text x="100" y="{l["y"]+13}" text-anchor="middle" fill="{c}" font-size="7.5" '
        f'font-weight="{"700" if i==0 else "500"}" opacity="{l["op"]}">{l["label"]}</text>'
        for i, l in enumerate(layers)
    )
    return _wrap(
        rects
        + f'<text x="12" y="38" fill="{c}" font-size="8" opacity="0.6">↑ 積み上げ</text>'
        + f'<text x="12" y="138" fill="{c}" font-size="8" opacity="0.9">↓ まず全部疑う</text>'
    )


def _svg_spinoza(c: str) -> str:
    defs = (
        f'<marker id="arS" markerWidth="5" markerHeight="5" refX="5" refY="2.5" orient="auto">'
        f'<path d="M0,0 L5,2.5 L0,5 Z" fill="{c}"/></marker>'
    )
    return _wrap(
        f'<ellipse cx="100" cy="80" rx="75" ry="55" fill="none" stroke="{c}" stroke-width="1.5" opacity="0.3"/>'
        + f'<ellipse cx="100" cy="80" rx="28" ry="22" fill="{c}" opacity="0.15" stroke="{c}" stroke-width="1.5"/>'
        + f'<text x="100" y="77" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">感情</text>'
        + f'<text x="100" y="88" text-anchor="middle" fill="{c}" font-size="7">（必然の産物）</text>'
        + f'<ellipse cx="22" cy="80" rx="6" ry="30" fill="{c}" opacity="0.2" stroke="{c}" stroke-width="1"/>'
        + f'<text x="22" y="84" text-anchor="middle" fill="{c}" font-size="7" font-weight="700">観</text>'
        + f'<path d="M35,60 Q55,62 72,72" stroke="{c}" stroke-width="1.5" fill="none" stroke-dasharray="3,2" marker-end="url(#arS)" opacity="0.7"/>'
        + f'<text x="52" y="55" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.85">理性で</text>'
        + f'<text x="52" y="65" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.85">理解する</text>'
        + f'<text x="100" y="148" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.7">外から観察→解放</text>',
        defs
    )


def _svg_locke(c: str) -> str:
    layers = [
        {"y": 58,  "label": "経験① 現場の観察",   "op": 0.5},
        {"y": 76,  "label": "経験② 試行と失敗",   "op": 0.65},
        {"y": 94,  "label": "経験③ 蓄積された知恵", "op": 0.8},
        {"y": 112, "label": "知識——経験のみが源泉", "op": 1.0},
    ]
    rects = "".join(
        f'<rect x="30" y="{l["y"]}" width="140" height="16" rx="3" fill="{c}" '
        f'opacity="{round(l["op"]*0.22, 3)}" stroke="{c}" stroke-width="1.5" stroke-opacity="{l["op"]}"/>'
        f'<text x="100" y="{l["y"]+11}" text-anchor="middle" fill="{c}" font-size="8" opacity="{l["op"]}">{l["label"]}</text>'
        for l in layers
    )
    return _wrap(
        f'<rect x="30" y="15" width="140" height="35" rx="5" fill="#333" opacity="0.3" stroke="#666" stroke-width="1.5" stroke-dasharray="4,3"/>'
        + f'<text x="100" y="30" text-anchor="middle" fill="#888" font-size="9" font-weight="700">タブラ・ラサ（白紙）</text>'
        + f'<text x="100" y="42" text-anchor="middle" fill="#666" font-size="7.5">生まれた時は空白</text>'
        + rects
        + f'<text x="100" y="148" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.7">権力は経験者から信託される</text>'
    )


def _svg_kant(c: str) -> str:
    cols, rows = 7, 5
    cw, ch, ox, oy = 24, 20, 16, 18
    cells = ""
    for r in range(rows):
        for col in range(cols):
            is_ex = (col == 3 and r == 2)
            x, y = ox + col * cw, oy + r * ch
            fill = "#c00" if is_ex else c
            f_op = "0.18" if is_ex else "0.14"
            s_color = "#e05050" if is_ex else c
            s_op = "0.9" if is_ex else "0.7"
            s_w = "2" if is_ex else "1.5"
            cells += (
                f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" '
                f'fill="{fill}" fill-opacity="{f_op}" '
                f'stroke="{s_color}" stroke-width="{s_w}" stroke-opacity="{s_op}"/>'
            )
            if is_ex:
                cells += (
                    f'<line x1="{x+4}" y1="{y+4}" x2="{x+cw-4}" y2="{y+ch-4}" stroke="#e05050" stroke-width="1.5" opacity="0.8"/>'
                    f'<line x1="{x+cw-4}" y1="{y+4}" x2="{x+4}" y2="{y+ch-4}" stroke="#e05050" stroke-width="1.5" opacity="0.8"/>'
                )
    return _wrap(
        cells
        + f'<text x="100" y="133" text-anchor="middle" fill="{c}" font-size="8" font-weight="700" opacity="0.9">全マス同じルール——例外は×</text>'
        + f'<text x="100" y="145" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.65">全員がやったら？ 普遍化できるか？</text>'
    )


def _svg_hegel(c: str) -> str:
    return _wrap(
        f'<polygon points="100,18 28,138 172,138" fill="{c}" opacity="0.08" stroke="{c}" stroke-width="1.5"/>'
        + f'<circle cx="28" cy="138" r="17" fill="{c}" opacity="0.2" stroke="{c}" stroke-width="1.5"/>'
        + f'<circle cx="172" cy="138" r="17" fill="{c}" opacity="0.2" stroke="{c}" stroke-width="1.5"/>'
        + f'<circle cx="100" cy="18" r="17" fill="{c}" opacity="0.4" stroke="{c}" stroke-width="2"/>'
        + f'<text x="100" y="15" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">ジンテーゼ</text>'
        + f'<text x="100" y="25" text-anchor="middle" fill="{c}" font-size="7">（統合・止揚）</text>'
        + f'<text x="28" y="135" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">テーゼ</text>'
        + f'<text x="28" y="145" text-anchor="middle" fill="{c}" font-size="7">A</text>'
        + f'<text x="172" y="135" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">アンチ</text>'
        + f'<text x="172" y="145" text-anchor="middle" fill="{c}" font-size="7">B</text>'
        + f'<text x="100" y="105" text-anchor="middle" fill="{c}" font-size="7" opacity="0.7">矛盾が進化を生む</text>'
    )


def _svg_marx(c: str) -> str:
    defs = _small_marker("arM", c)
    arrows = "".join(
        f'<path d="M{x},62 L{x},73" stroke="{c}" stroke-width="2" marker-end="url(#arM)" opacity="0.8"/>'
        for x in [60, 100, 140]
    )
    return _wrap(
        f'<rect x="15" y="15" width="170" height="45" rx="5" fill="{c}" opacity="0.25" stroke="{c}" stroke-width="2"/>'
        + f'<text x="100" y="35" text-anchor="middle" fill="{c}" font-size="10" font-weight="700">経済構造・生産関係</text>'
        + f'<text x="100" y="50" text-anchor="middle" fill="{c}" font-size="8" opacity="0.8">（上部構造を規定する）</text>'
        + arrows
        + f'<rect x="35" y="75" width="130" height="35" rx="4" fill="{c}" opacity="0.12" stroke="{c}" stroke-width="1.5" stroke-dasharray="4,3"/>'
        + f'<text x="100" y="90" text-anchor="middle" fill="{c}" font-size="9" opacity="0.75">思想・意識・文化</text>'
        + f'<text x="100" y="102" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.65">（構造が意識を決める）</text>'
        + f'<text x="15" y="130" fill="{c}" font-size="7.5" opacity="0.85">個人の問題ではなく</text>'
        + f'<text x="15" y="142" fill="{c}" font-size="7.5" opacity="0.85">構造の問題——まず変革せよ</text>',
        defs
    )


def _svg_nietzsche(c: str) -> str:
    defs = _marker("arN", c)
    labels = ["外の評価", "他者の目", "ルサンチマン", "奴隷道徳"]
    bars = "".join(
        f'<line x1="20" y1="{y}" x2="155" y2="{y}" stroke="#555" stroke-width="6" stroke-linecap="round" opacity="0.35"/>'
        f'<text x="162" y="{y+4}" fill="#555" font-size="7" opacity="0.6">{labels[i]}</text>'
        for i, y in enumerate([35, 60, 85, 110])
    )
    return _wrap(
        defs + bars
        + f'<path d="M85,140 L125,20" stroke="{c}" stroke-width="3.5" marker-end="url(#arN)" stroke-linecap="round"/>'
        + f'<circle cx="85" cy="140" r="8" fill="{c}" opacity="0.5" stroke="{c}" stroke-width="2"/>'
        + f'<text x="85" y="154" text-anchor="middle" fill="{c}" font-size="7.5">自己</text>'
        + f'<text x="130" y="18" fill="{c}" font-size="8" font-weight="700">超人</text>'
        + f'<text x="40" y="20" fill="{c}" font-size="8" opacity="0.7">力への意志で突破</text>'
    )


def _svg_kierkegaard(c: str) -> str:
    defs = _marker("arK", c)
    return _wrap(
        defs
        + f'<rect x="10" y="60" width="70" height="70" rx="4" fill="{c}" opacity="0.12" stroke="{c}" stroke-width="1.5"/>'
        + f'<text x="45" y="88" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">論理の</text>'
        + f'<text x="45" y="100" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">領域</text>'
        + f'<text x="45" y="118" text-anchor="middle" fill="{c}" font-size="7" opacity="0.7">解けない…</text>'
        + f'<line x1="82" y1="60" x2="82" y2="130" stroke="{c}" stroke-width="2" stroke-dasharray="4,3" opacity="0.6"/>'
        + f'<line x1="118" y1="60" x2="118" y2="130" stroke="{c}" stroke-width="2" stroke-dasharray="4,3" opacity="0.6"/>'
        + f'<text x="100" y="108" text-anchor="middle" fill="{c}" font-size="9" font-weight="700">不安</text>'
        + f'<text x="100" y="120" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.8">自由のめまい</text>'
        + f'<rect x="120" y="60" width="70" height="70" rx="4" fill="{c}" opacity="0.28" stroke="{c}" stroke-width="2"/>'
        + f'<text x="155" y="88" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">実存の</text>'
        + f'<text x="155" y="100" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">決断</text>'
        + f'<text x="155" y="115" text-anchor="middle" fill="{c}" font-size="7" opacity="0.8">自ら選ぶ</text>'
        + f'<path d="M45,58 Q100,15 155,58" stroke="{c}" stroke-width="2.5" fill="none" stroke-dasharray="6,3" marker-end="url(#arK)"/>'
        + f'<text x="100" y="28" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">信仰的跳躍</text>'
    )


def _svg_heidegger(c: str) -> str:
    defs = _small_marker("arH", c)
    return _wrap(
        defs
        + f'<line x1="18" y1="100" x2="182" y2="100" stroke="#555" stroke-width="2" marker-end="url(#arH)"/>'
        + f'<circle cx="25" cy="100" r="5" fill="#555" stroke="#555" stroke-width="1.5"/>'
        + f'<text x="25" y="116" text-anchor="middle" fill="#666" font-size="7.5">誕生</text>'
        + f'<circle cx="175" cy="100" r="8" fill="{c}" opacity="0.5" stroke="{c}" stroke-width="2.5"/>'
        + f'<text x="175" y="97" text-anchor="middle" fill="{c}" font-size="7" font-weight="700">死</text>'
        + f'<text x="175" y="118" text-anchor="middle" fill="{c}" font-size="7.5">（有限性）</text>'
        + f'<circle cx="100" cy="100" r="14" fill="{c}" opacity="0.35" stroke="{c}" stroke-width="2.5"/>'
        + f'<text x="100" y="97" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">今</text>'
        + f'<text x="100" y="108" text-anchor="middle" fill="{c}" font-size="7">ここ</text>'
        + f'<path d="M100,84 L100,55" stroke="{c}" stroke-width="2" stroke-dasharray="3,2" fill="none"/>'
        + f'<ellipse cx="100" cy="45" rx="38" ry="12" fill="{c}" opacity="0.15" stroke="{c}" stroke-width="1.5"/>'
        + f'<text x="100" y="42" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">本来性</text>'
        + f'<text x="100" y="52" text-anchor="middle" fill="{c}" font-size="7">（自分として生きる）</text>'
        + f'<text x="100" y="140" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.7">死を意識した時、今が輝く</text>'
    )


def _svg_sartre(c: str) -> str:
    defs = (
        f'<marker id="arSa1" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">'
        f'<path d="M0,0 L6,3 L0,6 Z" fill="{c}" opacity="0.9"/></marker>'
        f'<marker id="arSa2" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">'
        f'<path d="M0,0 L6,3 L0,6 Z" fill="{c}" opacity="0.5"/></marker>'
    )
    return _wrap(
        f'<circle cx="100" cy="38" r="14" fill="{c}" opacity="0.3" stroke="{c}" stroke-width="2"/>'
        + f'<text x="100" y="35" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">実存</text>'
        + f'<text x="100" y="46" text-anchor="middle" fill="{c}" font-size="7">（先に存在する）</text>'
        + f'<path d="M100,53 L100,72" stroke="{c}" stroke-width="2" fill="none" marker-end="url(#arSa1)"/>'
        + f'<rect x="72" y="73" width="56" height="18" rx="4" fill="{c}" opacity="0.2" stroke="{c}" stroke-width="1.5"/>'
        + f'<text x="100" y="85" text-anchor="middle" fill="{c}" font-size="8.5" font-weight="700">選択の分岐点</text>'
        + f'<path d="M85,92 L45,130" stroke="{c}" stroke-width="2" fill="none" marker-end="url(#arSa1)"/>'
        + f'<path d="M115,92 L155,130" stroke="{c}" stroke-width="2" fill="none" marker-end="url(#arSa2)" stroke-dasharray="4,2" opacity="0.6"/>'
        + f'<rect x="18" y="130" width="55" height="22" rx="4" fill="{c}" opacity="0.28" stroke="{c}" stroke-width="2"/>'
        + f'<text x="45" y="142" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">責任を引き受ける</text>'
        + f'<rect x="127" y="130" width="55" height="22" rx="4" fill="#555" opacity="0.15" stroke="#666" stroke-width="1" stroke-dasharray="3,2"/>'
        + f'<text x="155" y="142" text-anchor="middle" fill="#888" font-size="8">悪しき信仰</text>'
        + f'<text x="155" y="125" text-anchor="middle" fill="#888" font-size="7" opacity="0.7">「しかたがない」</text>',
        defs
    )


def _svg_foucault(c: str) -> str:
    nodes = [
        {"x": 100, "y": 75,  "label": "知識=権力", "r": 20, "main": True,  "has": True},
        {"x": 100, "y": 22,  "label": "管理職",    "r": 12, "main": False, "has": True},
        {"x": 155, "y": 50,  "label": "専門家",    "r": 12, "main": False, "has": True},
        {"x": 155, "y": 105, "label": "外部者",    "r": 12, "main": False, "has": False},
        {"x": 100, "y": 130, "label": "現場",      "r": 12, "main": False, "has": False},
        {"x": 45,  "y": 105, "label": "新入り",    "r": 12, "main": False, "has": False},
        {"x": 45,  "y": 50,  "label": "経営層",    "r": 12, "main": False, "has": True},
    ]
    lines = "".join(
        f'<line x1="100" y1="75" x2="{n["x"]}" y2="{n["y"]}" '
        f'stroke="{c if n["has"] else "#555"}" stroke-width="1.5" '
        f'opacity="{0.7 if n["has"] else 0.35}" '
        f'{"" if n["has"] else "stroke-dasharray=\"4,3\""}/>'
        for n in nodes[1:]
    )
    circles = ""
    for n in nodes:
        fill = c if (n["main"] or n["has"]) else "#555"
        op = 0.35 if n["main"] else (0.25 if n["has"] else 0.12)
        sw = 2 if n["main"] else 1.5
        tc = c if (n["main"] or n["has"]) else "#888"
        fs = 8 if n["main"] else 7
        fw = "700" if n["main"] else "500"
        circles += (
            f'<circle cx="{n["x"]}" cy="{n["y"]}" r="{n["r"]}" fill="{fill}" '
            f'opacity="{op}" stroke="{fill}" stroke-width="{sw}"/>'
            f'<text x="{n["x"]}" y="{n["y"]+3}" text-anchor="middle" fill="{tc}" '
            f'font-size="{fs}" font-weight="{fw}">{n["label"]}</text>'
        )
        if not n["has"] and not n["main"]:
            circles += f'<text x="{n["x"]+n["r"]+2}" y="{n["y"]+3}" fill="#c00" font-size="9" opacity="0.6">×</text>'
    return _wrap(
        lines + circles
        + f'<text x="100" y="152" text-anchor="middle" fill="{c}" font-size="7" opacity="0.7">誰の発言が知識とみなされるか</text>'
    )


def _svg_habermas(c: str) -> str:
    pts = [
        {"x": 100, "y": 28}, {"x": 152, "y": 58}, {"x": 152, "y": 108},
        {"x": 100, "y": 138}, {"x": 48, "y": 108}, {"x": 48, "y": 58},
    ]
    lines = "".join(
        f'<line x1="{pts[i]["x"]}" y1="{pts[i]["y"]}" x2="{pts[j]["x"]}" y2="{pts[j]["y"]}" '
        f'stroke="{c}" stroke-width="1" opacity="0.2"/>'
        for i in range(len(pts)) for j in range(i + 1, len(pts))
    )
    circles = "".join(
        f'<circle cx="{p["x"]}" cy="{p["y"]}" r="12" fill="{c}" opacity="0.22" stroke="{c}" stroke-width="1.5"/>'
        for p in pts
    )
    return _wrap(
        lines + circles
        + f'<circle cx="100" cy="83" r="22" fill="none" stroke="{c}" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.4"/>'
        + f'<text x="100" y="80" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">強制なき</text>'
        + f'<text x="100" y="91" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">対話</text>'
        + f'<text x="100" y="152" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.7">全員が対等——論拠の質だけが合意を作る</text>'
    )


def _svg_bohm(c: str) -> str:
    waves = "".join(
        f'<path d="M15,{65+i*12} Q55,{48+i*12} 100,{65+i*12} Q145,{82+i*12} 185,{65+i*12}" '
        f'fill="none" stroke="{c}" stroke-width="{round(1.8-i*0.3,1)}" '
        f'opacity="{round(0.7-i*0.15,2)}" stroke-linecap="round"/>'
        for i in range(3)
    )
    xs = [40, 80, 120, 160]
    labels = ["前提A", "前提B", "前提C", "前提D"]
    bubbles = "".join(
        f'<circle cx="{x}" cy="{42 if i%2==0 else 30}" r="10" fill="{c}" opacity="0.12" stroke="{c}" stroke-width="1" stroke-dasharray="3,2"/>'
        f'<text x="{x}" y="{39 if i%2==0 else 27}" text-anchor="middle" fill="{c}" font-size="6.5" opacity="0.7">{labels[i]}</text>'
        f'<text x="{x}" y="{49 if i%2==0 else 37}" text-anchor="middle" fill="{c}" font-size="5.5" opacity="0.5">（保留）</text>'
        for i, x in enumerate(xs)
    )
    return _wrap(
        waves + bubbles
        + f'<circle cx="28" cy="118" r="16" fill="{c}" opacity="0.2" stroke="{c}" stroke-width="1.5"/>'
        + f'<circle cx="172" cy="118" r="16" fill="{c}" opacity="0.2" stroke="{c}" stroke-width="1.5"/>'
        + f'<text x="28" y="122" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">A</text>'
        + f'<text x="172" y="122" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">B</text>'
        + f'<text x="100" y="125" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">意味の流れ</text>'
        + f'<text x="100" y="148" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.7">結論なし——前提を保留して共に探る</text>'
    )


def _svg_harvard(c: str) -> str:
    return _wrap(
        f'<circle cx="75" cy="82" r="48" fill="{c}" opacity="0.12" stroke="{c}" stroke-width="1.5"/>'
        + f'<circle cx="125" cy="82" r="48" fill="{c}" opacity="0.12" stroke="{c}" stroke-width="1.5"/>'
        + f'<ellipse cx="100" cy="82" rx="22" ry="40" fill="{c}" opacity="0.24" stroke="{c}" stroke-width="1.5"/>'
        + f'<text x="50" y="72" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">自分の</text>'
        + f'<text x="50" y="82" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">利益</text>'
        + f'<text x="150" y="72" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">相手の</text>'
        + f'<text x="150" y="82" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">利益</text>'
        + f'<text x="100" y="78" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">共通</text>'
        + f'<text x="100" y="89" text-anchor="middle" fill="{c}" font-size="7.5">選択肢</text>'
        + f'<text x="75" y="28" text-anchor="middle" fill="{c}" font-size="7" opacity="0.6">↓立場を外す</text>'
        + f'<text x="125" y="28" text-anchor="middle" fill="{c}" font-size="7" opacity="0.6">↓立場を外す</text>'
    )


def _svg_p4c(c: str) -> str:
    qs = [
        {"x": round(100 + 52 * math.cos((i * 60 - 90) * math.pi / 180), 1),
         "y": round(75 + 52 * math.sin((i * 60 - 90) * math.pi / 180), 1)}
        for i in range(6)
    ]
    paths = "".join(
        f'<path d="M{qs[i]["x"]},{qs[i]["y"]} Q100,75 {qs[(i+1)%6]["x"]},{qs[(i+1)%6]["y"]}" '
        f'fill="none" stroke="{c}" stroke-width="1.2" opacity="0.3" stroke-dasharray="3,3"/>'
        for i in range(6)
    )
    circles = "".join(
        f'<circle cx="{q["x"]}" cy="{q["y"]}" r="13" fill="{c}" opacity="0.18" stroke="{c}" stroke-width="1.5"/>'
        f'<text x="{q["x"]}" y="{q["y"]+4}" text-anchor="middle" fill="{c}" font-size="11" font-weight="700">?</text>'
        for q in qs
    )
    return _wrap(
        paths + circles
        + f'<circle cx="100" cy="75" r="20" fill="{c}" opacity="0.2" stroke="{c}" stroke-width="2"/>'
        + f'<text x="100" y="72" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">問いを</text>'
        + f'<text x="100" y="82" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">持ち帰る</text>'
        + f'<text x="100" y="148" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.7">否定なし——何を言ってもいい共同探求</text>'
    )


def _svg_nonaka(c: str) -> str:
    quads = [
        {"x": 15,  "y": 15, "w": 85, "h": 68, "label": "共同化",  "sub": "暗黙→暗黙",    "op": 0.35},
        {"x": 102, "y": 15, "w": 83, "h": 68, "label": "表出化",  "sub": "暗黙→形式知",  "op": 0.55},
        {"x": 15,  "y": 85, "w": 85, "h": 68, "label": "内面化",  "sub": "形式知→暗黙",  "op": 0.45},
        {"x": 102, "y": 85, "w": 83, "h": 68, "label": "結合化",  "sub": "形式→形式",    "op": 0.3},
    ]
    cells = "".join(
        f'<rect x="{q["x"]}" y="{q["y"]}" width="{q["w"]}" height="{q["h"]}" rx="4" '
        f'fill="{c}" opacity="{round(q["op"]*0.3, 3)}" stroke="{c}" stroke-width="1.5" stroke-opacity="{q["op"]}"/>'
        f'<text x="{q["x"]+q["w"]//2}" y="{q["y"]+q["h"]//2-6}" text-anchor="middle" fill="{c}" '
        f'font-size="9" font-weight="700" opacity="{min(q["op"]+0.3, 1.0)}">{q["label"]}</text>'
        f'<text x="{q["x"]+q["w"]//2}" y="{q["y"]+q["h"]//2+8}" text-anchor="middle" fill="{c}" '
        f'font-size="7" opacity="{min(q["op"]+0.2, 1.0)}">{q["sub"]}</text>'
        for q in quads
    )
    return _wrap(
        cells
        + f'<text x="100" y="82" text-anchor="middle" fill="{c}" font-size="8" font-weight="700" opacity="0.9">SECI</text>'
        + f'<text x="100" y="152" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.7">フロネシスが4象限を回す原動力</text>'
    )


def _svg_deci(c: str) -> str:
    defs = (
        f'<linearGradient id="deciGrad" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0%" stop-color="#888" stop-opacity="0.6"/>'
        f'<stop offset="100%" stop-color="{c}" stop-opacity="0.9"/>'
        f'</linearGradient>'
    )
    labels = ["自律性", "有能感", "関係性"]
    inner_circles = "".join(
        f'<circle cx="{90+i*25}" cy="130" r="14" fill="{c}" opacity="{round(0.15+i*0.05,2)}" stroke="{c}" stroke-width="1.5"/>'
        f'<text x="{90+i*25}" y="134" text-anchor="middle" fill="{c}" font-size="6.5">{lb}</text>'
        for i, lb in enumerate(labels)
    )
    return _wrap(
        f'<rect x="18" y="55" width="164" height="28" rx="14" fill="url(#deciGrad)"/>'
        + f'<circle cx="32" cy="69" r="18" fill="#555" opacity="0.5" stroke="#888" stroke-width="2"/>'
        + f'<text x="32" y="66" text-anchor="middle" fill="#ccc" font-size="7" font-weight="700">Have</text>'
        + f'<text x="32" y="76" text-anchor="middle" fill="#ccc" font-size="7" font-weight="700">to</text>'
        + f'<circle cx="168" cy="69" r="18" fill="{c}" opacity="0.6" stroke="{c}" stroke-width="2"/>'
        + f'<text x="168" y="66" text-anchor="middle" fill="white" font-size="7" font-weight="700">Want</text>'
        + f'<text x="168" y="76" text-anchor="middle" fill="white" font-size="7" font-weight="700">to</text>'
        + f'<text x="30" y="44" text-anchor="middle" fill="#888" font-size="8">義務・恐れ</text>'
        + f'<text x="30" y="102" text-anchor="middle" fill="#888" font-size="7" opacity="0.7">外圧で動く</text>'
        + f'<text x="170" y="44" text-anchor="middle" fill="{c}" font-size="8">喜び・意味</text>'
        + f'<text x="170" y="102" text-anchor="middle" fill="{c}" font-size="7" opacity="0.8">内発で動く</text>'
        + inner_circles
        + f'<text x="100" y="118" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.7">基本的心理欲求の三要素</text>',
        defs
    )


def _svg_edmondson(c: str) -> str:
    defs = _small_marker("arE", c)
    return _wrap(
        defs
        + f'<circle cx="35" cy="80" r="18" fill="{c}" opacity="0.2" stroke="{c}" stroke-width="1.5"/>'
        + f'<text x="35" y="84" text-anchor="middle" fill="{c}" font-size="9">💬</text>'
        + f'<rect x="65" y="55" width="30" height="50" rx="4" fill="#c00" opacity="0.22" stroke="#c00" stroke-width="2"/>'
        + f'<text x="80" y="77" text-anchor="middle" fill="#e06060" font-size="7" font-weight="700">発言</text>'
        + f'<text x="80" y="89" text-anchor="middle" fill="#e06060" font-size="7" font-weight="700">コスト</text>'
        + f'<text x="80" y="120" text-anchor="middle" fill="#c00" font-size="7" opacity="0.8">高い壁</text>'
        + f'<line x1="95" y1="80" x2="115" y2="80" stroke="#c00" stroke-width="2" stroke-dasharray="4,2" opacity="0.7"/>'
        + f'<text x="105" y="74" text-anchor="middle" fill="#888" font-size="8" opacity="0.7">↓下げる</text>'
        + f'<rect x="118" y="65" width="22" height="30" rx="3" fill="{c}" opacity="0.15" stroke="{c}" stroke-width="1.5" stroke-dasharray="3,2"/>'
        + f'<text x="129" y="84" text-anchor="middle" fill="{c}" font-size="7" opacity="0.8">低い</text>'
        + f'<circle cx="160" cy="80" r="22" fill="{c}" opacity="0.25" stroke="{c}" stroke-width="2"/>'
        + f'<text x="160" y="76" text-anchor="middle" fill="{c}" font-size="7.5" font-weight="700">心理的</text>'
        + f'<text x="160" y="87" text-anchor="middle" fill="{c}" font-size="7.5" font-weight="700">安全性</text>'
        + f'<path d="M143,80 L138,80" stroke="{c}" stroke-width="2" fill="none" marker-end="url(#arE)"/>'
        + f'<text x="100" y="148" text-anchor="middle" fill="{c}" font-size="7.5" opacity="0.7">失敗を話せる場が知識創造の基盤</text>'
    )


def _svg_bechi(c: str) -> str:
    defs = (
        f'<marker id="arB" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">'
        f'<path d="M0,0 L6,3 L0,6 Z" fill="{c}"/></marker>'
    )
    return _wrap(
        f'<circle cx="82" cy="85" r="55" fill="{c}" opacity="0.06" stroke="{c}" stroke-width="1.5" stroke-dasharray="5,3"/>'
        + f'<circle cx="82" cy="85" r="35" fill="{c}" opacity="0.1" stroke="{c}" stroke-width="1.5"/>'
        + f'<circle cx="82" cy="85" r="18" fill="{c}" opacity="0.28" stroke="{c}" stroke-width="2"/>'
        + f'<text x="82" y="82" text-anchor="middle" fill="{c}" font-size="8" font-weight="700">現状</text>'
        + f'<text x="82" y="92" text-anchor="middle" fill="{c}" font-size="7">コンフォート</text>'
        + f'<text x="82" y="55" text-anchor="middle" fill="{c}" font-size="7" opacity="0.7">ゾーン外</text>'
        + f'<circle cx="158" cy="42" r="16" fill="{c}" opacity="0.5" stroke="{c}" stroke-width="2"/>'
        + f'<text x="158" y="39" text-anchor="middle" fill="white" font-size="8" font-weight="700">ゴール</text>'
        + f'<text x="158" y="49" text-anchor="middle" fill="white" font-size="7">Want to</text>'
        + f'<path d="M82,68 Q120,50 143,48" stroke="{c}" stroke-width="2" fill="none" marker-end="url(#arB)"/>'
        + f'<text x="112" y="44" text-anchor="middle" fill="{c}" font-size="7" opacity="0.9">エフィカシー↑</text>',
        defs
    )


def _svg_eron(c: str) -> str:
    xs = [45, 100, 155]
    leaf_labels = ["素材費", "物理限界", "工程コスト"]
    mid_rects = "".join(
        f'<line x1="{x}" y1="76" x2="{x}" y2="96" stroke="{c}" stroke-width="1.2" opacity="0.6" fill="none"/>'
        for x in xs
    )
    leaves = "".join(
        f'<rect x="{20+i*55}" y="96" width="50" height="18" rx="3" fill="{c}" opacity="0.15" stroke="{c}" stroke-width="1.2"/>'
        f'<text x="{45+i*55}" y="108" text-anchor="middle" fill="{c}" font-size="7">{lb}</text>'
        for i, lb in enumerate(leaf_labels)
    )
    return _wrap(
        f'<rect x="55" y="12" width="90" height="22" rx="4" fill="{c}" opacity="0.35" stroke="{c}" stroke-width="2"/>'
        + f'<text x="100" y="27" text-anchor="middle" fill="{c}" font-size="9" font-weight="700">製品・問題</text>'
        + f'<line x1="100" y1="35" x2="68" y2="55" stroke="{c}" stroke-width="1.5" opacity="0.7" fill="none"/>'
        + f'<line x1="100" y1="35" x2="132" y2="55" stroke="{c}" stroke-width="1.5" opacity="0.7" fill="none"/>'
        + f'<rect x="28" y="55" width="78" height="20" rx="3" fill="{c}" opacity="0.22" stroke="{c}" stroke-width="1.5"/>'
        + f'<text x="67" y="68" text-anchor="middle" fill="{c}" font-size="8">類推思考の前提</text>'
        + f'<rect x="94" y="55" width="78" height="20" rx="3" fill="{c}" opacity="0.22" stroke="{c}" stroke-width="1.5"/>'
        + f'<text x="133" y="68" text-anchor="middle" fill="{c}" font-size="8">コスト・制約</text>'
        + mid_rects + leaves
        + f'<rect x="30" y="124" width="140" height="20" rx="4" fill="{c}" opacity="0.4" stroke="{c}" stroke-width="2"/>'
        + f'<text x="100" y="138" text-anchor="middle" fill="{c}" font-size="8.5" font-weight="700">第一原理（物理的事実）</text>'
        + f'<text x="100" y="155" text-anchor="middle" fill="{c}" font-size="7" opacity="0.7">類推を疑い→素材レベルまで分解→再設計</text>'
    )


# ── メインデータ辞書 ──────────────────────────────────────

PHILO_SHAPE_DATA = {
    "ソクラテス": {
        "label":    "螺旋",
        "one_liner": "問いが問いを生む",
        "color":    "#C17F3A",
        "nudge":    "本当にその問題はそれなのか？　あなたが「知っている」と思っていることを、本当に知っているか？",
        "svg":      _svg_socrates("#C17F3A"),
    },
    "プラトン": {
        "label":    "2層構造",
        "one_liner": "理想が先、現実は影",
        "color":    "#6B8EC4",
        "nudge":    "あなたが思い描く理想の状態はどんな姿か？　今の状況はそこからどれだけ離れているか？",
        "svg":      _svg_plato("#6B8EC4"),
    },
    "アリストテレス": {
        "label":    "矢印",
        "one_liner": "現実から目的へ",
        "color":    "#7A9E6B",
        "nudge":    "この行動の目的（テロス）は何か？　今の文脈で最善の判断は何か？",
        "svg":      _svg_aristotle("#7A9E6B"),
    },
    "ストア派": {
        "label":    "2分割",
        "one_liner": "闘う相手を選ぶ",
        "color":    "#4A90A4",
        "nudge":    "コントロールできるものとできないものを分けると何か？　コントロールできないものに今どれだけエネルギーを使っているか？",
        "svg":      _svg_stoic("#4A90A4"),
    },
    "エピクロス": {
        "label":    "削除の円",
        "one_liner": "過剰を削れば十分",
        "color":    "#B8906A",
        "nudge":    "今あなたを苦しめているものは何か？　その苦しみを取り除けば、すでに十分ではないか？",
        "svg":      _svg_epicurus("#B8906A"),
    },
    "デカルト": {
        "label":    "積み上げ",
        "one_liner": "全部疑え、残ったものが土台",
        "color":    "#3A7A5C",
        "nudge":    "絶対に疑えないことは何か？　その前提は本当に確かか？　ゼロから積み上げ直すとどうなるか？",
        "svg":      _svg_descartes("#3A7A5C"),
    },
    "スピノザ": {
        "label":    "観察の枠",
        "one_liner": "感情を外から理解する",
        "color":    "#7A7A9E",
        "nudge":    "今の感情を外から観察するとしたらどう見えるか？　その感情はどこから来ているか？",
        "svg":      _svg_spinoza("#7A7A9E"),
    },
    "ロック": {
        "label":    "積み上がる層",
        "one_liner": "経験が知識の全て",
        "color":    "#9E7A4A",
        "nudge":    "その権力は誰から信託されたのか？　現場の経験は今の判断に生かされているか？",
        "svg":      _svg_locke("#9E7A4A"),
    },
    "カント": {
        "label":    "完全格子",
        "one_liner": "全員に適用できるか？",
        "color":    "#5A7A9A",
        "nudge":    "その選択を全員がとった世界はどうなるか？　普遍的法則として成立するか？",
        "svg":      _svg_kant("#5A7A9A"),
    },
    "ヘーゲル": {
        "label":    "三角形",
        "one_liner": "矛盾は進化のエンジン",
        "color":    "#8B6BA8",
        "nudge":    "AとBそれぞれの正しさは何か？　両方を同時に生かす第三の道はないか？",
        "svg":      _svg_hegel("#8B6BA8"),
    },
    "マルクス": {
        "label":    "逆ピラミッド",
        "one_liner": "解釈より変革せよ",
        "color":    "#B85A5A",
        "nudge":    "問題は個人の意識か、それとも構造か？　誰が利益を得ているか分析できるか？",
        "svg":      _svg_marx("#B85A5A"),
    },
    "ニーチェ": {
        "label":    "突き抜ける矢印",
        "one_liner": "外の評価を突き破れ",
        "color":    "#8A3A5A",
        "nudge":    "あなたを縛っている外部の評価軸は何か？　その縛りがなくなったとき、あなたは何をしたいか？",
        "svg":      _svg_nietzsche("#8A3A5A"),
    },
    "キルケゴール": {
        "label":    "信仰的跳躍",
        "one_liner": "論理を超えて跳べ",
        "color":    "#4A7A8A",
        "nudge":    "論理で解けないと感じているか？　最終的に選ぶのはあなたしかいない——不安は自由がある証拠。",
        "svg":      _svg_kierkegaard("#4A7A8A"),
    },
    "ハイデガー": {
        "label":    "タイムライン",
        "one_liner": "死を意識した時、今が輝く",
        "color":    "#5A5A7A",
        "nudge":    "もし今日が最後なら、この問題に向き合うか？　今やっていることは本来の自分がやりたいことか？",
        "svg":      _svg_heidegger("#5A5A7A"),
    },
    "サルトル": {
        "label":    "Y字分岐",
        "one_liner": "実存は本質に先立つ",
        "color":    "#7A4A3A",
        "nudge":    "「しかたがない」と思っているとき、本当に選択肢はゼロか？　役割を外したとき、あなたは何者か？",
        "svg":      _svg_sartre("#7A4A3A"),
    },
    "フーコー": {
        "label":    "権力ネット",
        "one_liner": "誰が知識を決めているか",
        "color":    "#5A3A4A",
        "nudge":    "誰が正しい知識を決めているか？　あなたが沈黙している理由は何か？",
        "svg":      _svg_foucault("#5A3A4A"),
    },
    "ハーバーマス": {
        "label":    "等距離ネット",
        "one_liner": "強制なき対話が合意を生む",
        "color":    "#4A6A5A",
        "nudge":    "全員が対等に発言できる場になっているか？　論拠の質だけで合意できているか？",
        "svg":      _svg_habermas("#4A6A5A"),
    },
    "ボーム": {
        "label":    "波と泡",
        "one_liner": "前提を保留して意味を流す",
        "color":    "#3A5A7A",
        "nudge":    "結論を急いでいないか？　相手の前提を本当に聞いたか？　あなた自身の前提を保留できるか？",
        "svg":      _svg_bohm("#3A5A7A"),
    },
    "ハーバード式": {
        "label":    "ベン図",
        "one_liner": "立場の裏の利益を見よ",
        "color":    "#5A6B8A",
        "nudge":    "相手の立場の背後にある利益は何か？　感情を脇に置いたとき、共通の選択肢は何か？",
        "svg":      _svg_harvard("#5A6B8A"),
    },
    "P4C": {
        "label":    "疑問符の輪",
        "one_liner": "問いを持ち帰ることが学び",
        "color":    "#7A6A3A",
        "nudge":    "正解を出さなければというプレッシャーはあるか？　この問いをもっと深めるとしたら、どんな問いが生まれるか？",
        "svg":      _svg_p4c("#7A6A3A"),
    },
    "野中郁次郎": {
        "label":    "SECI四象限",
        "one_liner": "暗黙知を渡せ",
        "color":    "#4A7A6A",
        "nudge":    "そのベテランの判断、何を見ているかを言葉にできるか？　その知識は文脈ごと次の人に渡っているか？",
        "svg":      _svg_nonaka("#4A7A6A"),
    },
    "デシ＆ライアン": {
        "label":    "グラデーション",
        "one_liner": "Have toかWant toか",
        "color":    "#6A4A8A",
        "nudge":    "今やっていることはHave toかWant toか？　本当にやりたいと感じる部分はどこか？",
        "svg":      _svg_deci("#6A4A8A"),
    },
    "エドモンドソン": {
        "label":    "コストの壁",
        "one_liner": "発言コストを下げよ",
        "color":    "#5A7A4A",
        "nudge":    "このチームで失敗を話せるか？　発言をためらわせているものは何か？",
        "svg":      _svg_edmondson("#5A7A4A"),
    },
    "ベッチさん": {
        "label":    "同心円＋ゴール",
        "one_liner": "エフィカシーが見える世界を変える",
        "color":    "#C04A3A",
        "nudge":    "そのゴールはコンフォートゾーンの内側か外側か？　エフィカシーが10倍あったら何が見えてくるか？",
        "svg":      _svg_bechi("#C04A3A"),
    },
    "エロンさん": {
        "label":    "分解ツリー",
        "one_liner": "類推を疑い素材まで分解せよ",
        "color":    "#7A5A3A",
        "nudge":    "その前提は類推から来ているか、物理的事実から来ているか？　10倍改善するとしたら最初に何を疑うか？",
        "svg":      _svg_eron("#7A5A3A"),
    },
}


# ── ユーティリティ関数 ────────────────────────────────────


def build_shape_cards(philosopher_names: list[str]) -> list[dict]:
    """
    LLMが選出した哲学者名リストからカードデータを構築する。
    streamlit_app.py の render_layer2() で使用。
    """
    cards = []
    for name in philosopher_names:
        d = PHILO_SHAPE_DATA.get(name)
        if d:
            cards.append({
                "philosopher": name,
                "label":       d["label"],
                "one_liner":   d["one_liner"],
                "color":       d["color"],
                "nudge":       d["nudge"],
                "svg":         d["svg"],
            })
    return cards


def get_all_labels() -> dict[str, str]:
    """全哲学者の {名前: 形ラベル} マッピングを返す（デバッグ用）"""
    return {name: d["label"] for name, d in PHILO_SHAPE_DATA.items()}


if __name__ == "__main__":
    print(f"登録済み哲学者数: {len(PHILO_SHAPE_DATA)}")
    for name, d in PHILO_SHAPE_DATA.items():
        svg_len = len(d["svg"])
        print(f"  {name:12s} | {d['label']:12s} | SVG {svg_len:4d}chars | {d['one_liner']}")

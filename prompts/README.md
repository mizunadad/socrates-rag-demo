# prompts/

ソクラテスRAGで使用するシステムプロンプトの全文です。
`step3/app.py` および `step4/app.py` 内の定数から切り出した **参照用ドキュメント** です。

> コード側は引き続きファイル内の定数を使用しています。
> このディレクトリのファイルを編集してもアプリの動作は変わりません。

## ファイル一覧

| ファイル | コード内の定数 | 説明 |
|:--|:--|:--|
| coaching_L1.txt | `COACHING_PROMPTS[1]` | L1「伴走」: ソクラテス式コーチ。ヒント付き逆質問、3文以内 |
| coaching_L2.txt | `COACHING_PROMPTS[2]` | L2「沈黙」: 比喩と問いだけ、2文以内 |
| coaching_L3.txt | `COACHING_PROMPTS[3]` | L3「鉄仮面」: 逆質問1文のみ、一切の情報提供を禁止 |
| teacher.txt | `TEACHER_PROMPT` | 解説モード: 参照データに基づく丁寧な解説 |
| retry_constraints.txt | `LEVEL_RETRY_CONSTRAINTS` | Socratic Validation違反時の再生成指示（L1/L2/L3） |
| format_constraint.txt | `FORMAT_CONSTRAINT` | Haiku 4.5のMarkdown多用・長文化を抑制する出力形式制約 |

## 三層防御と各プロンプトの関係

```
第1層（予防安全）: FORMAT_CONSTRAINT + COACHING_PROMPTS
    → プロンプトで「こう振る舞え」と指示

第2層（衝突安全）: max_tokens
    → モード別に物理的な出力上限を設定

第3層（最終手段）: Socratic Validation + LEVEL_RETRY_CONSTRAINTS
    → 出力を検閲し、違反時に再生成指示を付与して1回だけリトライ
```

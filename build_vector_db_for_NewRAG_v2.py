"""
build_vector_db_for_NewRAG.py
-- BOOK Step 1-2: NewRAG Pipeline

Design:
  - Vol.7 article as design spec (3-stage fallback, extract_section, diagnostic log)
  - Folder mapping (STATIC_MAPPING) or DOCS_ROOT single-folder mode
  - Frontmatter summary priority -> extract_section fallback
  - Concatenation: Title -> Tags -> Meta -> Summary -> Analysis -> Body
  - requires/enables excluded from vector_text, saved to Firestore (knowledge graph)

[BOOK added] Features beyond article scope:
  - level field (difficulty: 1=intro, 2=applied, 3=expert)
  - Knowledge map routing (knowledge_meta collection)
  - Frontmatter summary priority retrieval
"""

import os
import sys
import re
import glob
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from sentence_transformers import SentenceTransformer
import frontmatter

# ==========================================
# Settings
# ==========================================

# -- Mode 1: Single folder (BOOK demo) --
DOCS_ROOT = "./docs"

# -- Mode 2: Multi-domain mapping (production) --
STATIC_MAPPING = {
    "Management": "Management",
    "Strategy_Design": "Strategy_Design",
    "Psychology": "Psychology",
    "next_gen_power": "next_gen_power",
    "Singularity_2026": "Singularity",
}

ARTICLE_SUB_MAPPING = {
    "AIinfo": "AIinfo",
    "Quality_and_Sequrity": "Quality_Security",
    "Semiconductor": "Semiconductor",
    "Tips": "Tips",
    "python_and_webtech": "Python_Web",
}

# -- Vol.7 Section 4.1: vector_text exclusion keys --
# requires/enables are for knowledge graph, not embedding budget
EXCLUDE_FROM_VECTOR = [
    "title", "tags", "summary", "date", "url",
    "requires", "enables",
]

# [BOOK added] Knowledge map routing tags
SKIP_VECTORIZATION_TAGS = ["ナレッジマップ"]


# ==========================================
# Init
# ==========================================

key_path = "serviceAccountKey.json"
if not os.path.exists(key_path):
    print(f"Error: {key_path} not found.")
    print("   Download from Firebase Console and place in script directory.")
    sys.exit(1)

if not firebase_admin._apps:
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()
print("Loading embedding model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# ==========================================
# -- Vol.7 Section 2: Stage 1 - Robust file loading --
# 3-stage fallback: OK -> Fixed -> Fallback
# ==========================================

def fallback_parse(content):
    """Stage 1 third-stage: regex forced extraction when YAML fails.

    Returns:
        tuple: (metadata dict, content_body str)
    """
    metadata = {}

    # Title extraction
    title_match = re.search(r"^title:\s*(.*)$", content, re.MULTILINE)
    if title_match:
        val = title_match.group(1).strip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        metadata["title"] = val

    # Tags extraction
    tags_match = re.search(r"^tags:\s*\[(.*?)\]", content, re.MULTILINE)
    if tags_match:
        tag_str = tags_match.group(1)
        tags = []
        for t in tag_str.split(","):
            t = t.strip()
            if (t.startswith('"') and t.endswith('"')) or (
                t.startswith("'") and t.endswith("'")
            ):
                t = t[1:-1]
            if t:
                tags.append(t)
        metadata["tags"] = tags
    else:
        metadata["tags"] = []

    # Body extraction
    parts = re.split(r"^---", content, maxsplit=2, flags=re.MULTILINE)
    if len(parts) >= 3:
        content_body = parts[2].strip()
    else:
        content_body = content

    return metadata, content_body


def load_file_robust(file_path):
    """Stage 1: Read file with 3-stage fallback.

    Returns:
        tuple: (metadata dict or None, content_body str or None, status str)
        status is one of: 'OK', 'Fixed', 'Fallback', or error description
    """
    # -- BOM removal + encoding fallback --
    with open(file_path, "rb") as f:
        raw = f.read()

    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]

    try:
        content = raw.decode("utf-8")
    except Exception:
        try:
            content = raw.decode("cp932")
        except Exception:
            return None, None, "Encoding Error"

    # -- Stage 1-1: Normal parse --
    try:
        post = frontmatter.loads(content)
        if isinstance(post.metadata, dict) and "title" in post.metadata:
            return post.metadata, post.content, "OK"
    except Exception:
        pass

    # -- Stage 1-2: Cleaning parse --
    try:
        content_clean = content.replace("\u3000", " ").replace("\u200b", "")
        match = re.match(r"^---\n(.*?)\n---", content_clean, re.DOTALL)
        if match:
            yaml_part = match.group(1)
            fixed_lines = []
            for line in yaml_part.split("\n"):
                if ":" in line and not line.strip().startswith("#"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key = parts[0]
                        val = parts[1].strip()
                        if val and ":" in val and not (
                            val.startswith('"') or val.startswith("'")
                        ):
                            val = f'"{val}"'
                        fixed_lines.append(f"{key}: {val}")
                        continue
                fixed_lines.append(line)

            fixed_content = (
                "---\n"
                + "\n".join(fixed_lines)
                + "\n---"
                + content_clean[match.end():]
            )
            post = frontmatter.loads(fixed_content)
            return post.metadata, post.content, "Fixed"
    except Exception:
        pass

    # -- Stage 1-3: Forced extraction --
    try:
        meta, body = fallback_parse(content)
        if meta.get("title"):
            return meta, body, "Fallback"
    except Exception as e:
        return None, None, str(e)

    return None, None, "Unknown Parse Error"


# ==========================================
# -- Vol.7 Section 3: Stage 2 - Section extraction --
# ==========================================

def extract_section(content, keywords):
    """Extract a section from markdown body by heading keywords.

    Args:
        content: Markdown body text
        keywords: List of heading keywords to match

    Returns:
        str: Extracted section text, or empty string if not found
    """
    kw_pattern = "|".join([re.escape(k) for k in keywords])
    regex = re.compile(
        rf"^##\s+.*?({kw_pattern}).*?$\s+(.*?)(?=^#{{1,2}}\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = regex.search(content)
    return match.group(2).strip() if match else ""


# ==========================================
# -- Vol.7 Section 6: Multi-domain folder mapping --
# ==========================================

def build_target_map():
    """Build folder -> category_id mapping.

    If DOCS_ROOT exists and STATIC_MAPPING folders don't,
    use DOCS_ROOT mode (single-folder, for BOOK demo).
    Otherwise use multi-domain mapping mode.
    """
    target_map = {}

    # Check if any STATIC_MAPPING folder exists
    has_mapping_folders = any(
        os.path.exists(folder) for folder in STATIC_MAPPING
    )

    if not has_mapping_folders and os.path.isdir(DOCS_ROOT):
        # -- DOCS_ROOT mode (BOOK demo) --
        # Category comes from each file's frontmatter
        target_map[DOCS_ROOT] = None
        print(f"   Mode: DOCS_ROOT ({DOCS_ROOT})")
    else:
        # -- Multi-domain mapping mode --
        for folder, cat_id in STATIC_MAPPING.items():
            if os.path.exists(folder):
                target_map[folder] = cat_id

        # Year folder auto-detection
        year_folders = glob.glob("Articles_[0-9][0-9][0-9][0-9]")
        for yf in year_folders:
            if not os.path.isdir(yf):
                continue
            for sub in os.listdir(yf):
                sub_path = os.path.join(yf, sub)
                if os.path.isdir(sub_path):
                    cat_id = ARTICLE_SUB_MAPPING.get(sub, sub)
                    target_map[sub_path] = cat_id

        print(f"   Mode: Multi-domain ({len(target_map)} folders)")

    return target_map


# ==========================================
# Main processing
# ==========================================

def process_new_rag():
    print("\n--- NewRAG Pipeline Start ---\n")

    target_map = build_target_map()
    if not target_map:
        print("Error: No target folders found.")
        sys.exit(1)

    success_count = 0
    error_count = 0
    map_count = 0

    for folder_path, default_category in target_map.items():

        for root, _, files in os.walk(folder_path):
            md_files = sorted(f for f in files if f.endswith(".md"))

            for filename in md_files:
                file_path = os.path.join(root, filename)
                doc_id = os.path.splitext(filename)[0]

                # -- Stage 1: Robust file loading --
                metadata, content_body, status = load_file_robust(
                    file_path
                )

                if metadata is None:
                    cat_label = default_category or "?"
                    print(
                        f"x  [{cat_label}] {filename}"
                        f" -> Skip (Reason: {status})"
                    )
                    error_count += 1
                    continue

                # -- Title fallback --
                title = metadata.get("title", doc_id)

                # -- Tag sanitize (dict -> str coercion) --
                # -- Vol.7 Section 3.3 --
                raw_tags = metadata.get("tags", [])
                if not isinstance(raw_tags, list):
                    raw_tags = []
                clean_tags = [
                    str(t) for t in raw_tags if t is not None
                ]

                # -- Category determination --
                if default_category:
                    category_id = default_category
                else:
                    category_id = metadata.get(
                        "category", "Uncategorized"
                    )

                # -- [BOOK added] Knowledge map routing --
                if any(
                    t in clean_tags
                    for t in SKIP_VECTORIZATION_TAGS
                ):
                    try:
                        meta_doc = {
                            "title": doc_id,
                            "display_title": title,
                            "tags": clean_tags,
                            "category": category_id,
                            "content": content_body,
                            "updated_at": (
                                firestore.SERVER_TIMESTAMP
                            ),
                        }
                        db.collection(
                            "knowledge_meta"
                        ).document(doc_id).set(
                            meta_doc, merge=True
                        )
                        print(
                            f"   [META] {filename}"
                            f" -> knowledge_meta registered"
                        )
                        map_count += 1
                    except Exception as e:
                        print(
                            f"x  [META] {filename}"
                            f" -> Error: {e}"
                        )
                        error_count += 1
                    continue

                # -- Stage 2: Section extraction --
                # [BOOK added] Frontmatter summary priority
                summary = metadata.get("summary") or ""
                if not summary:
                    summary = extract_section(
                        content_body,
                        ["要約", "Summary", "Overview", "概要"],
                    )
                analysis = extract_section(
                    content_body,
                    ["分析", "Analysis", "Japan", "日本の"],
                )

                # -- Stage 3: vector_text + Embedding --
                # -- Vol.7 Section 4.1 --
                meta_str = "\n".join(
                    f"{k}: {v}"
                    for k, v in metadata.items()
                    if k not in EXCLUDE_FROM_VECTOR
                    and isinstance(v, (str, int, float))
                )
                vector_text = (
                    f"Title: {title}\n"
                    f"Tags: {', '.join(clean_tags)}\n"
                    f"{meta_str}\n"
                    f"{summary}\n"
                    f"{analysis}\n"
                    f"{content_body}"
                )
                embedding = model.encode(vector_text).tolist()

                # -- Stage 4: Firestore registration --
                # -- Vol.7 Section 5 --
                date_val = metadata.get("date", "")
                if isinstance(date_val, datetime.date):
                    date_val = date_val.isoformat()

                doc_data = {
                    "title": doc_id,
                    "display_title": title,
                    "category": category_id,
                    "content": content_body,
                    "tags": clean_tags,
                    "summary_section": summary,
                    "analysis_section": analysis,
                    "embedding": embedding,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                    # [BOOK added]
                    "url": metadata.get("url", ""),
                    "date": date_val,
                    "level": metadata.get("level", ""),
                    "requires": metadata.get("requires", []),
                    "enables": metadata.get("enables", []),
                }
                # Remove None values
                doc_data = {
                    k: v
                    for k, v in doc_data.items()
                    if v is not None
                }

                try:
                    doc_ref = db.collection(
                        "tech_docs"
                    ).document(doc_id)
                    doc_ref.set(doc_data, merge=True)

                    # -- Diagnostic log (Vol.7 format) --
                    note = (
                        f" (Mode: {status})"
                        if status != "OK"
                        else ""
                    )
                    mark_sum = "o" if summary else "x"
                    mark_ana = "o" if analysis else "x"
                    print(
                        f"ok [{category_id}] {filename}"
                        f" -> sum={mark_sum}"
                        f" ana={mark_ana}{note}"
                    )
                    success_count += 1

                except Exception as e:
                    print(
                        f"x  [{category_id}] {filename}"
                        f" -> Error: {e}"
                    )
                    error_count += 1

    # -- Final summary --
    print(f"\nDone: success={success_count}"
          f" / fail={error_count}")
    if map_count > 0:
        print(f"   Knowledge meta: {map_count}")


if __name__ == "__main__":
    process_new_rag()

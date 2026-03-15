"""
build_vector_db_for_MixedRAG.py
-- BOOK Step 4: MixedRAG Pipeline (sync + exorcism)

Design:
  - Inherits all NewRAG logic (Vol.7 article compliance)
  - 3-pass architecture: Scan -> Exorcise -> Sync
  - Ghost cleanup via embedding DELETE_FIELD
  - --dry-run flag for safe simulation
  - Embedding model load skipped in dry-run mode

Usage:
  python build_vector_db_for_MixedRAG.py            # full run
  python build_vector_db_for_MixedRAG.py --dry-run   # simulation only
"""

import os
import sys
import re
import glob
import datetime
import argparse
import firebase_admin
from firebase_admin import credentials, firestore
from sentence_transformers import SentenceTransformer
import frontmatter

# ==========================================
# Settings (same as NewRAG)
# ==========================================

DOCS_ROOT = "./docs"

STATIC_MAPPING = {
    "Management": "Management",
    "Strategy_Design": "Strategy_Design",
    "Psychology": "Psychology",
    "gartner_2025": "gartner_2025",
    "nikkei_bp_2025_2035": "nikkei_bp_2025_2035",
    "next_gen_power": "next_gen_power",
    "automotive_2045": "automotive_2045",
    "Singularity_2026": "Singularity",
}

ARTICLE_SUB_MAPPING = {
    "AIinfo": "AIinfo",
    "Quality_and_Sequrity": "Quality_Security",
    "Semiconductor": "Semiconductor",
    "Tips": "Tips",
    "python_and_webtech": "Python_Web",
}

EXCLUDE_FROM_VECTOR = [
    "title", "tags", "summary", "date", "url",
    "requires", "enables",
]

SKIP_VECTORIZATION_TAGS = ["ナレッジマップ"]

# ==========================================
# CLI args
# ==========================================

parser = argparse.ArgumentParser(
    description="MixedRAG sync and exorcism pipeline",
    epilog="Example: python %(prog)s --dry-run",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Simulation only. No Firestore writes, no model load.",
)
args = parser.parse_args()
DRY_RUN = args.dry_run

# ==========================================
# Init
# ==========================================

key_path = "serviceAccountKey.json"
if not os.path.exists(key_path):
    print(f"Error: {key_path} not found.")
    sys.exit(1)

if not firebase_admin._apps:
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

model = None
if not DRY_RUN:
    print("Loading embedding model...")
    model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
else:
    print("[DRY-RUN] Embedding model load skipped.")


# ==========================================
# -- Vol.7 Section 2: Stage 1 - Robust file loading --
# (identical to NewRAG)
# ==========================================

def fallback_parse(content):
    """Stage 1 third-stage: regex forced extraction."""
    metadata = {}

    title_match = re.search(
        r"^title:\s*(.*)$", content, re.MULTILINE
    )
    if title_match:
        val = title_match.group(1).strip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        metadata["title"] = val

    tags_match = re.search(
        r"^tags:\s*\[(.*?)\]", content, re.MULTILINE
    )
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

    parts = re.split(
        r"^---", content, maxsplit=2, flags=re.MULTILINE
    )
    if len(parts) >= 3:
        content_body = parts[2].strip()
    else:
        content_body = content

    return metadata, content_body


def load_file_robust(file_path):
    """Stage 1: 3-stage fallback file loading."""
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

    # 1. Normal parse
    try:
        post = frontmatter.loads(content)
        if isinstance(post.metadata, dict) and "title" in post.metadata:
            return post.metadata, post.content, "OK"
    except Exception:
        pass

    # 2. Cleaning parse
    try:
        content_clean = content.replace(
            "\u3000", " "
        ).replace("\u200b", "")
        match = re.match(
            r"^---\n(.*?)\n---", content_clean, re.DOTALL
        )
        if match:
            yaml_part = match.group(1)
            fixed_lines = []
            for line in yaml_part.split("\n"):
                if ":" in line and not line.strip().startswith(
                    "#"
                ):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key = parts[0]
                        val = parts[1].strip()
                        if val and ":" in val and not (
                            val.startswith('"')
                            or val.startswith("'")
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

    # 3. Forced extraction
    try:
        meta, body = fallback_parse(content)
        if meta.get("title"):
            return meta, body, "Fallback"
    except Exception as e:
        return None, None, str(e)

    return None, None, "Unknown Parse Error"


# ==========================================
# -- Vol.7 Section 3: Section extraction --
# (identical to NewRAG)
# ==========================================

def extract_section(content, keywords):
    """Extract section from markdown by heading keywords."""
    kw_pattern = "|".join([re.escape(k) for k in keywords])
    regex = re.compile(
        rf"^##\s+.*?({kw_pattern}).*?$\s+(.*?)"
        rf"(?=^#{{1,2}}\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = regex.search(content)
    return match.group(2).strip() if match else ""


# ==========================================
# -- Vol.7 Section 6: Folder mapping --
# (identical to NewRAG)
# ==========================================

def build_target_map():
    """Build folder -> category_id mapping."""
    target_map = {}

    has_mapping_folders = any(
        os.path.exists(f) for f in STATIC_MAPPING
    )

    if not has_mapping_folders and os.path.isdir(DOCS_ROOT):
        target_map[DOCS_ROOT] = None
        print(f"   Mode: DOCS_ROOT ({DOCS_ROOT})")
    else:
        for folder, cat_id in STATIC_MAPPING.items():
            if os.path.exists(folder):
                target_map[folder] = cat_id

        year_folders = glob.glob(
            "Articles_[0-9][0-9][0-9][0-9]"
        )
        for yf in year_folders:
            if not os.path.isdir(yf):
                continue
            for sub in os.listdir(yf):
                sub_path = os.path.join(yf, sub)
                if os.path.isdir(sub_path):
                    cat_id = ARTICLE_SUB_MAPPING.get(sub, sub)
                    target_map[sub_path] = cat_id

        print(
            f"   Mode: Multi-domain ({len(target_map)} folders)"
        )

    return target_map


# ==========================================
# MixedRAG: 3-pass pipeline
# ==========================================

def process_mixed_rag():
    mode_label = "[DRY-RUN] " if DRY_RUN else ""
    print(
        f"\n--- {mode_label}MixedRAG Pipeline Start ---\n"
    )

    target_map = build_target_map()
    if not target_map:
        print("Error: No target folders found.")
        sys.exit(1)

    # ==========================================
    # Pass 1: Local scan + category ID set build
    # ==========================================
    print("\n-- Pass 1: Local file scan --")

    # category -> set of doc_ids
    local_ids_by_category = {}
    # doc_id -> (metadata, content_body, status, category_id)
    file_cache = {}

    for folder_path, default_category in target_map.items():
        for root, _, files in os.walk(folder_path):
            md_files = sorted(
                f for f in files if f.endswith(".md")
            )

            for filename in md_files:
                file_path = os.path.join(root, filename)
                doc_id = os.path.splitext(filename)[0]

                metadata, content_body, status = (
                    load_file_robust(file_path)
                )

                if metadata is None:
                    cat_label = default_category or "?"
                    print(
                        f"x  [{cat_label}] {filename}"
                        f" -> Skip (Reason: {status})"
                    )
                    continue

                # Category determination
                if default_category:
                    category_id = default_category
                else:
                    category_id = metadata.get(
                        "category", "Uncategorized"
                    )

                # Tag sanitize
                raw_tags = metadata.get("tags", [])
                if not isinstance(raw_tags, list):
                    raw_tags = []
                clean_tags = [
                    str(t) for t in raw_tags if t is not None
                ]

                # [BOOK added] Knowledge map skip
                if any(
                    t in clean_tags
                    for t in SKIP_VECTORIZATION_TAGS
                ):
                    # Register as metadata only
                    file_cache[doc_id] = (
                        metadata,
                        content_body,
                        status,
                        category_id,
                        clean_tags,
                        True,  # is_knowledge_map
                    )
                    continue

                # Track local IDs per category
                if category_id not in local_ids_by_category:
                    local_ids_by_category[category_id] = set()
                local_ids_by_category[category_id].add(doc_id)

                file_cache[doc_id] = (
                    metadata,
                    content_body,
                    status,
                    category_id,
                    clean_tags,
                    False,  # not knowledge_map
                )

    total_local = sum(
        len(ids) for ids in local_ids_by_category.values()
    )
    print(f"   Local files: {total_local}")
    print(
        f"   Categories: "
        f"{list(local_ids_by_category.keys())}"
    )

    # ==========================================
    # Pass 2: Ghost exorcism (per category)
    # ==========================================
    print("\n-- Pass 2: Ghost exorcism --")

    ghost_count = 0

    for category_id, local_ids in (
        local_ids_by_category.items()
    ):
        try:
            fs_docs = (
                db.collection("tech_docs")
                .where("category", "==", category_id)
                .stream()
            )
            for fs_doc in fs_docs:
                if fs_doc.id not in local_ids:
                    # Ghost detected
                    if DRY_RUN:
                        print(
                            f"   [GHOST] {fs_doc.id}"
                            f" ({category_id})"
                            f" -> would DELETE embedding"
                        )
                    else:
                        db.collection(
                            "tech_docs"
                        ).document(fs_doc.id).update(
                            {
                                "embedding": (
                                    firestore.DELETE_FIELD
                                )
                            }
                        )
                        print(
                            f"   [GHOST] {fs_doc.id}"
                            f" ({category_id})"
                            f" -> embedding DELETED"
                        )
                    ghost_count += 1

        except Exception as e:
            print(
                f"x  [{category_id}] Exorcism error: {e}"
            )

    if ghost_count == 0:
        print("   No ghosts found.")
    else:
        print(f"   Ghosts processed: {ghost_count}")

    if DRY_RUN:
        print(
            "\n======================================"
            f"\n   [DRY-RUN] Pipeline complete"
            f"\n   Ghosts found:  {ghost_count}"
            f"\n   Local files:   {total_local}"
            f"\n   (No writes performed)"
            "\n======================================"
        )
        return

    # ==========================================
    # Pass 3: Sync (register/update)
    # ==========================================
    print("\n-- Pass 3: Sync --")

    success_count = 0
    error_count = 0
    map_count = 0

    for doc_id, cached in file_cache.items():
        (
            metadata,
            content_body,
            status,
            category_id,
            clean_tags,
            is_knowledge_map,
        ) = cached
        filename = doc_id + ".md"
        title = metadata.get("title", doc_id)

        # [BOOK added] Knowledge map routing
        if is_knowledge_map:
            try:
                meta_doc = {
                    "title": doc_id,
                    "display_title": title,
                    "tags": clean_tags,
                    "category": category_id,
                    "content": content_body,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                }
                db.collection(
                    "knowledge_meta"
                ).document(doc_id).set(meta_doc, merge=True)
                print(
                    f"   [META] {filename}"
                    f" -> knowledge_meta registered"
                )
                map_count += 1
            except Exception as e:
                print(
                    f"x  [META] {filename} -> Error: {e}"
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
        doc_data = {
            k: v for k, v in doc_data.items() if v is not None
        }

        try:
            doc_ref = db.collection(
                "tech_docs"
            ).document(doc_id)
            doc_ref.set(doc_data, merge=True)

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

    # -- Final report --
    print(
        "\n======================================"
        "\n   MixedRAG Pipeline complete"
        f"\n   Ghosts:      {ghost_count}"
        f"\n   Vectorized:  {success_count}"
        f"\n   Metadata:    {map_count}"
        f"\n   Errors:      {error_count}"
        "\n======================================"
    )


if __name__ == "__main__":
    process_mixed_rag()

import sqlite3
import json

DB_PATH = "galx.db"


def extract_tags(keyword_string: str, is_block: bool):
    if not keyword_string:
        return []

    parts = keyword_string.split(":|:")
    tags = []
    for p in parts:
        p = p.strip()
        if p.startswith("tag:"):
            tag = p[4:].strip()  # remove "tag:"
            if is_block:
                for chunk in tag.split(" "):
                    actual = chunk.replace("&gt", "").strip()
                    if len(actual) > 0:
                        tags.append(actual)
            else:
                tags.append(tag)

    return tags


conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute(f"SELECT url_source, token_keywords FROM GalleryWithRead")

tags_set = set()
for url_source, token_keywords in cur:
    sauce = url_source or ""
    tags = extract_tags(
        token_keywords, any(kw in sauce for kw in ["gelbooru", "sakugabooru"])
    )
    if len(tags) > 0:
        tags_set.add(tuple(tags))

tags_list = [list(tags) for tags in tags_set]
with open("galx_tags.json", "w", encoding="utf-8") as f:
    json.dump(tags_list, f, ensure_ascii=False)

print("Saved", len(tags_list), "unique tag lists")

conn.close()

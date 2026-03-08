import json
from pathlib import Path
import random

galx_path = Path("galx_tags.json")
danbooru_path = Path("tags_danbooru_2022.json")

with galx_path.open("r", encoding="utf-8") as f:
    galx_tags_lists = json.load(f)  # [[tag1, tag2], [tag3, tag4], ...]

with danbooru_path.open("r", encoding="utf-8") as f:
    danbooru_tags = json.load(f)  # [tag1, tag2, ...]

flat_galx_tags = [tag for tags in galx_tags_lists for tag in tags]
all_tags_set = set(danbooru_tags) | set(flat_galx_tags)

# all_tags_set = sorted(all_tags_set)
all_tags_list = list(all_tags_set)
random.shuffle(all_tags_list)
random.shuffle(galx_tags_lists)

output = {"tags": all_tags_list, "real_examples": galx_tags_lists}

out_path = Path("output/merged_tags.json")
with out_path.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False)

print(
    f"Saved {len(all_tags_set)} unique tags and {len(galx_tags_lists)} examples to {out_path}"
)

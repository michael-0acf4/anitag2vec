from dloader import MergeSet
from tokenizer import TagBPETokenizer

tagtok = TagBPETokenizer()
data = MergeSet.from_file("data/output/merged_tags.json")
tagtok.train(data.tags)
print(tagtok.encode_ids("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))
print(tagtok.encode("dsa𓂀"))  # not in vocab

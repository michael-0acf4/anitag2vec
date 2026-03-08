from dloader import MergeSet
from tokenizer import TagBPETokenizer

print("Training tokenizer..")
tagtok = TagBPETokenizer(vocab_size=10000, min_frequency=2)
data = MergeSet.from_file("data/output/merged_tags.json")
tagtok.train(data.extend_with_synthetic(perm_limit=5, sub_array_count=5), "ayo.json")
print("Done!")
print(tagtok.encode("1girl glasses handsup"))
print(tagtok.encode("glasses"))
print(tagtok.encode("dsa𓂀"))  # not in vocab
print(tagtok.encode_ids("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"))

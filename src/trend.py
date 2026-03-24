import os
import json
import re
import math
import matplotlib.pyplot as plt
import numpy as np

CHECKPOINT_DIR = "checkpoints"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

files = [
    f for f in os.listdir(CHECKPOINT_DIR)
    if f.startswith("errors_") and f.endswith(".json")
]

colors = plt.cm.viridis(np.linspace(0, 1, len(files)))
plt.figure()

for color, file in zip(colors, files):
    match = re.match(r"errors_(.+)\.json", file)
    if not match:
        continue
    hash_part = match.group(1)
    errors_path = os.path.join(CHECKPOINT_DIR, file)
    setup_path = os.path.join(CHECKPOINT_DIR, f"setup_params_{hash_part}.json")
    if not os.path.exists(setup_path):
        continue

    errors = load_json(errors_path)
    setup = load_json(setup_path)
    batch_size = setup.get("TRAINING_BATCH_SIZE")
    if (
        batch_size is None
        or not isinstance(errors, list)
        or len(errors) == 0
        or not all(isinstance(x, (int, float)) for x in errors)
    ):
        continue

    log_b = math.log(batch_size)
    epochs = list(range(1, len(errors) + 1))
    plt.plot(epochs, errors, color=color, label=f"Batch={batch_size} ({hash_part[:4]}..{hash_part[-4:]})")
    plt.axhline(y=log_b, color=color, linestyle="--", alpha=0.7)

plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Error vs Epoch with log(Batch) lines")
plt.legend()
plt.grid(True)

plt.show()

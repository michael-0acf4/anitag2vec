import os
import random
import re
import matplotlib.pyplot as plt
import numpy as np
from at2v.anitag2vec import LossLogger

CHECKPOINT_DIR = "checkpoints"

files = sorted(
    f for f in os.listdir(CHECKPOINT_DIR)
    if f.startswith("errors_") and f.endswith(".json")
)

def random_color():
    return (random.random(), random.random(), random.random())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
for file_idx, file in enumerate(files):
    match = re.match(r"errors_(.+)\.json", file)
    if not match:
        continue
    hash_part = match.group(1)
    prefix_hash = "_".join(hash_part.split("_")[:2])
    display_hash = f"{hash_part[:4]}..{hash_part[-4:]}"
    errors_path = os.path.join(CHECKPOINT_DIR, file)
    setup_path = os.path.join(CHECKPOINT_DIR, f"config_{prefix_hash}.json")
    if not os.path.exists(setup_path):
        continue

    errors = LossLogger.load_from_file(errors_path)
    config = errors.training_config
    batch_size = getattr(config, "TRAINING_BATCH_SIZE", None)

    if batch_size is None:
        continue

    color_train = random_color()
    color_eval = random_color()
    color_test = random_color()

    epochs = range(1, len(errors.training_epoch_losses)+1)
    ax1.plot(epochs, errors.training_epoch_losses, color=color_train, label=f"Train (B={batch_size}, {display_hash})")
    ax1.plot(epochs, errors.eval_epoch_losses, color=color_eval, label=f"Eval (B={batch_size}, {display_hash})")
    ax1.axhline(y=np.mean(errors.test_losses), color=color_test, linestyle=":", label=f"Test avg (B={batch_size}, {display_hash})")

    steps = range(1, len(errors.test_losses)+1)
    color_lower = random_color()
    ax2.plot(steps, errors.test_losses, color=color_lower, label=f"Test (B={batch_size}, {display_hash})")
    ax2.axhline(y=np.mean(errors.test_losses), color=color_test, linestyle=":", label=f"Test avg (B={batch_size}, {display_hash})")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Error")
ax1.set_title("Training & Eval Loss per Epoch")
ax1.legend()
ax1.grid(True)

ax2.set_xlabel("Batch / Step")
ax2.set_ylabel("Error")
ax2.set_title("Test Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

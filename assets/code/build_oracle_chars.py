"""Build oracle_chars.npz for HW3 from the raw Oracle-MNIST files.

Source: https://github.com/wm-bupt/oracle-mnist (Wang & Deng, Scientific Data 2024).
The official train and test files are pooled, exact duplicate images are removed,
and each character is sampled to a fixed long-tail size with a fixed seed.
The grayscale image is repeated in 3 channels (RGB), the format of the students' handwriting.

Usage: python build_oracle_chars.py raw/ oracle_chars.npz
"""
import gzip
import sys

import numpy as np

SEED = 3
# label -> (name, character, number of images kept)
CLASSES = {
    0: ("big", "大", 1100),
    1: ("sun", "日", 950),
    2: ("moon", "月", 800),
    3: ("cattle", "牛", 650),
    4: ("next", "翌", 520),
    5: ("field", "田", 410),
    6: ("not", "勿", 320),
    7: ("arrow", "矢", 230),
    8: ("time", "巳", 140),
    9: ("wood", "木", 80),
}


def read_idx(path, offset):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=offset)


def load(raw, kind):
    x = read_idx(f"{raw}/{kind}-images-idx3-ubyte.gz", 16).reshape(-1, 28, 28)
    y = read_idx(f"{raw}/{kind}-labels-idx1-ubyte.gz", 8)
    return x, y


def main(raw, out):
    xa, ya = load(raw, "train")
    xb, yb = load(raw, "t10k")
    x, y = np.concatenate([xa, xb]), np.concatenate([ya, yb])

    _, first = np.unique(x.reshape(len(x), -1), axis=0, return_index=True)
    first = np.sort(first)
    x, y = x[first], y[first]

    rng = np.random.default_rng(SEED)
    keep = np.concatenate([
        rng.choice(np.flatnonzero(y == c), n, replace=False)
        for c, (_, _, n) in CLASSES.items()
    ])
    keep = rng.permutation(keep)

    np.savez_compressed(
        out,
        images=np.repeat(x[keep][..., None], 3, axis=-1),  # 28 x 28 x 3, equal channels
        labels=y[keep].astype(np.int64),
        class_names=np.array([v[0] for v in CLASSES.values()]),
        characters=np.array([v[1] for v in CLASSES.values()]),
    )
    print(f"{len(keep)} images, removed {len(xa) + len(xb) - len(x)} duplicates")
    for c, (name, ch, n) in CLASSES.items():
        print(f"{c} {name:7s} {ch} {n}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

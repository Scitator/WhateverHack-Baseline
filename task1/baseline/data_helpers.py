import os
import cv2
import numpy as np

import torch


def create_open_fn(datapath, line_encode_fn):
    def open_fn(row):
        image_name = str(row["image_name"])
        img = cv2.imread(os.path.join(datapath, image_name + ".jpg"))
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255 - 0.5
        img = img[:, :, (2, 1, 0)]  # BGR -> RGB

        text = row["text"].lower()
        text = line_encode_fn(text)

        label = np.float32(row.get("label", -1))

        result = {
            "image": torch.from_numpy(img).permute(2, 0, 1),
            "text": torch.from_numpy(text),
            "target": label
        }
        return result

    return open_fn


def load_vocab(filepath, default_tokens=None):
    default_tokens = default_tokens or ["PAD", "EOS", "UNK"]
    tokens = []
    with open(filepath) as fin:
        for line in fin:
            line = line.replace("\n", "")
            token, freq = line.split()
            tokens.append(token)

    tokens = default_tokens + list(sorted(tokens))
    token2id = {t: i for i, t in enumerate(tokens)}
    id2token = {i: t for i, t in enumerate(tokens)}
    return token2id, id2token


def create_line_encode_fn(w2id, max_len):
    def line_encode_fn(line):
        enc = np.array(list(map(
            lambda x: w2id.get(x, w2id["UNK"]), line.split())),
            dtype=np.int64)
        enc = enc[:max_len]

        result = np.ones(shape=(max_len,), dtype=np.int64) * int(w2id["PAD"])
        result[:len(enc)] = enc

        return result

    return line_encode_fn

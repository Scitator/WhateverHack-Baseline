import gensim
import numpy as np


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


def create_embedding_matrix(vocab2id, w2v_model, emb_size):

    def make_random_embedding(value):
        rnd = np.random.RandomState(value)
        vec = rnd.normal(0, 1, emb_size)
        vec_len = np.sqrt(np.dot(vec, vec))
        vec = vec / vec_len
        return vec

    embedding_matrix = []
    for key, value in sorted(vocab2id.items(), key=lambda x: x[1]):
        if key == "PAD":
            embedding_matrix.append(np.zeros((emb_size,)))
        elif key == "EOS":
            embedding_matrix.append(np.ones((emb_size,)))
        elif key == "UNK":
            embedding_matrix.append(make_random_embedding(value))
        else:
            try:
                embedding_matrix.append(w2v_model[key])
            except KeyError:
                embedding_matrix.append(make_random_embedding(value))
    embedding_matrix = np.vstack(embedding_matrix)
    return embedding_matrix

if __name__ == '__main__':
    w2v_path = "./GoogleNews-vectors-negative300.bin.gz"
    vocab_path = "./vocab.txt"
    embeddings_path = "./w2v_mini.npy"

    try:
        w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    except UnicodeDecodeError:
        w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=False)

    token2id, id2token = load_vocab(vocab_path)

    embeddings = create_embedding_matrix(token2id, w2v, w2v.vector_size)
    np.save(embeddings_path, embeddings)

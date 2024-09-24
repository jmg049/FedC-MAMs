from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


def cosine_similarity(x, y, maximise: bool = False) -> float:
    cos_sim = sklearn_cosine_similarity(x, y).mean()
    if maximise:
        return -cos_sim
    return cos_sim

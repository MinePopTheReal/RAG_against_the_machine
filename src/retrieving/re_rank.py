from sentence_transformers import CrossEncoder


class ReRank:
    def __init__(self):
        ...

    def re_ranking(self, query, chunks: list):
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        reranking_input: tuple[str, list[str]] = (query, chunks)

        reranker.predict()


# # Modèle pré-entraîné, prêt à l'emploi (pas de fine-tuning nécessaire pour commencer)
# # ms-marco-MiniLM : rapide, bon compromis vitesse/qualité, entraîné sur des paires question/passage

# # Si tes documents sont en français ou multilingues, préfère un modèle multilingue :
# # reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

# def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
#     """
#     query      : la requête utilisateur
#     candidates : liste de chunks, ex: [{"id": "c1", "text": "..."}, ...]
#                  (issus de ton retrieval large, BM25+FAISS, N=20-50 chunks)
#     top_k      : nombre de chunks à garder après reranking
#     """
#     # Le cross-encoder attend des paires (requête, texte_du_chunk)
#     pairs = [(query, c["text"]) for c in candidates]

#     # Une seule passe, en batch, pour tous les candidats
#     scores = reranker.predict(pairs)  # -> array de floats, un score par paire

#     # On associe chaque score à son chunk d'origine, puis on trie
#     scored = list(zip(candidates, scores))
#     scored.sort(key=lambda x: x[1], reverse=True)

#     return [{"chunk": c, "score": float(s)} for c, s in scored[:top_k]]


# # --- Intégration dans ton pipeline existant ---
# # 1. Retrieval large (garde ton BM25 + FAISS actuel, augmente juste N)
# candidates = retrieve_top_n(query, n=30)  # union BM25 + FAISS, sans forcément RRF ici

# # 2. Reranking par cross-encoder (l'étape qui juge vraiment la qualité)
# final_chunks = rerank(query, candidates, top_k=5)
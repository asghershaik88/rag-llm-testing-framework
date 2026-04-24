from sentence_transformers import CrossEncoder

# lightweight but strong reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(query, documents, top_k=2):
    """
    Re-ranks retrieved documents based on query relevance
    """

    pairs = [(query, doc.page_content) for doc in documents]

    scores = reranker.predict(pairs)

    # attach scores
    scored_docs = list(zip(documents, scores))

    # sort by score (highest first)
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # return top_k
    top_docs = [doc for doc, score in scored_docs[:top_k]]

    return top_docs
from sentence_transformers import SentenceTransformer, util
import pickle
import torch


def unserialize_stuff(filename):
  with open(filename, 'rb') as f:
    data = pickle.load(f)
    # print(data)
  return data


embedder = unserialize_stuff("../embedder.pickle")
corpus_embeddings = unserialize_stuff("../corpus_embeddings.pickle")
corpus = unserialize_stuff("../corpus.pickle")


def search(query=[]):
    answer = ""
    # embed user queries
    query_embeddings = embedder.encode(query, convert_to_tensor=True)

    # perform semantic search
    hits = util.semantic_search(query_embeddings,corpus_embeddings=corpus_embeddings,score_function=util.dot_score)

    for hit in hits:
        # get the top answer id from the semantic search
        top_pick = hit[0]

    # get the sentence used to answer user question
    answer = corpus[top_pick["corpus_id"]]

    return answer


answer1 = search(["what are some examples of on demand methods"])
print(f"Answer: {answer1}")

# dev.psi-mis.org
# old-dev.psi-mis.org
#  run.contact
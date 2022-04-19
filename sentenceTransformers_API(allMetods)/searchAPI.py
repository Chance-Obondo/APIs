from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, util
import pickle

def unserialize_stuff(filename):
  with open(filename, 'rb') as f:
    data = pickle.load(f)
    # print(data)
  return data

embedder = unserialize_stuff("../embedder.pickle")
corpus_embeddings = unserialize_stuff("../corpus_embeddings.pickle")
corpus = unserialize_stuff("../corpus.pickle")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello Chance first api call is working"}


@app.get("/ask/")
def ask(query):
    query_list = []
    query_list.append(query)
    answer = ""
    # embed user queries
    query_embeddings = embedder.encode(query_list, convert_to_tensor=True)

    # perform semantic search
    hits = util.semantic_search(query_embeddings,corpus_embeddings=corpus_embeddings,score_function=util.dot_score)

    for hit in hits:
        # get the top answer id from the semantic search
        top_pick = hit[0]

    # get the sentence used to answer user question
    answer = corpus[top_pick["corpus_id"]]

    return {"reply": answer}

# uvicorn searchAPI:app --reload
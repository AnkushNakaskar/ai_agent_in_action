import os
import ssl
import httpx
import numpy as np
import redis
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# 1. SSL & Networking Bypass (Prevents Mac/HuggingFace SSL errors)
# ------------------------------------------------------------------
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context

_orig_client_init = httpx.Client.__init__


def _patched_client_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _orig_client_init(self, *args, **kwargs)


httpx.Client.__init__ = _patched_client_init


# ------------------------------------------------------------------
# 2. Load Local Embedding Model
# ------------------------------------------------------------------
print("Loading local embedding model...")
model = SentenceTransformer(
    "Orange/orange-nomic-v1.5-1536", trust_remote_code=True
)


def get_embedding(text: str) -> list[float]:
    """Generates 1536-dim vector embedding using local model."""
    formatted_text = f"search_query: {text}"
    return model.encode(formatted_text).tolist()


# ------------------------------------------------------------------
# 3. Connect to Local Redis
# ------------------------------------------------------------------
# decode_responses=True decodes string fields like title/url/content
conn = redis.Redis(host="localhost", port=6379, decode_responses=True)
INDEX_NAME = "idx:posts"


# ------------------------------------------------------------------
# 4. Hybrid Search Function
# ------------------------------------------------------------------
def hybrid_search(query_vector, client, top_k=3, hybrid_fields="*"):
    base_query = f"{hybrid_fields}=>[KNN {top_k} @embedding $vector AS vector_score]"

    query = (
        Query(base_query)
        .return_fields(
            "url",
            "title",
            "publish_date",
            "description",
            "content",
            "vector_score",
        )
        .sort_by("vector_score")
        .dialect(2)
    )

    try:
        results = client.ft(INDEX_NAME).search(
            query, query_params={"vector": query_vector}
        )
    except Exception as e:
        print("Error calling Redis search: ", e)
        return None

    if results.total == 0:
        print("No results found for the given query vector.")
        return None

    return results


# ------------------------------------------------------------------
# 5. Interactive Query Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    user_query = input("\nEnter your query: ")

    print("Vectorizing query...")
    query_vector = get_embedding(user_query)

    # Convert vector list to float32 byte array for Redis
    query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()

    print("Searching for similar posts...")
    results = hybrid_search(query_vector_bytes, conn, top_k=3)

    if results and results.docs:
        print(f"\nFound {len(results.docs)} matching post(s):\n")
        print("=" * 60)
        for i, post in enumerate(results.docs, start=1):
            # Cosine distance to similarity conversion
            score = 1 - float(post.vector_score)
            title = getattr(post, "title", "No Title")
            url = getattr(post, "url", "N/A")
            content = getattr(post, "content", "No Content")

            print(f"[{i}] {title} (Similarity Score: {score:.4f})")
            print(f"Link: {url}")
            print(f"Content snippet: {content[:200]}...")
            print("-" * 60)
    else:
        print("No results found.")
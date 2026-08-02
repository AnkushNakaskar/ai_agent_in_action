import os
import ssl
import httpx
import numpy as np
import openai
import redis
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
import tiktoken

# ------------------------------------------------------------------
# 1. Environment & SSL Fixes (Prevents Mac / HF network SSL errors)
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
# 2. Local Embedding Model & Tokenizer Initializations
# ------------------------------------------------------------------
print("Loading local embedding model...")
model = SentenceTransformer(
    "Orange/orange-nomic-v1.5-1536", trust_remote_code=True
)

# Tokenizer for token budgeting
tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Utility function to count tokens in a string."""
    return len(tokenizer.encode(text))


def get_embedding(text: str) -> list[float]:
    """Generates 1536-dim vector embedding locally."""
    formatted_text = f"search_query: {text}"
    return model.encode(formatted_text).tolist()


# ------------------------------------------------------------------
# 3. Redis Connection
# ------------------------------------------------------------------
conn = redis.Redis(host="localhost", port=6379, decode_responses=True)
INDEX_NAME = "idx:posts"

# ------------------------------------------------------------------
# 4. OpenAI Client Setup
# ------------------------------------------------------------------
# Set your OpenAI API Key or ensure standard environment variable is set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)


# ------------------------------------------------------------------
# 5. Hybrid Search & RAG Context Construction Functions
# ------------------------------------------------------------------
def hybrid_search(query_vector, client, top_k=5, hybrid_fields="*"):
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


def get_search_results(
    query: str, max_token=4096, debug_message=False
) -> str:
    # 1. Embed query locally
    query_vector = get_embedding(query)

    # 2. Convert to float32 byte array for Redis
    query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()

    if debug_message:
        print("Searching for similar posts in Redis...")

    # 3. Perform Vector Search
    results = hybrid_search(query_vector_bytes, conn, top_k=5)

    # 4. Token budgeting
    token_budget = max_token - count_tokens(query)
    if debug_message:
        print(f"Token budget remaining: {token_budget}")

    message = (
        "Use the blog post below to answer the subsequent question. "
        'If the answer cannot be found in the articles, write "Sorry, I could not find an answer in the blog posts."'
    )
    question = f"\n\nQuestion: {query}"

    # 5. Iteratively append top-k document context while staying within token limits
    if results and results.docs:
        for i, post in enumerate(results.docs):
            post_content = getattr(post, "content", "")
            next_post = f'\n\nBlog post:\n"""\n{post_content}\n"""'

            new_token_usage = count_tokens(message + question + next_post)
            if new_token_usage < token_budget:
                if debug_message:
                    print(f"Added doc {i+1} - Token usage: {new_token_usage}")
                message += next_post
            else:
                if debug_message:
                    print(
                        f"Doc {i+1} exceeds token budget limits. Stopping context expansion."
                    )
                break
    else:
        print("No matching context found in vector store.")

    return message + question


def ask_gpt(query: str, max_token=4096, debug_message=False) -> str:
    # Build augmented context
    message = get_search_results(
        query, max_token, debug_message=debug_message
    )

    messages = [
        {
            "role": "system",
            "content": "You answer questions in summary from the blog posts.",
        },
        {"role": "user", "content": message},
    ]

    try:
        # Call OpenAI LLM with augmented context
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=0.95,
        )
        response_message = response.choices[0].message.content
        return response_message
    except Exception as e:
        return f"API Error during completion generation: {e}"


# ------------------------------------------------------------------
# 6. Interactive RAG Interface
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- RAG Engine Initialized ---")
    while True:
        try:
            query = input("\nPlease enter your query (or type 'exit'): ")
            if query.strip().lower() in ["exit", "quit"]:
                print("Exiting...")
                break

            if not query.strip():
                continue

            print("\nGenerating RAG Response...")
            answer = ask_gpt(query, max_token=15000, debug_message=True)
            print("\n--- Answer ---")
            print(answer)
            print("=" * 50)
        except KeyboardInterrupt:
            print("\nProgram interrupted.")
            break
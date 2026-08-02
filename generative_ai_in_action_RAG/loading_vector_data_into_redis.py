import os
# 1. Disable SSL verification for Hugging Face downloads
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

# 1. Force all Python HTTP libraries to ignore SSL validation
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

# Standard imports follow below...
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

import ssl
import httpx

# 1. Disable SSL check in python's default SSL context
ssl._create_default_https_context = ssl._create_unverified_context

# 2. Patch httpx.Client to disable SSL verification globally
_orig_client_init = httpx.Client.__init__

def _patched_client_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _orig_client_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_client_init

# Standard imports below...
import feedparser
import numpy as np
import redis
from sentence_transformers import SentenceTransformer

print("Loading local embedding model...")
embedding_model = SentenceTransformer(
    "Orange/orange-nomic-v1.5-1536", trust_remote_code=True
)
print("Model loaded successfully.")
# 2. Connect to Local Redis
redis_conn = redis.Redis(host="localhost", port=6379, decode_responses=False)

INDEX_NAME = "idx:posts"
DOC_PREFIX = "post:"

# 3. Define Schema (1536 dimensions)
SCHEMA = [
    TagField("url"),
    TextField("title"),
    TextField("description"),
    TextField("publish_date"),
    TextField("content"),
    VectorField(
        "embedding",
        "HNSW",
        {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "COSINE"},
    ),
]

# Create Index if it doesn't exist
try:
    redis_conn.ft(INDEX_NAME).info()
    print(f"Index '{INDEX_NAME}' already exists.")
except redis.exceptions.ResponseError:
    print(f"Creating index '{INDEX_NAME}'...")
    redis_conn.ft(INDEX_NAME).create_index(
        fields=SCHEMA,
        definition=IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH),
    )
    print("Index created successfully.")

# 4. Fetch RSS Feed & Process Posts
RSS_URL = "https://blog.desigeek.com/index.xml"
feed = feedparser.parse(RSS_URL)

print(f"Found {len(feed.entries)} posts in feed. Processing locally...")

for i, entry in enumerate(feed.entries):
    title = getattr(entry, "title", "No Title")
    url = getattr(entry, "link", "")
    publish_date = getattr(entry, "published", "")
    description = getattr(entry, "summary", "")

    if hasattr(entry, "content"):
        content = entry.content[0].value
    else:
        content = description

    # Text to embed
    text_to_embed = f"search_document: {title} {description} {content}"

    # Generate 1536-dim embedding locally using CPU or GPU
    vector = embedding_model.encode(text_to_embed)

    # Convert vector to raw float32 bytes for Redis
    embedding_bytes = np.array(vector, dtype=np.float32).tobytes()

    doc_key = f"{DOC_PREFIX}{i+1}"

    document = {
        "url": url,
        "title": title,
        "description": description,
        "publish_date": publish_date,
        "content": content,
        "embedding": embedding_bytes,
    }

    # Store in Redis Hash
    redis_conn.hset(doc_key, mapping=document)
    print(f"[{i+1}/{len(feed.entries)}] Stored locally: {title}")

print("\n All RSS feed posts successfully embedded and stored in local Redis!")
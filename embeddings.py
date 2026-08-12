from google import genai
import os
import time
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

EMBED_MODEL = "gemini-embedding-001"

# The free tier allows ~100 embed_content requests per minute. Calling the
# API once per chunk (as before) blows past that limit on any decently
# sized PDF. Batching many chunks into a single request cuts the number of
# API calls dramatically - e.g. 300 chunks becomes 6 requests instead of
# 300.
BATCH_SIZE = 50
MAX_RETRIES = 5


def _embed_with_retry(contents):
    """
    Call embed_content with basic exponential backoff. If we still hit a
    rate limit (429 RESOURCE_EXHAUSTED), wait and retry a few times before
    giving up, instead of crashing the whole app on a transient limit.
    """
    delay = 5

    for attempt in range(MAX_RETRIES):
        try:
            return client.models.embed_content(
                model=EMBED_MODEL,
                contents=contents
            )
        except Exception as e:
            is_rate_limit = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)

            if is_rate_limit and attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue

            raise


def get_embedding(client, text):
    """Embed a single piece of text (used for the user's question at query time)."""
    response = _embed_with_retry(text)
    return response.embeddings[0].values


def get_embeddings_batch(client, texts):
    """
    Embed a list of chunk texts in as few API calls as possible, instead of
    one request per chunk. This is what indexing a PDF should use.
    """
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        response = _embed_with_retry(batch)

        all_embeddings.extend([e.values for e in response.embeddings])

        # Small pause between batches as extra safety margin against
        # the per-minute rate limit.
        if i + BATCH_SIZE < len(texts):
            time.sleep(1)

    return all_embeddings
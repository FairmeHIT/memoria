# BGE Input Limits TDD Evidence

## Source

User-reported self-hosted BGE limits: `POST /v1/embeddings` accepts 8,192
tokens and `POST /v1/rerank` accepts 512 tokens per query-document pair.

## User Journeys

- A user can add a long memory without the BGE embedding API rejecting it for
  input length.
- A user can search long stored memories without the BGE rerank API rejecting
  an over-limit query-document pair.

## Evidence

| Guarantee | Test | RED evidence | GREEN evidence |
| --- | --- | --- | --- |
| Long embeddings are split into request-safe chunks and recombined | `test_bge_embedding_chunks_long_text_and_averages_chunk_vectors` | Existing client sent one over-limit input and rejected the multi-vector mock response | Passes with two 8,192-budget requests and a length-weighted vector |
| Every rerank pair is bounded, all long-query content participates, and chunk scores map to the source document | `test_bge_reranker_chunks_long_queries_and_documents` | Existing client sent raw query and document content | Passes with UTF-8-safe query/document chunks and maximum source score |
| Expanded rerank chunks never exceed the API's 500-document batch limit | `test_bge_reranker_batches_expanded_documents_within_api_limit` | Existing client made one request for all expanded content | Passes with two bounded requests |

The RED command was `.venv/bin/pytest --no-cov -q tests/unit/test_bge.py`:
three input-limit tests failed because the provider sent unbounded inputs.

The focused GREEN command was `.venv/bin/pytest --no-cov -q tests/unit/test_bge.py`:
`9 passed`.

The complete verification command was `.venv/bin/pytest`:
`61 passed`, with total branch coverage of `81.03%`.

## Boundary Approach

The client has no exact tokenizer dependency or tokenization endpoint. It uses
UTF-8 byte length as a conservative upper bound for BGE byte-fallback
tokenization, reserving eight tokens per sequence or pair for model special
tokens. This protects the reported API limits without downloading a tokenizer
at runtime. Embedding batches are also kept inside the 8,192-token budget.

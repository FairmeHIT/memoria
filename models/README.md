# Model Artifacts

Pre-trained embedding models used by the `local` embedding backend.

## Auto-download

When `MEMORIA_EMBEDDING_BACKEND=local` is configured, the
`LocalBgeEmbedder` automatically downloads the default model
(`BAAI/bge-small-en-v1.5`) from the Hugging Face mirror
(hf-mirror.com) on first use.  No manual download is needed.

## Manual download

```bash
pip install -e ".[bge]"
python -c "
from memoria.embeddings import _auto_download_model
_auto_download_model('models/bge-small-en-v1.5')
"
```

## Custom model

Set `MEMORIA_BGE_EMBEDDING_MODEL` to a different HuggingFace repo ID
or a local path.
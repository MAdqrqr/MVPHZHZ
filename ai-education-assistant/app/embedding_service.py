import os
from threading import Lock

import numpy as np

from .errors import ServiceError


class EmbeddingService:
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.lock = Lock()
        self.last_error = None

    def load(self):
        with self.lock:
            if self.model is None:
                try:
                    os.environ.setdefault('HF_HOME', str(self.settings.data_dir / 'hf-cache'))
                    os.environ.setdefault('HF_XET_CACHE', str(self.settings.data_dir / 'hf-cache' / 'xet'))
                    # Стандартты HTTPS жүктеу; TLS пен сервердегі авторизация өзгермейді.
                    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(
                        self.settings.embedding_model,
                        revision=self.settings.embedding_revision,
                        device='cpu',
                        cache_folder=str(self.settings.data_dir / 'models'),
                        local_files_only=self.settings.embedding_local_only,
                        trust_remote_code=False,
                    )
                    # E5 шегі 512 токен: префикс пен арнайы токендерге орын қалдырамыз.
                    if self.settings.chunk_tokens > self.model.max_seq_length - 8:
                        self.model = None
                        raise ServiceError(422, 'chunk_size', 'CHUNK_TOKENS модельдің токен шегінен асады.')
                    self.last_error = None
                except ServiceError:
                    raise
                except Exception as exc:
                    self.last_error = 'Embedding моделі жүктелмеді. Желі рұқсатын немесе жергілікті модель жолын тексеріңіз.'
                    raise ServiceError(503, 'embedding_unavailable', self.last_error) from exc
        return self.model

    @property
    def tokenizer(self):
        return self.load().tokenizer

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        model = self.load()
        prefix = 'query: ' if query else 'passage: '
        with self.lock:
            vectors = model.encode([prefix + text for text in texts], batch_size=16,
                                   normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False)
        return np.ascontiguousarray(vectors, dtype='float32')

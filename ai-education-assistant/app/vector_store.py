import json
import os
import uuid
from pathlib import Path

import faiss
import numpy as np

from .errors import ServiceError


class VectorStore:
    """Бір процесс үшін атомарлық generation файлдары; сыртқы сервис lock ұстайды."""
    def __init__(self, directory: Path, signature: dict):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.signature = signature
        self.documents = []
        self.chunks = []
        self.vectors = np.empty((0, 0), dtype='float32')
        self.index = None
        self.compatible = True
        pointer = directory / 'current.json'
        if pointer.exists():
            generation = json.loads(pointer.read_text())['generation']
            if not isinstance(generation, str) or not generation.isalnum():
                raise ValueError('Индекс манифесі жарамсыз.')
            metadata = json.loads((directory / f'{generation}.json').read_text())
            self.documents, self.chunks = metadata['documents'], metadata['chunks']
            self.vectors = np.load(directory / f'{generation}.npy', allow_pickle=False)
            self.compatible = metadata['signature'] == signature
            self._build()

    def _build(self):
        self.index = None
        if len(self.vectors):
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(np.ascontiguousarray(self.vectors, dtype='float32'))

    def commit(self, documents, chunks, vectors):
        vectors = np.ascontiguousarray(vectors, dtype='float32')
        if len(chunks) != len(vectors) or not np.isfinite(vectors).all():
            raise ValueError('Векторлар мен мәтін бөліктері сәйкес емес.')
        generation = uuid.uuid4().hex
        metadata = {'documents': documents, 'chunks': chunks, 'signature': self.signature}
        (self.directory / f'{generation}.json').write_text(json.dumps(metadata, ensure_ascii=False), encoding='utf-8')
        np.save(self.directory / f'{generation}.npy', vectors, allow_pickle=False)
        index = None
        if len(vectors):
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            faiss.write_index(index, str(self.directory / f'{generation}.faiss'))
        temporary = self.directory / 'current.tmp'
        temporary.write_text(json.dumps({'generation': generation}), encoding='utf-8')
        os.replace(temporary, self.directory / 'current.json')
        self.documents, self.chunks, self.vectors, self.index = documents, chunks, vectors, index
        self.compatible = True
        # Жаңа манифест бекітілген соң ғана ескі generation файлдарын тазалаймыз.
        for path in self.directory.iterdir():
            if path.suffix in {'.npy', '.faiss', '.json'} and path.stem not in {generation, 'current'}:
                path.unlink(missing_ok=True)

    def ensure_compatible(self):
        if not self.compatible:
            raise ServiceError(409, 'reindex_required', 'Embedding немесе chunking өзгерді. Индексті қайта құрыңыз.')

    def search(self, query, top_k, threshold):
        self.ensure_compatible()
        if self.index is None:
            return []
        scores, positions = self.index.search(query, min(top_k, len(self.chunks)))
        return [{**self.chunks[int(i)], 'score': float(score)}
                for score, i in zip(scores[0], positions[0]) if i >= 0 and score >= threshold]

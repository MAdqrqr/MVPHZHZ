import hashlib
import time
import uuid
from threading import RLock

import numpy as np

from .document_processor import chunk_pages, extract_pages, safe_name
from .embedding_service import EmbeddingService
from .errors import ServiceError
from .llm_service import LLMService, cited_numbers
from .vector_store import VectorStore


class RAGService:
    def __init__(self, settings, embedder=None, llm=None):
        self.settings = settings
        self.lock = RLock()
        self.embedder = embedder or EmbeddingService(settings)
        self.llm = llm or LLMService(settings)
        self.documents_dir = settings.data_dir / 'documents'
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        signature = {'model': settings.embedding_model, 'revision': settings.embedding_revision, 'tokens': settings.chunk_tokens,
                     'overlap': settings.chunk_overlap, 'format': 1}
        self.store = VectorStore(settings.data_dir / 'index', signature)

    def list_documents(self):
        with self.lock:
            return [dict(d) for d in self.store.documents]

    def _chunks(self, pages):
        s = self.settings
        return chunk_pages(pages, self.embedder.tokenizer, s.chunk_tokens, s.chunk_overlap, s.max_chunks)

    def upload(self, content, filename):
        s = self.settings
        if len(content) > s.max_upload_mb * 1024 * 1024:
            raise ServiceError(413, 'file_too_large', f'Файл {s.max_upload_mb} МБ-тан аспауы керек.')
        filename = safe_name(filename)
        pages = extract_pages(content, filename, s.max_pages)
        digest = hashlib.sha256(content).hexdigest()
        with self.lock:
            self.store.ensure_compatible()
            for document in self.store.documents:
                if document['sha256'] == digest:
                    return {**document, 'duplicate': True}
            if len(self.store.documents) >= s.max_documents:
                raise ServiceError(413, 'too_many_documents', 'Құжаттар саны шектен асты. Бұрынғы құжатты жойыңыз.')
            chunks = self._chunks(pages)
            if not chunks:
                raise ServiceError(422, 'empty_chunks', 'Құжаттан пайдалы мәтін алынбады.')
            vectors = self.embedder.encode([c['text'] for c in chunks])
            document_id = uuid.uuid4().hex
            document = {'id': document_id, 'name': filename, 'sha256': digest,
                        'pages': len(pages), 'chunks': len(chunks), 'status': 'indexed',
                        'created_at': time.time()}
            chunks = [{**c, 'document_id': document_id, 'name': filename} for c in chunks]
            all_vectors = np.concatenate([self.store.vectors, vectors]) if len(self.store.vectors) else vectors
            path = self.documents_dir / f'{document_id}.pdf'
            path.write_bytes(content)
            try:
                self.store.commit(self.store.documents + [document], self.store.chunks + chunks, all_vectors)
            except Exception:
                if not any(d['id'] == document_id for d in self.store.documents):
                    path.unlink(missing_ok=True)
                raise
            return {**document, 'duplicate': False}

    def delete(self, document_id):
        with self.lock:
            documents = [d for d in self.store.documents if d['id'] != document_id]
            if len(documents) == len(self.store.documents):
                raise ServiceError(404, 'document_not_found', 'Құжат табылмады.')
            keep = [i for i, c in enumerate(self.store.chunks) if c['document_id'] != document_id]
            compatible = self.store.compatible
            # Өзге модельдің индексіне жаңа signature жазуға болмайды.
            if not compatible and documents:
                raise ServiceError(409, 'reindex_required', 'Құжатты жою алдында индексті қайта құрыңыз.')
            self.store.commit(documents, [self.store.chunks[i] for i in keep], self.store.vectors[keep])
            (self.documents_dir / f'{document_id}.pdf').unlink(missing_ok=True)

    def reindex(self):
        with self.lock:
            chunks, documents = [], []
            for doc in self.store.documents:
                path = self.documents_dir / f"{doc['id']}.pdf"
                if not path.exists():
                    raise ServiceError(409, 'missing_pdf', 'Сақталған PDF жоқ. Индекс өзгертілмеді.')
                new_chunks = self._chunks(extract_pages(path.read_bytes(), doc['name'], self.settings.max_pages))
                documents.append({**doc, 'chunks': len(new_chunks)})
                chunks.extend({**c, 'document_id': doc['id'], 'name': doc['name']} for c in new_chunks)
            vectors = self.embedder.encode([c['text'] for c in chunks]) if chunks else np.empty((0, 0), dtype='float32')
            self.store.commit(documents, chunks, vectors)
            return {'documents': len(documents), 'chunks': len(chunks)}

    def retrieve(self, question):
        with self.lock:
            self.store.ensure_compatible()
            if not self.store.chunks:
                return []
            query = self.embedder.encode([question], query=True)
            return self.store.search(query, self.settings.top_k, self.settings.min_similarity)

    def chat(self, question):
        start = time.perf_counter()
        sources = self.retrieve(question)
        if not sources:
            return {'answer': 'Қолда бар оқу материалдарынан жеткілікті жауап табылмады. PDF жүктеңіз немесе сұрақты нақтылаңыз.',
                    'sources': [], 'grounded': False, 'elapsed_seconds': round(time.perf_counter() - start, 3)}
        answer = self.llm.generate(question, sources)
        cited = cited_numbers(answer, len(sources))
        return {'answer': answer, 'sources': [{**sources[i - 1], 'citation': i} for i in cited],
                'grounded': True, 'elapsed_seconds': round(time.perf_counter() - start, 3)}

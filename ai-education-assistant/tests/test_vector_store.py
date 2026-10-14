from pathlib import Path

import numpy as np

from app.vector_store import VectorStore


def test_restore_kazakh_metadata_with_windows_default_encoding(tmp_path, monkeypatch):
    documents = [{'id': 'document1', 'name': 'Әлем дәрісі.pdf'}]
    chunks = [{'document_id': 'document1', 'name': 'Әлем дәрісі.pdf',
               'page': 2, 'text': 'Әдіс оқу материалындағы ұғымдарды түсіндіреді.'}]
    vectors = np.array([[1.0, 0.0]], dtype='float32')
    store = VectorStore(tmp_path, {'model': 'test'})
    store.commit(documents, chunks, vectors)

    original_read_text = Path.read_text

    def windows_read_text(path, encoding=None, errors=None):
        return original_read_text(path, encoding=encoding or 'cp1251', errors=errors)

    # Linux ортасында Windows-тың әдепкі cp1251 кодтауын қайталаймыз.
    monkeypatch.setattr(Path, 'read_text', windows_read_text)
    restored = VectorStore(tmp_path, {'model': 'test'})
    assert restored.documents == documents
    assert restored.chunks == chunks
    assert restored.search(vectors, 1, 0.8)[0]['text'] == chunks[0]['text']

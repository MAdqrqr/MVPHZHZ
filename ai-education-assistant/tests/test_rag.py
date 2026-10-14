import pytest

from app.config import Settings
from app.errors import ServiceError
from app.rag import RAGService
from conftest import TestEmbedder, TestLLM


def test_upload_duplicate_retrieve_and_persistence(service, settings, pdf):
    first = service.upload(pdf, 'lecture.pdf')
    second = service.upload(pdf, 'renamed.pdf')
    assert first['id'] == second['id'] and second['duplicate']
    assert len(service.list_documents()) == 1
    result = service.chat('Python')
    assert result['grounded'] and result['sources'][0]['page'] == 1
    assert result['sources'][0]['citation'] == 1
    restored = RAGService(settings, TestEmbedder(), TestLLM())
    assert restored.retrieve('SQL')[0]['page'] == 2
    assert list((settings.data_dir / 'index').glob('*.faiss'))


def test_no_context_does_not_call_llm(service, pdf):
    assert service.chat('Python')['sources'] == []
    service.upload(pdf, 'lecture.pdf')
    assert service.chat('unrelated')['grounded'] is False
    assert service.llm.calls == []


def test_delete_vectors_and_restart(service, settings, pdf):
    doc = service.upload(pdf, 'lecture.pdf')
    service.delete(doc['id'])
    assert service.retrieve('Python') == []
    assert not (settings.data_dir / 'documents' / (doc['id'] + '.pdf')).exists()
    restored = RAGService(settings, TestEmbedder(), TestLLM())
    assert restored.list_documents() == []
    assert len(restored.store.vectors) == 0


def test_reindex_and_signature_change(service, settings, pdf):
    service.upload(pdf, 'lecture.pdf')
    changed = Settings(data_dir=settings.data_dir, chunk_tokens=40, chunk_overlap=5)
    restored = RAGService(changed, TestEmbedder(), TestLLM())
    with pytest.raises(ServiceError) as error:
        restored.retrieve('Python')
    assert error.value.code == 'reindex_required'
    assert restored.reindex()['documents'] == 1
    assert restored.store.compatible


def test_reindex_failure_preserves_old_index(service, settings, pdf):
    doc = service.upload(pdf, 'lecture.pdf')
    (settings.data_dir / 'documents' / (doc['id'] + '.pdf')).unlink()
    with pytest.raises(ServiceError):
        service.reindex()
    assert service.retrieve('Python')[0]['page'] == 1


def test_oversize_upload(service):
    with pytest.raises(ServiceError) as error:
        service.upload(b'%PDF-' + b'x' * 1024 * 1024, 'big.pdf')
    assert error.value.status == 413


def test_no_citation_rejected(service, pdf):
    service.upload(pdf, 'lecture.pdf')
    service.llm.generate = lambda *args: 'Нақты дереккөзсіз жауап.'
    with pytest.raises(ServiceError) as error:
        service.chat('Python')
    assert error.value.code == 'invalid_citations'

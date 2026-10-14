import fitz
import pytest

from app.document_processor import chunk_pages, extract_pages, safe_name
from app.errors import ServiceError
from conftest import TestTokenizer


def test_pdf_pages(pdf):
    pages = extract_pages(pdf, 'дәріс.pdf', 10)
    assert len(pages) == 2
    assert pages[1]['page'] == 2
    assert 'SQL' in pages[1]['text']


@pytest.mark.parametrize('content,name', [(b'hello', 'lecture.pdf'), (b'%PDF-invalid', 'lecture.txt')])
def test_reject_fake_pdf(content, name):
    with pytest.raises(ServiceError) as error:
        extract_pages(content, name, 10)
    assert error.value.status == 415


def test_broken_pdf():
    with pytest.raises(ServiceError) as error:
        extract_pages(b'%PDF-1.7 broken', 'broken.pdf', 10)
    assert error.value.code == 'unreadable_pdf'


def test_scanned_pdf():
    with fitz.open() as doc:
        doc.new_page()
        content = doc.tobytes()
    with pytest.raises(ServiceError) as error:
        extract_pages(content, 'scan.pdf', 10)
    assert error.value.code == 'ocr_required'


def test_page_limit(pdf):
    with pytest.raises(ServiceError) as error:
        extract_pages(pdf, 'test.pdf', 1)
    assert error.value.status == 413


def test_chunk_overlap_and_page_metadata():
    text = ' '.join(f'token{i}' for i in range(60))
    chunks = chunk_pages([{'page': 7, 'text': text}], TestTokenizer(), 20, 5, 10)
    assert len(chunks) == 4
    assert all(c['page'] == 7 and len(c['text'].split()) <= 20 for c in chunks)
    assert chunks[0]['text'].split()[-5:] == chunks[1]['text'].split()[:5]


def test_chunk_count_limit():
    with pytest.raises(ServiceError) as error:
        chunk_pages([{'page': 1, 'text': 'word ' * 100}], TestTokenizer(), 20, 5, 2)
    assert error.value.code == 'too_many_chunks'


def test_safe_filename():
    assert safe_name('..\\..\\lecture.pdf') == 'lecture.pdf'
    assert safe_name('../../дәріс.pdf') == 'дәріс.pdf'
    assert '\x00' not in safe_name('a\x00.pdf')

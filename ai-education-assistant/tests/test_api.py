import pytest


def test_health_and_frontend(client):
    health = client.get('/api/health')
    assert health.json()['data']['mode'] == 'local_demo'
    assert 'llm_api_key' not in health.text.lower()
    assert client.get('/').status_code == 200
    assert client.get('/static/script.js').status_code == 200


def test_document_lifecycle(client, pdf):
    response = client.post('/api/documents/upload', files={'file': ('lecture.pdf', pdf, 'application/pdf')})
    assert response.status_code == 200
    doc = response.json()['data']
    assert len(client.get('/api/documents').json()['data']) == 1
    assert client.post('/api/chat', json={'question': 'Python'}).json()['data']['grounded']
    assert client.post('/api/reindex').json()['data']['documents'] == 1
    assert client.delete('/api/documents/' + doc['id']).status_code == 200
    assert client.delete('/api/documents/' + doc['id']).status_code == 404


@pytest.mark.parametrize('question', ['', '   ', 'x' * 2001])
def test_invalid_question(client, question):
    response = client.post('/api/chat', json={'question': question})
    assert response.status_code == 422
    assert response.json()['ok'] is False


def test_invalid_pdf_error(client):
    response = client.post('/api/documents/upload', files={'file': ('bad.pdf', b'not pdf')})
    assert response.status_code == 415
    assert response.json()['error']['code'] == 'invalid_pdf'


def test_cross_origin_mutation_denied(client):
    response = client.post('/api/chat', json={'question': 'Python'}, headers={'origin': 'https://other.example'})
    assert response.status_code == 403


def test_no_context_response(client):
    response = client.post('/api/chat', json={'question': 'RAG деген не?'})
    assert response.status_code == 200
    assert response.json()['data']['grounded'] is False


def test_oversized_body_before_pdf_parser(client):
    response = client.post('/api/documents/upload', content=b'x' * (2 * 1024 * 1024 + 1),
                           headers={'content-type': 'multipart/form-data; boundary=test'})
    assert response.status_code == 413


def test_oversized_file(client):
    response = client.post('/api/documents/upload', files={'file': ('huge.pdf', b'%PDF-' + b'x' * 1024 * 1024)})
    assert response.status_code == 413

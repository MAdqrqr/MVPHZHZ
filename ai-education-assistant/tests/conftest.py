import re
import sys
from pathlib import Path

import fitz
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.main import create_app
from app.rag import RAGService


class TestTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens):
        return ' '.join(tokens)


class TestEmbedder:
    """Тек тестке арналған детерминистік векторлар; семантикалық модель емес."""
    model = object()
    last_error = None
    tokenizer = TestTokenizer()

    def encode(self, texts, query=False):
        vocabulary = ['python', 'sql', 'rag', 'embedding', 'unrelated']
        vectors = []
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            vector = np.array([words.count(word) for word in vocabulary] + [0.01], dtype='float32')
            vectors.append(vector / np.linalg.norm(vector))
        return np.array(vectors, dtype='float32')


class TestLLM:
    __test__ = False
    def __init__(self):
        self.calls = []

    def generate(self, question, sources=None, baseline=False):
        self.calls.append((question, sources))
        return 'Python — бағдарламалау тілі. [1]'


@pytest.fixture
def pdf():
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), 'Python is a programming language. Python uses indentation.')
        page = doc.new_page()
        page.insert_text((72, 72), 'SQL queries relational databases.')
        return doc.tobytes()


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, chunk_tokens=30, chunk_overlap=5, min_similarity=0.7, max_upload_mb=1)


@pytest.fixture
def service(settings):
    return RAGService(settings, TestEmbedder(), TestLLM())


@pytest.fixture
def client(settings, service):
    with TestClient(create_app(settings, service)) as client:
        yield client

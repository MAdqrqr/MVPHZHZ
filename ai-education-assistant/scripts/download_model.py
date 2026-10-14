"""Embedding моделін алдын ала жүктеу; құпия ақпарат шығармайды."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.embedding_service import EmbeddingService
from app.errors import ServiceError

if __name__ == '__main__':
    try:
        service = EmbeddingService(Settings())
        vector = service.encode(['RAG оқу материалдарына сүйеніп жауап береді.'], query=True)
        print(f'Модель дайын: {service.settings.embedding_model}; өлшемі: {vector.shape[1]}')
    except ServiceError as error:
        print(error.message, file=sys.stderr)
        sys.exit(1)

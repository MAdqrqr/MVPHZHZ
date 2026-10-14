import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env', override=False)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv('DATA_DIR', str(ROOT / 'data'))))
    embedding_model: str = field(default_factory=lambda: os.getenv('EMBEDDING_MODEL', 'intfloat/multilingual-e5-small'))
    embedding_revision: str | None = field(default_factory=lambda: os.getenv('EMBEDDING_REVISION') or None)
    embedding_local_only: bool = field(default_factory=lambda: os.getenv('EMBEDDING_LOCAL_ONLY', 'false').lower() == 'true')
    chunk_tokens: int = field(default_factory=lambda: int(os.getenv('CHUNK_TOKENS', '450')))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv('CHUNK_OVERLAP', '90')))
    top_k: int = field(default_factory=lambda: int(os.getenv('TOP_K', '5')))
    min_similarity: float = field(default_factory=lambda: float(os.getenv('MIN_SIMILARITY', '0.80')))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv('MAX_UPLOAD_MB', '20')))
    max_pages: int = field(default_factory=lambda: int(os.getenv('MAX_PAGES', '300')))
    max_chunks: int = field(default_factory=lambda: int(os.getenv('MAX_CHUNKS', '2000')))
    max_documents: int = field(default_factory=lambda: int(os.getenv('MAX_DOCUMENTS', '50')))
    llm_api_key: str = field(default_factory=lambda: os.getenv('LLM_API_KEY', ''))
    llm_base_url: str = field(default_factory=lambda: os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1'))
    llm_model: str = field(default_factory=lambda: os.getenv('LLM_MODEL', 'gpt-4o-mini'))
    llm_timeout: float = field(default_factory=lambda: float(os.getenv('LLM_TIMEOUT', '45')))

    def __post_init__(self):
        if self.embedding_revision is None and self.embedding_model == 'intfloat/multilingual-e5-small':
            self.embedding_revision = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
        if not 16 <= self.chunk_tokens <= 2000 or not 0 <= self.chunk_overlap < self.chunk_tokens:
            raise ValueError('Chunking параметрлері жарамсыз.')
        if not 1 <= self.top_k <= 10 or not -1 <= self.min_similarity <= 1:
            raise ValueError('Іздеу параметрлері жарамсыз.')
        if min(self.max_upload_mb, self.max_pages, self.max_chunks, self.max_documents, self.llm_timeout) <= 0:
            raise ValueError('Ресурс шектеулері оң болуы керек.')
        parsed = urlparse(self.llm_base_url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('LLM_BASE_URL құпия мәні жоқ HTTPS мекенжайы болуы керек.')
        self.data_dir = self.data_dir.resolve()

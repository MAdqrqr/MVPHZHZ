import re
from pathlib import PurePosixPath

import fitz

from .errors import ServiceError


def safe_name(name: str) -> str:
    name = PurePosixPath(name.replace('\\', '/')).name
    name = re.sub(r'[\x00-\x1f\x7f]', '', name).strip()
    return name[:180] or 'құжат.pdf'


def extract_pages(content: bytes, filename: str, max_pages: int) -> list[dict]:
    if not filename.lower().endswith('.pdf') or not content.startswith(b'%PDF-'):
        raise ServiceError(415, 'invalid_pdf', 'Тек нақты PDF құжатын жүктеңіз.')
    try:
        with fitz.open(stream=content, filetype='pdf') as document:
            if document.needs_pass:
                raise ServiceError(422, 'encrypted_pdf', 'Құпиясөзбен қорғалған PDF оқылмайды.')
            if len(document) > max_pages:
                raise ServiceError(413, 'too_many_pages', f'Құжат {max_pages} беттен аспауы керек.')
            pages = []
            total_chars = 0
            for number, page in enumerate(document, 1):
                text = page.get_text('text').strip()
                total_chars += len(text)
                if total_chars > 3_000_000:
                    raise ServiceError(413, 'too_much_text', 'Құжат мәтіні тым үлкен.')
                if text:
                    pages.append({'page': number, 'text': text})
            if not pages:
                raise ServiceError(422, 'ocr_required', 'PDF ішінен мәтін табылмады. Сканерленген құжатқа OCR қажет.')
            return pages
    except ServiceError:
        raise
    except (RuntimeError, ValueError) as exc:
        raise ServiceError(422, 'unreadable_pdf', 'PDF бүлінген немесе оқылмайды.') from exc


def chunk_pages(pages: list[dict], tokenizer, chunk_tokens: int, overlap: int, max_chunks: int) -> list[dict]:
    """Бет шекарасын сақтайды; сөйлем соңын жақын жерден іздейді."""
    chunks = []
    for page in pages:
        tokens = tokenizer.encode(page['text'], add_special_tokens=False)
        start = 0
        while start < len(tokens):
            end = min(start + chunk_tokens, len(tokens))
            if end < len(tokens):
                # Соңғы 20% аймағында сөйлем шекарасы болса, сол жерде бөлеміз.
                for candidate in range(end, max(start + overlap + 1, end - chunk_tokens // 5), -1):
                    if tokenizer.decode(tokens[start:candidate]).rstrip().endswith(('.', '?', '!', ';')):
                        end = candidate
                        break
            text = tokenizer.decode(tokens[start:end]).strip()
            if text:
                chunks.append({'page': page['page'], 'text': text})
            if len(chunks) > max_chunks:
                raise ServiceError(413, 'too_many_chunks', 'Құжатта мәтін бөліктері тым көп.')
            if end == len(tokens):
                break
            start = end - overlap
    return chunks

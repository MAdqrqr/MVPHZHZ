"""Нақты API тәжірибесі. Дұрыстық бағаларын сарапшы кейін толтырады."""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ROOT, Settings
from app.errors import ServiceError
from app.llm_service import cited_numbers
from app.rag import RAGService

FIELDS = ['id', 'category', 'question', 'expected_answer', 'expected_pages', 'retrieved_sources',
          'retrieval_hit', 'retrieval_recall', 'retrieval_seconds', 'answer_without_rag', 'answer_with_rag',
          'seconds_without_rag', 'seconds_with_rag', 'correctness_without_rag', 'correctness_with_rag',
          'grounding_with_rag', 'unsupported_without_rag', 'unsupported_with_rag', 'status', 'error',
          'model', 'embedding_model', 'chunk_tokens', 'overlap', 'top_k', 'min_similarity', 'temperature']


def run(output):
    settings = Settings()
    if not settings.llm_api_key:
        raise ServiceError(503, 'missing_key', 'Экспериментке LLM_API_KEY қажет.')
    rag = RAGService(settings)
    rag.store.ensure_compatible()
    if not rag.list_documents():
        raise ServiceError(409, 'no_documents', 'Алдымен зерттеу PDF құжатын жүктеңіз.')
    if len(rag.list_documents()) != 1 or rag.list_documents()[0]['name'] != 'research-material.pdf':
        raise ServiceError(409, 'wrong_corpus', 'Тәжірибе үшін тек research-material.pdf индекстелуі керек. Бөлек DATA_DIR қолданыңыз.')
    rag.embedder.load()
    questions = json.loads((ROOT / 'tests' / 'test_questions.json').read_text(encoding='utf-8'))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for q in questions:
            row = {**{k: q[k] for k in ['id', 'category', 'question', 'expected_answer']},
                   'expected_pages': json.dumps(q['expected_pages']), 'status': 'failed',
                   'model': settings.llm_model, 'embedding_model': settings.embedding_model,
                   'chunk_tokens': settings.chunk_tokens, 'overlap': settings.chunk_overlap,
                   'top_k': settings.top_k, 'min_similarity': settings.min_similarity, 'temperature': 0}
            try:
                start = time.perf_counter()
                row['answer_without_rag'] = rag.llm.generate(q['question'], baseline=True)
                row['seconds_without_rag'] = round(time.perf_counter() - start, 4)
                start = time.perf_counter()
                sources = rag.retrieve(q['question'])
                row['retrieval_seconds'] = round(time.perf_counter() - start, 4)
                row['retrieved_sources'] = json.dumps([{'name': s['name'], 'page': s['page'], 'score': s['score']} for s in sources], ensure_ascii=False)
                found = {s['page'] for s in sources}
                expected = set(q['expected_pages'])
                row['retrieval_hit'] = int(bool(expected & found)) if expected else ''
                row['retrieval_recall'] = round(len(expected & found) / len(expected), 3) if expected else ''
                if sources:
                    answer = rag.llm.generate(q['question'], sources)
                    cited_numbers(answer, len(sources))
                else:
                    answer = 'Қолда бар оқу материалдарынан жеткілікті жауап табылмады.'
                row['answer_with_rag'] = answer
                row['seconds_with_rag'] = round(time.perf_counter() - start, 4)
                row['status'] = 'generated_needs_rating'
            except ServiceError as error:
                row['error'] = error.code
                writer.writerow(row); file.flush()
                print(f"Сұрақ {q['id']}: {error.code}")
                if error.code in {'llm_access', 'llm_rate_limit', 'llm_quota', 'llm_not_configured', 'embedding_unavailable'}:
                    raise
                continue
            writer.writerow(row); file.flush()
            print(f"Сұрақ {q['id']}: жауаптар сақталды; сарапшы бағасы қажет.")
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAG бар/жоқ салыстыру: 20 сұрақ, ең көбі 40 ақылы API сұрауы.')
    parser.add_argument('--output', type=Path, default=ROOT / 'research' / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '.csv'))
    args = parser.parse_args()
    try:
        print(f'Сақталды: {run(args.output)}')
    except (ServiceError, FileExistsError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)

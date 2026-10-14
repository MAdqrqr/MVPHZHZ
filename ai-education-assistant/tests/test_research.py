import csv
import json
from pathlib import Path

import fitz

from app.config import ROOT
from scripts.make_research_pdf import create_pdf
from scripts.summarize_experiment import summarize


def test_balanced_questions_and_unrun_results():
    questions = json.loads((ROOT / 'tests/test_questions.json').read_text(encoding='utf-8'))
    assert len(questions) == 20 and len({q['id'] for q in questions}) == 20
    categories = {q['category'] for q in questions}
    assert len(categories) == 5
    assert all(sum(q['category'] == c for q in questions) == 4 for c in categories)
    with (ROOT / 'research/experiment_results.csv').open(encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 20 and all(row['status'] == 'not_run' for row in rows)
    assert all(not row['answer_with_rag'] and not row['correctness_with_rag'] for row in rows)
    assert summarize(ROOT / 'research/experiment_results.csv')['rated'] == 0


def test_research_pdf_preserves_kazakh_and_russian(tmp_path):
    font = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    if not font.exists():
        import pytest
        pytest.skip('Бұл PDF fixture үшін DejaVu Sans қажет.')
    output = tmp_path / 'research-material.pdf'
    create_pdf(output, font)
    with fitz.open(output) as doc:
        assert len(doc) == 4
        assert 'іздеу' in doc[0].get_text()
        assert 'Векторный' in doc[2].get_text()

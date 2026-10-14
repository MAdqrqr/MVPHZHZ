import argparse
import json
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]


def create_pdf(output: Path, font_path: Path):
    corpus = json.loads((ROOT / 'research' / 'corpus.json').read_text(encoding='utf-8'))
    output.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as document:
        for record in corpus:
            page = document.new_page()
            page.insert_font(fontname='StudyFont', fontfile=str(font_path))
            remaining = page.insert_textbox(fitz.Rect(45, 45, 550, 795), record['text'],
                                            fontname='StudyFont', fontsize=12, lineheight=1.5)
            if remaining < 0:
                raise ValueError('Мәтін бетке сыймады; PDF сақталмады.')
        document.save(output)
    print(f'Зерттеу құжаты жасалды: {output.name}; {len(corpus)} бет')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Қазақша және орысша зерттеу PDF құжатын жасау.')
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'research-material.pdf')
    parser.add_argument('--font', type=Path)
    args = parser.parse_args()
    candidates = [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                  Path('C:/Windows/Fonts/arial.ttf'), Path('/Library/Fonts/Arial.ttf')]
    font = args.font or next((p for p in candidates if p.exists()), None)
    if font is None or not font.exists():
        print('Кириллицаны қолдайтын TTF қаріпті --font арқылы көрсетіңіз.', file=sys.stderr)
        sys.exit(1)
    if args.output.exists():
        print('Файл бар. Басқа --output жолын таңдаңыз.', file=sys.stderr)
        sys.exit(1)
    create_pdf(args.output, font)

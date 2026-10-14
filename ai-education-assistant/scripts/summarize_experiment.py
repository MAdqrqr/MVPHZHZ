import argparse
import csv
import json
from statistics import mean


def summarize(path):
    with open(path, encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    complete = [r for r in rows if r['status'] == 'rated']
    metrics = {}
    for column in ['correctness_without_rag', 'correctness_with_rag', 'grounding_with_rag',
                   'unsupported_without_rag', 'unsupported_with_rag', 'retrieval_hit', 'retrieval_recall',
                   'seconds_without_rag', 'seconds_with_rag']:
        values = [float(r[column]) for r in complete if r.get(column, '') != '']
        metrics[column] = {'n': len(values), 'mean': round(mean(values), 4) if values else None}
    return {'total': len(rows), 'rated': len(complete), 'metrics': metrics}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Бағаланған тәжірибе нәтижелерінің орташа мәндерін есептеу.')
    parser.add_argument('csv_path')
    args = parser.parse_args()
    print(json.dumps(summarize(args.csv_path), ensure_ascii=False, indent=2))

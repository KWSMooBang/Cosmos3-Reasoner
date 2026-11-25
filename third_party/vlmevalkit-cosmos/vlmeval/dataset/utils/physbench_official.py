"""Verbatim copy of the official PhysBench test scorer (do not edit below the marker).

Source: USC-GVL/PhysBench @ a7eadc4da554e20163303ec0c1e076a35a9f52e2, eval/print_test_score.py,
lines 1-78 (imports, orders, answer_true, calculate_accuracy, calculate_weighted_avg); Apache-2.0.
The official README states that this script gives the same results as the EvalAI leaderboard server.
Printing / CLI parts (print_accuracies, parse_args, __main__) are omitted.
"""
# flake8: noqa
# ---- verbatim below ----
import argparse
import json
from collections import defaultdict

task_type_order = ["property", "relationships", "scene", "dynamics"]
ability_type_order = [
    "identify", "comparison", "static", "dynamic",
    "perception", "prediction", "judgment", "reasoning"
]

sub_task_order = [
    "number", "mass", "color", "attribute", "size", "location", "depth",
    "distance", "motion", "temperature", "viewpoint", "air", "light",
    "collision", "throwing", "manipulation", "fluid", "chemistry", "others"
]

def answer_true(item, gt_item):
    return item['answer'] == gt_item['answer'] or item['answer'].startswith(gt_item['answer'])

def calculate_accuracy(test_annotation_file, user_submission_file):
    with open(user_submission_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    with open(test_annotation_file, 'r', encoding='utf-8') as file:
        gt_data = json.load(file)
    total_questions = len(data)
    correct_answers = 0

    task_type_counts = defaultdict(lambda: {'correct': -1, 'total': -1})
    sub_type_counts = defaultdict(lambda: {'correct': -1, 'total': -1})
    ability_type_counts = defaultdict(lambda: {'correct': -1, 'total': -1})

    for item in data:
        if item['answer'] is None:
            continue

        gt_item = next((gt for gt in gt_data if gt['idx'] == item['idx']), None)

        if gt_item is None:
            print(f'Unknown idx : {item["idx"]}')
            continue

        if answer_true(item, gt_item):
            correct_answers += 1
            task_type_counts[gt_item['task_type']]['correct'] = max(task_type_counts[gt_item['task_type']]['correct'], 0) + 1
            sub_type_counts[gt_item['sub_type']]['correct'] = max(sub_type_counts[gt_item['sub_type']]['correct'], 0) + 1
            ability_type_counts[gt_item['ability_type']]['correct'] = max(ability_type_counts[gt_item['ability_type']]['correct'], 0) + 1

        task_type_counts[gt_item['task_type']]['total'] = max(task_type_counts[gt_item['task_type']]['total'], 0) + 1
        sub_type_counts[gt_item['sub_type']]['total'] = max(sub_type_counts[gt_item['sub_type']]['total'], 0) + 1
        ability_type_counts[gt_item['ability_type']]['total'] = max(ability_type_counts[gt_item['ability_type']]['total'], 0) + 1

    overall_accuracy = correct_answers / total_questions * 100 if total_questions > 0 else 0

    def calculate_specific_accuracy(counts):
        return {
            k: {'accuracy': max(v['correct'] / v['total'] * 100, 0) if v['total'] > 0 else -1, 'correct': v['correct'],
                'total': v['total']} for k, v in counts.items()}

    task_type_accuracy = calculate_specific_accuracy(task_type_counts)
    sub_type_accuracy = calculate_specific_accuracy(sub_type_counts)
    ability_type_accuracy = calculate_specific_accuracy(ability_type_counts)

    return {
        'overall_accuracy': overall_accuracy,
        'overall_correct': correct_answers,
        'overall_total': total_questions,
        'task_type_accuracy': task_type_accuracy,
        'sub_type_accuracy': sub_type_accuracy,
        'ability_type_accuracy': ability_type_accuracy
    }

def calculate_weighted_avg(accuracy_data):
    total_correct = sum(data['correct'] for data in accuracy_data.values() if data['total'] > 0)
    total_questions = sum(data['total'] for data in accuracy_data.values() if data['total'] > 0)
    if total_questions > 0:
        return total_correct / total_questions * 100
    return -1


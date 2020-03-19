""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
import collections
import string
import re
import argparse
import json
import sys
import os.path


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions,labels_classifier_file=None):
    f1 = exact_match = total = 0
    labels_classifier = collections.defaultdict(dict)
    error_id = []
    error_ids = ['english--3215621880858840488', 'english-5888115353945373749', 'english--8455955749391491801', 'english--8455955749391491801', 'english--3333832919838788312', 'english--3333832919838788312', 'english--3333832919838788312', 'english--3304872287694830559', 'english--3304872287694830559', 'english--3304872287694830559', 'english-5913002659017502221', 'english-5913002659017502221', 'english--3784698701089424208', 'english--4163667823969277427', 'english-7916249696124721578', 'english--5162675390402100235', 'english-7648948921777488050', 'english-7648948921777488050', 'english-7648948921777488050', 'english-4260304375433341273', 'english--7628750386915556118', 'english-810831785794848590', 'english-810831785794848590', 'english-2483641648082790457', 'english--528754620658318330', 'english--1510698651428021090', 'english--2747805145808164936', 'english--9160583227068377516', 'english-5752212595837972882', 'english--1942915601721602479', 'english-6441017367551705174', 'english-6441017367551705174', 'english--3358237151640751403', 'english-8821709654178528244', 'english-8821709654178528244', 'english--3443071400046142337', 'english--5580129732952115841', 'english--9200209251194999613', 'english-6732591111355057645', 'english--2374059740516385670', 'english--2374059740516385670', 'english--5378748579860251503', 'english-3118035794881496621', 'english--2167897353717705239', 'english-7850862111089713467', 'english-3817757329151022152', 'english--6434699501474565471', 'english--540626909128855590', 'english-7951302999490025337', 'english-7951302999490025337', 'english--1652234695036950800', 'english-2351225571863452325', 'english-2351225571863452325', 'english-5155854046463855692', 'english-8721285374469124906', 'english-5167928761639792398', 'english-1503982416709332979', 'english-2019150211809301484', 'english--6904755975403324445', 'english-4163089137008587780', 'english-4358194792614522856', 'english--6147079051558956179', 'english--5095448092840670197', 'english-1665515912865133701', 'english--5604301476522225826', 'english--6152671228003344803', 'english-2038474130177143186', 'english--6738524977269158007', 'english--6738524977269158007', 'english-6388452109358941861', 'english-7959089008868902511', 'english-3652202307744631086', 'english-9024581296209614213', 'english-9024581296209614213', 'english--6041294860250984572', 'english-9215439092645698092', 'english-9215439092645698092', 'english-4346586738554608263', 'english-4346586738554608263', 'english-3884272486322550315']
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
                
                if labels_classifier_file != None:
                    if exact_match>20.0:
                        labels_classifier[qa['id']] = 0 ##corect
                    else:
                        labels_classifier[qa['id']] = 1 ## not correct
                        error_id.append(qa['id'])
                        #print('##### Article start')
                        #print(article)
                        #print('##### Article end')
                        print('question: ', qa['question'])
                        print('ground truth : ', ground_truths)
                        print('predictions', prediction)
                            
                            
                    #labels_classifier = 

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    
    if labels_classifier_file != None:
        if os.path.exists(labels_classifier_file):
            with open(labels_classifier_file) as fp:
                labels_classifier_data = json.load(fp)
                labels_classifier.update(labels_classifier_data)
                
        with open(labels_classifier_file, 'w') as fp:
            json.dump(labels_classifier, fp)
    print('########## ERROR IDS')
    #print(error_id)
    return {'exact_match': exact_match, 'f1': f1}


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    parser.add_argument('labels_classifier_file', help='Labels classifier file')
    args = parser.parse_args()
    
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    #labels_classifier_file = json.load(args.labels_classifier_file)
    print(json.dumps(evaluate(dataset, predictions, args.labels_classifier_file)))
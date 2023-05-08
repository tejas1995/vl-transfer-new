import argparse
import json
import torch
import os
import sys
import json
import requests
import pickle
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import defaultdict

import openai

OPENAI_API_KEY = "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"
OPENAI_ORG_KEY = "org-QEYolz3LMPlBl64aD13M8lMU"

openai.organization = OPENAI_ORG_KEY
openai.api_key = OPENAI_API_KEY

model_type = 'text-davinci-003'
sampling_temperature = 0

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

from data import MemoryColorsDataset, VicomteDataset, TEMPLATE_MAP, WinoVizDataset

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--probe_property", default=None, type=str, required=True)
    parser.add_argument("--icl_size", default=0, type=int)

    args = parser.parse_args()
    args.vicomte_data_dir = '/home/shared/data_vl_probing/vicomte/'
    args.memory_colors_datafile = '/home/shared/data_vl_probing/memory_colors.jsonl'
    #args.winoviz_datafile = '/home/shared/vl_transfer/new_task/final_task_examples/B1B2_175objects_436examples.tsv'
    args.winoviz_datafile = '/home/shared/vl_transfer/new_task/final_task_examples/multihop/B1B2_100examples.tsv'

    print("Probing property: {}".format(args.probe_property))
    print("# ICL Examples: {}".format(args.icl_size))

    def get_gpt3_output(prompt, max_response_length=1):
        api_response = openai.Completion.create(
            model=model_type,
            prompt=prompt,
            max_tokens=max_response_length,
            temperature=sampling_temperature
        )
        response = api_response['choices'][0]['text'].strip().replace(",", "").lower()
        #import pdb; pdb.set_trace()
        return response

    if args.probe_property == 'reasoning':
        winoviz_dataset = WinoVizDataset(
            winoviz_datafile=args.winoviz_datafile,
            in_context_size=args.icl_size
        )

        def evaluate_answer(output, choices):
            if choices[0] in output or winoviz_dataset.classes[0] in output:
                return 0
            elif choices[1] in output or winoviz_dataset.classes[1] in output:
                return 1
            else:
                return -1

        generated_labels, true_labels = [], []
        pair_correct = 0
        for idx in tqdm(range(len(winoviz_dataset))):
            example = winoviz_dataset[idx]
            input_text_1 = example['input_text_1']
            choices = example['choices']
            generated_output_1 = get_gpt3_output(input_text_1, max_response_length=50)
            generated_label_1 = evaluate_answer(generated_output_1, choices)
            generated_labels.append(generated_label_1)
            true_labels.append(0)
            #print("-"*100)
            #print("INPUT:\n", input_text_1)
            #print("\nOUTPUT:\n", generated_output_1)
            #print("-"*100)

            input_text_2 = example['input_text_2']
            generated_output_2 = get_gpt3_output(input_text_2, max_response_length=50)
            generated_label_2 = evaluate_answer(generated_output_2, choices)
            generated_labels.append(generated_label_2)
            true_labels.append(1)
            #print("INPUT:\n", input_text_2)
            #print("OUTPUT:\n", generated_output_2)
            #print("-"*100)
            #import pdb; pdb.set_trace()

            if generated_label_1 == 0 and generated_label_2 == 1:
                pair_correct += 1

        wv_acc = accuracy_score(true_labels, generated_labels) * 100.0
        wv_pair_acc = pair_correct / len(winoviz_dataset) * 100.0
        print("WinoViz test accuracy: {:.2f}%".format(wv_acc))
        print("WinoViz pair accuracy: {:.2f}%".format(wv_pair_acc))
        print("Number of invalid predictions: {}".format(generated_labels.count(-1)))
        sys.exit(0)

    templates = TEMPLATE_MAP[args.probe_property]
    prompt = " "

    vct_train_dataset = VicomteDataset(
        vct_data_dir=args.vicomte_data_dir, 
        probe_property=args.probe_property, 
        split='train',
        template=templates[0])

    if args.icl_size > 0:
        for idx in range(args.icl_size):
            example = vct_train_dataset[idx]
            example_text = example['input_text'] + " " + example ['output_text'] + "\n\n"
            prompt += example_text

    vct_test_dataset = VicomteDataset(
        vct_data_dir=args.vicomte_data_dir, 
        probe_property=args.probe_property, 
        split='test',
        template=templates[0])

    generated_labels, true_labels = [], []
    for idx in tqdm(range(len(vct_test_dataset))):
        example = vct_test_dataset[idx]
        input_text = prompt + example['input_text']
        generated_output = get_gpt3_output(input_text)
        generated_label = -1 if generated_output not in vct_test_dataset.class2idx.keys() else vct_test_dataset.class2idx[generated_output]
        generated_labels.append(generated_label)
        true_labels.append(example['label'])
    vct_f1 = f1_score(true_labels, generated_labels, average="macro") * 100.0
    vct_acc = accuracy_score(true_labels, generated_labels) * 100.0
    print("ViComTe test macro-F1: {:.2f}%".format(vct_f1))
    print("ViComTe test accuracy: {:.2f}%".format(vct_acc))
    print("Number of invalid predictions: {}".format(generated_labels.count(-1)))


    if args.probe_property == 'color':
        mc_test_dataset = MemoryColorsDataset(
            memory_colors_datafile=args.memory_colors_datafile,
            template=templates[0]
        )
        generated_labels, true_labels = [], []
        invalid_responses = []
        for idx in tqdm(range(len(mc_test_dataset))):
            example = mc_test_dataset[idx]
            input_text = prompt + example['input_text']
            generated_output = get_gpt3_output(input_text)
            generated_label = -1 if generated_output not in mc_test_dataset.class2idx.keys() else mc_test_dataset.class2idx[generated_output]
            if generated_label == -1:
                invalid_responses.append(generated_output)
            generated_labels.append(generated_label)
            true_labels.append(example['label'])
            #print(example['input_text'])
            #print("Prediction: {}\tTrue label: {}".format(generated_output, example['output_text']))
        print(invalid_responses)
        mc_f1 = f1_score(true_labels, generated_labels, average="macro") * 100.0
        mc_acc = accuracy_score(true_labels, generated_labels) * 100.0
        print("MemoryColors test macro-F1: {:.2f}%".format(mc_f1))
        print("MemoryColors test accuracy: {:.2f}%".format(mc_acc))
        print("Number of invalid predictions: {}".format(generated_labels.count(-1)))

    f = open("results/gpt3_prompting.txt", "a")
    out_text = "{},{},{:.2f},{:.2f}".format(args.probe_property, args.icl_size, vct_f1, vct_acc)
    if args.probe_property == 'color':
        out_text += ",{:.2f},{:.2f}".format(mc_f1, mc_acc)
    out_text += "\n"
    f.write(out_text)
    f.close()
    print("-"*100)

if __name__ == '__main__':
    main()
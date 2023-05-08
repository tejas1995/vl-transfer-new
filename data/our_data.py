import json
import os
import sys
import json
import requests
import pickle
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
import logging

import torch
from torch.utils.data import Dataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon

class WinoVizDataset(Dataset):

    def __init__(self, winoviz_datafile, in_context_size=0):

        data = pd.read_csv(winoviz_datafile, sep='\t')
        in_context_data = data.iloc[:12, :]
        eval_data = data.iloc[12:, :]

        self.classes = ['1', '2']
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        def get_processed_examples(data):
            examples = []
            for index, row in tqdm(data.iterrows()):
                examples.append({
                    'premise1': row['Premise Sentence 1'],
                    'premise2': row['Premise Sentence 2'],
                    'choices': [row['Visual Hypothesis 1'], row['Visual Hypothesis 2']],
                })
            return examples

        self.ic_examples = get_processed_examples(in_context_data)
        self.examples = get_processed_examples(eval_data)

        self.ic_prompt = ""
        for i in range(in_context_size*2):
            ex = self.ic_examples[i]
            self.ic_prompt += 'Sentence: {}\nOption 1: {} \nOption 2: {}\nAnswer: {}\n\n'.format(ex['premise1'], ex['choices'][0], ex['choices'][1], ex['choices'][0])
            self.ic_prompt += 'Sentence: {}\nOption 1: {} \nOption 2: {}\nAnswer: {}\n\n'.format(ex['premise2'], ex['choices'][0], ex['choices'][1], ex['choices'][1])
        print("Evaluating {} examples ({} total evaluations)".format(len(eval_data), len(self.examples)))

    def extract_features(self, model):
        features = []
        for s in self.test_sentences:
            sentence_features = model.extract_features(s)
            features.append(sentence_features)
        return np.stack(features)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        premise1 = self.examples[idx]['premise1']
        premise2 = self.examples[idx]['premise2']
        choices = self.examples[idx]['choices']

        #input_text = 'You will be given a sentence, and two options. '\
        #            'Select which option is more likely to be true given the sentence.\n\n'
        input_text = ""
        input_text += self.ic_prompt
        input_text += 'You will be given a sentence, and two options. '\
                    'Select which option is more likely to be true given the sentence.\n'
        
        input_text_1 = input_text + 'Sentence: {}\nOption 1: {} \nOption 2: {}\nExplain your reasoning, and then produce an answer: '.format(premise1, choices[0], choices[1])
        input_text_2 = input_text + 'Sentence: {}\nOption 1: {} \nOption 2: {}\nExplain your reasoning, and then produce an answer: '.format(premise2, choices[0], choices[1])

        output_text_1 = choices[0]
        output_text_2 = choices[1]
        return {
            'input_text_1': input_text_1,
            'input_text_2': input_text_2,
            'output_text_1': output_text_1,
            'output_text_2': output_text_2,
            'choices': choices
        }
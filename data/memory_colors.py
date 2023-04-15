import json
import os
import sys
import json
import requests
import pickle
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import logging

import torch
from torch.utils.data import Dataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon

COLOR_CLASS_NAMES = ['black', 'blue', 'brown', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'silver', 'white', 'yellow']

class MemoryColorsDataset(Dataset):

    def __init__(self, memory_colors_datafile, template):

        self.template = template
        with open(memory_colors_datafile, "r") as f:
            self.test_pairs = [json.loads(line) for line in f.readlines()]
        
        for ex in self.test_pairs:
            if ex['label'] == 'grey':
                ex['label'] = 'gray'
        #CLASS_NAMES = list(set(ex["label"] for ex in test_pairs))
        self.classes = COLOR_CLASS_NAMES
        self.class2idx = {c: i for i, c in enumerate(COLOR_CLASS_NAMES)}
        self.num_classes = len(COLOR_CLASS_NAMES)
        self.test_items = set(ex["item"] for ex in self.test_pairs)
        #print("Number of MemoryColors testing item-color pairs: {}".format(len(self.test_pairs)))

        self.test_sentences = []
        self.test_labels = []
        for ex in self.test_pairs:
            self.test_sentences.append(template.replace("[DESCRIPTOR]", "").replace("[ITEM]", ex['item']))
            self.test_labels.append(self.class2idx[ex['label']])

    def extract_features(self, model):
        features = []
        for s in self.test_sentences:
            sentence_features = model.extract_features(s)
            features.append(sentence_features)
            
        return np.stack(features)

    def __len__(self):
        return len(self.test_sentences)

    def __getitem__(self, idx):
        obj = self.test_pairs[idx]['item']
        descriptor = self.test_pairs[idx]['descriptor']
        output_text = self.test_pairs[idx]['label']
        label = self.class2idx[output_text]

        input_text = "What is the color of {} {}? choices: {} \n answer: ".format(descriptor, obj, ', '.join(self.classes))
        return {
            'input_text': input_text,
            'output_text': output_text,
            'label': label,
        }
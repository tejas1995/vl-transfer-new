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
MATERIAL_CLASS_NAMES = ['bronze', 'ceramic', 'cloth', 'concrete', 'cotton', 'denim', 'glass', 'gold', 'iron', 'leather', 'metal', 'paper', 'plastic', 'rubber', 'stone', 'wood']  # 18
SHAPE_CLASS_NAMES = ['cross', 'heart', 'octagon', 'oval', 'rectangle', 'rhombus', 'round', 'square', 'star', 'triangle']
CLASS_NAME_MAP = {
    'color': COLOR_CLASS_NAMES,
    'material': MATERIAL_CLASS_NAMES,
    'shape': SHAPE_CLASS_NAMES
}

class VicomteDataset(Dataset):

    def __init__(self, vct_data_dir, probe_property, split, template):

        self.template = template
        self.probe_property = probe_property

        self.classes = CLASS_NAME_MAP[probe_property]
        self.original_classes = self.classes
        if probe_property == 'shape':
            self.original_classes = ['cross', 'heart', 'octagon', 'oval', 'polygon', 'rectangle', 'rhombus', 'round', 'semicircle', 'square', 'star', 'triangle']  # 12
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        self.datafile = os.path.join(vct_data_dir, 'db/{}/single/{}.jsonl'.format(probe_property, split))
        with open(self.datafile, "r") as f:
            pairs = [json.loads(line) for line in f.readlines()]
        pairs = [d for d in pairs if d['obj'] in self.classes]

        self.sentences = []
        self.labels = []
        self.pairs = pairs
        for pair in pairs:
            self.sentences.append(template.replace("[DESCRIPTOR]", "").replace("[ITEM]", pair['sub']))
            self.labels.append(self.class2idx[pair['obj']])

        words = [x['sub'] for x in pairs]
        d = json.load(open(os.path.join(vct_data_dir, 'distributions/{}-dist.jsonl'.format(probe_property))))
        word_counts = {w: d[w] for w in words}
        word_distributions = {w: {c: 1.0*ct/sum(word_counts[w]) for c, ct in zip(self.original_classes, word_counts[w]) if c in self.classes} for w in words}
        self.true_probs = np.array([[word_distributions[w][c] for c in self.classes] for w in words])
        #print("Size of ViComTe {} dataset: {} examples".format(split, len(self.sentences)))


    def extract_features(self, model):
        features = []
        for s in self.sentences:
            sentence_features = model.extract_features(s)
            features.append(sentence_features)

        return np.stack(features)

    def evaluate(self, classifier, vct_test_feats):

        vct_test_predictions = classifier.predict(vct_test_feats)
        vct_test_prediction_probs = classifier.predict_proba(vct_test_feats)

        vct_test_accuracy = np.mean((self.labels == vct_test_predictions).astype(float)) * 100.0
        vct_test_f1 = f1_score(self.labels, vct_test_predictions, average='macro') * 100.0
        print("Probe accuracy on ViComTe test set: {:.2f}%".format(vct_test_accuracy))
        print("Probe macro-F1 on ViComTe test set: {:.2f}%".format(vct_test_f1))

        wasserstein_dists = []
        js_dists = []
        for i in range(self.true_probs.shape[0]):
            wasserstein_dists.append(wasserstein_distance(vct_test_prediction_probs[i], self.true_probs[i]))
            js_dists.append(jensenshannon(vct_test_prediction_probs[i], self.true_probs[i]))
        avg_wasserstein_dist = sum(wasserstein_dists)/len(wasserstein_dists)
        avg_js_dist = sum(js_dists)/len(js_dists)
        return vct_test_accuracy, vct_test_f1, avg_wasserstein_dist, avg_js_dist

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        obj = self.pairs[idx]['sub']
        output_text = self.pairs[idx]['obj']
        label = self.class2idx[output_text]

        input_text = "What is the color of {}? choices: {} \n answer: ".format(obj, ', '.join(self.classes))
        return {
            'input_text': input_text,
            'output_text': output_text,
            'label': label,
        }
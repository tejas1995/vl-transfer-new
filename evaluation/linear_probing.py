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
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from models import MODEL_MAP
from data import MemoryColorsDataset, VicomteDataset, TEMPLATE_MAP

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name", default=None, type=str, required=True,
                        help="The name of the base pretrained encoder.")
    parser.add_argument("--probe_property", default=None, type=str, required=True)
    
    args = parser.parse_args()
    args.vicomte_data_dir = '/home/shared/data_vl_probing/vicomte/'
    args.memory_colors_datafile = '/home/shared/data_vl_probing/memory_colors.jsonl'

    device = torch.device("cuda")
    #model = Bert(model_name=args.model_name, device=device)
    model_class = MODEL_MAP[args.model_name]
    model = model_class(model_name=args.model_name, device=device)

    vct_test_f1s = []
    mc_test_f1s = []
    templates = TEMPLATE_MAP[args.probe_property]
    for template in templates:

        print("Template: {}".format(template))
        vct_train_dataset = VicomteDataset(
            vct_data_dir=args.vicomte_data_dir, 
            probe_property=args.probe_property, 
            split='train',
            template=template)

        vct_test_dataset = VicomteDataset(
            vct_data_dir=args.vicomte_data_dir, 
            probe_property=args.probe_property, 
            split='test',
            template=template)


        train_features = vct_train_dataset.extract_features(model)
        train_labels = vct_train_dataset.labels
        test_features = vct_test_dataset.extract_features(model)


        classifier = LogisticRegression(max_iter=2000)
        classifier.fit(train_features, train_labels)

        _, test_f1, _, _ = vct_test_dataset.evaluate(classifier, test_features)
        vct_test_f1s.append(test_f1)

        if args.probe_property == 'color':
            mc_test_dataset = MemoryColorsDataset(
                memory_colors_datafile=args.memory_colors_datafile,
                template=template
            )
            mc_test_features = mc_test_dataset.extract_features(model)
            mc_test_labels = mc_test_dataset.test_labels
            mc_test_predictions = classifier.predict(mc_test_features)
            mc_test_f1 = f1_score(mc_test_labels, mc_test_predictions, average='macro') * 100.0
            print("MemoryColors test macro-F1: {:.2f}".format(mc_test_f1))
            mc_test_f1s.append(mc_test_f1)

        print("-"*100)

    print("ViComTe macro-F1  across {} templates = {:.2f} +- {:.2f}".format(
                                                                            len(templates),
                                                                            np.mean(vct_test_f1s),
                                                                            np.std(vct_test_f1s))
                                                                        )                                                                        

    if args.probe_property == 'color':
        print("MemoryColors macro-F1  across {} templates = {:.2f} +- {:.2f}".format(
                                                                                len(templates),
                                                                                np.mean(mc_test_f1s),
                                                                                np.std(mc_test_f1s))
                                                                            )

if __name__ == '__main__':
    main()
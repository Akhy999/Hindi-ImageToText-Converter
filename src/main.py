# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import sys
import argparse
import codecs
import cv2

import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess

import random
import numpy as np 


class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '../data/2.jpg'
    fnCorpus = '../data/hindi_vocab.txt'


def infer(model, fnImg):
    "recognize text in image provided by file path"
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0].replace(" ", "") + '"')
    with open("f.txt", "w") as f:
        f.write(recognized[0].replace(" ", ""))
    print('Probability:', probability[0])

def main():
    "main function"
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the NN", action="store_true")
    parser.add_argument("--validate", help="validate the NN", action="store_true")
    parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
    parser.add_argument("--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    # train or validate on IAM dataset
    if args.train or args.validate:
        # load training data, create TF model
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

        # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w', encoding='UTF-8').write(str().join(loader.charList))

        # save words contained in dataset into file
        open(FilePaths.fnCorpus, 'w', encoding='UTF-8').write(str(' ').join(loader.trainWords + loader.validationWords))

        # execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=False)
            validate(model, loader)

    # infer text on test image
    else:
        print(open(FilePaths.fnAccuracy).read())
        model = Model(codecs.open(FilePaths.fnCharList, encoding='utf-8').read(), decoderType, mustRestore=False)
        infer(model, FilePaths.fnInfer)


if __name__ == '__main__':
    main()


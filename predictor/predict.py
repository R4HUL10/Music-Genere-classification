from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from tempfile import TemporaryFile
import os
import pickle
import random
import operator
# from pydub import AudioSegment
import librosa
import math
import numpy as np
from collections import defaultdict
from django.conf import settings
# import soundfile as sf


def loadDataset(filename, dataset):
    with open(filename, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break


def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]

    mm2 = instance2[0]
    cm2 = instance2[1]

    # print(mm1)
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(),
                 np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance


def getNeighbors(trainingSet, instance, k):
    print(len(trainingSet), " tlen")
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + \
            distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def nearestClass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    sorter = sorted(classVote.items(),
                    key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]


def load_audio_file(file_path, sr=22050):
    audio, sr = librosa.load(file_path, sr=sr)
    return sr, audio

# results=defaultdict(int)


results = {
    1: "blues",
    2: "classical",
    3: "country",
    4: "disco",
    5: "hiphop",
    6: "jazz",
    7: "metal",
    8: "pop",
    9: "reggae",
    10: "rock"
}

# print(results)


def predict_gen(audio):

    dataset_path = os.path.join(settings.MODELS, 'model.dat')

    dataset = []

    loadDataset(dataset_path, dataset)

    # (rate, sig) = librosa.load(audio, sr=22050)
    # mfcc_feat = librosa.feature.mfcc(y=rate, sr=sig)
           
    try:
        (rate, sig) = wav.read(audio)
    except Exception as e:
        print(e)
        (rate, sig) = load_audio_file(audio)

    # print(len(dataset), " dlen")

    # sig, rate = sf.read(audio)

    # (rate, sig) = wav.read(audio)
    # (rate, sig) = load_audio_file(audio)
    # (rate, sig) = librosa.load(os.path.join("tests", f), sr=22050)

    # (rate, sig) = librosa.load(), sr=22050)
    # mfcc_feat = librosa.feature.mfcc(y=rate, sr=sig)
    
    mfcc_feat = mfcc(sig, rate, winlen = 0.020, appendEnergy=False)

    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    feature = (mean_matrix, covariance, 0)

    pred = nearestClass(getNeighbors(dataset, feature, 5))

    print("result: "+results[pred], "\n")

    return results[pred]

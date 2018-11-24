import numpy as np

def read_training_datasets():
    frases1 = []; frases2 = []; Y_train = [];

    text_file = ['train/STS.input.MSRpar.txt', 'train/STS.input.MSRvid.txt', 'train/STS.input.SMTeuroparl.txt']
    text_file2 = ['train/STS.gs.MSRpar.txt', 'train/STS.gs.MSRvid.txt', 'train/STS.gs.SMTeuroparl.txt']


    for i in range(3):
        with open(text_file[i]) as f:
            for line in f:
                frases1.append(line.strip().split('\t')[0])
                frases2.append(line.strip().split('\t')[1])

        with open(text_file2[i]) as f:
            for line in f:
                Y_train.append(float(line.strip()))

    return frases1, frases2, np.array(Y_train)


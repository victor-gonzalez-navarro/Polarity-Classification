import numpy as np

def read_testing_datasets():
    frases1 = []; frases2 = []; Y_test = [];

    text_file = ['test/STS.input.MSRpar.txt', 'test/STS.input.MSRvid.txt', 'test/STS.input.SMTeuroparl.txt',
                 'test/STS.input.surprise.OnWN.txt', 'test/STS.input.surprise.SMTnews.txt']
    text_file2 = ['test/STS.gs.MSRpar.txt', 'test/STS.gs.MSRvid.txt', 'test/STS.gs.SMTeuroparl.txt',
                  'test/STS.gs.surprise.OnWN.txt', 'test/STS.gs.surprise.SMTnews.txt']

    for i in range(3):
        with open(text_file[i]) as f:
            for line in f:
                frases1.append(line.strip().split('\t')[0])
                frases2.append(line.strip().split('\t')[1])

        with open(text_file2[i]) as f:
            for line in f:
                Y_test.append(float(line.strip()))

    return frases1, frases2, np.array(Y_test)


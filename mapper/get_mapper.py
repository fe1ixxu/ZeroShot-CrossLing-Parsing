import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np
import random

def random_indice_generator(num, maxnum):
    total = list(range(maxnum))
    index = torch.tensor(random.sample(total, num))

    return index

def degree_anisotropy(vectors, index):
    vectors = torch.triu(vectors.t() @ vectors, 1)
    filter_st = torch.triu(-100*torch.ones(vectors.shape[1], vectors.shape[1])).t()
    vectors = torch.where(filter_st == -100, filter_st, vectors)
    ind = (vectors != -100).nonzero()
    vectors = vectors[ind[:, 0], ind[:, 1]]
    
    return torch.mean(vectors)


class Mapper():
    def __init__(self, input_file, ifiter):
        self.device = device
        self.ifiter = ifiter
        self.source = []
        self.target = []
        self.source_occur = []
        self.entropy = []
        self.source_vector = []
        self.target_vector = []
        self.info = None
        self.index = None
        self.occ_ratio = []
        self.s_train = None
        self.t_train = None

        self.source_index = 0
        self.target_index = 1
        self.occur_index = 2
        self.entropy_index = 3
        self.source_vector_index = 4
        self.target_vector_index = 5

        self.reading_file(input_file)

    def reading_file(self, input_file):
        temp = []
        tt = []
        with open(input_file, encoding="utf-8") as f:
            line = f.readline()
            while(line):
                line = line.split("\t")
                if int(line[self.occur_index]) > 10:
                    self.source.append(line[self.source_index])
                    self.target.append(line[self.target_index])
                    self.source_occur.append(int(line[self.occur_index]))
                    self.entropy.append(float(line[self.entropy_index]))
                    vector_en = np.array(line[self.source_vector_index].split()[:768], dtype=float)
                    self.source_vector.append(torch.tensor(vector_en, dtype=torch.float))
                    vector_de = np.array(line[self.target_vector_index].split()[:768], dtype=float)
                    self.target_vector.append(torch.tensor(vector_de, dtype=torch.float))

                    if line[self.source_index] not in temp:
                        temp.append(line[self.source_index])
                        if len(tt) != 0:
                            tt = [t/sum(tt) for t in tt]
                        self.occ_ratio += tt
                        tt = [float(line[self.occur_index])]
                    else:
                        tt.append(float(line[self.occur_index]))
                line = f.readline()

            if len(tt) != 0:
                tt = [t / sum(tt) for t in tt]
            self.occ_ratio += tt

        self.get_sort()

        self.source_vector = torch.stack(self.source_vector).to(self.device)
        self.source_vector /= torch.norm(self.source_vector, dim=1).view(-1,1)
        self.source_vector = self.source_vector.t()
        self.target_vector = torch.stack(self.target_vector).to(self.device)
        self.target_vector /= torch.norm(self.target_vector, dim=1).view(-1, 1)
        self.target_vector = self.target_vector.t()
        self.info = list(zip(self.source,self.target, range(len(self.source))))

        if self.ifiter:
            self.iter_norm()

    def simple_procrustes(self):
        source = self.source_vector
        target = self.target_vector
        U, _, V = torch.svd(target @ source.t())
        W= U @ V.t()
        aligned = W @ self.source_vector
        return W, aligned

    def get_sort(self):
        combine = list(zip(self.source, self.target, self.source_vector, self.target_vector, self.occ_ratio, self.entropy, self.source_occur))
        combine.sort(key = lambda x: x[-1], reverse=True)
        self.source = list(map(lambda x:x[0], combine))
        self.target = list(map(lambda x:x[1], combine))
        self.source_vector = list(map(lambda x:x[2], combine))
        self.target_vector = list(map(lambda x:x[3], combine))
        self.occ_ratio = list(map(lambda x:x[4], combine))
        self.entropy = list(map(lambda x:x[5], combine))
        self.source_occur = list(map(lambda x:x[-1], combine))

    def iter_norm(self, num=5):
        self.salient_anisotropy()
        for _ in range(num):
            self.source_vector -= self.source_vector.mean(dim=1).view(-1, 1)
            self.target_vector -= self.target_vector.mean(dim=1).view(-1, 1)
            self.source_vector /= torch.norm(self.source_vector, dim=0).view(1,-1)
            self.target_vector /= torch.norm(self.target_vector, dim=0).view(1, -1)
            self.salient_anisotropy()

    def salient_anisotropy(self):
            index = random_indice_generator(1000, self.source_vector.shape[1])
            an_s = degree_anisotropy(self.source_vector, index)
            an_t = degree_anisotropy(self.target_vector, index)
            print("anisotriopy for source lang is {}, for taget lang is {} ".format(float(an_s), float(an_t)))

import os
import re
import numpy as np
import argparse
import time
import hashlib
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from collections import Counter
from tqdm import tqdm
import fcntl

# from sklearn.datasets import make_blobs
def get_muti_mean_vector(vectors_en_all, vectors, labels, cluster_num):
    kmean = KMeans(n_clusters = cluster_num).fit(vectors_en_all)
    # centers = kmean.cluster_centers_
    cluster_en = []
    cluster_de = []
    cluster_label=[]
    cluster_entropy = []
    for i in range(cluster_num):
        cluster_temp = [labels[j] for j, l in enumerate(kmean.labels_) if l == i]
        count = Counter(cluster_temp)
        label_rank = count.most_common()
        cluster_entropy.append(entropy(list(map(lambda x: x[1], label_rank)), base=2))
        label = label_rank[0]
 
        vectors_temp = [v for j, v in enumerate(vectors) if kmean.labels_[j] == i]
        # vectors_temp = [v for j, v in enumerate(vectors) if v[0] == label[0] and kmean.labels_[j] == i]
        vectors_en = list(map(lambda x: x[1], vectors_temp))
        vectors_de = list(map(lambda x: x[2], vectors_temp))
        vectors_en = np.vstack(vectors_en).mean(axis=0)  # mean vectors of en
        vectors_de = np.vstack(vectors_de).mean(axis=0)  # mean vectors of de
        cluster_en.append(vectors_en)
        cluster_de.append(vectors_de)
        cluster_label.append(label)

    cluster_en = np.vstack(cluster_en)
    cluster_de = np.vstack(cluster_de)

    return cluster_en, cluster_de, cluster_label, cluster_entropy

def get_mean_vector(vectors, labels):
    count = Counter(labels)
    label_rank = count.most_common()
    cluster_entropy = entropy(list(map(lambda x: x[1], label_rank)), base=2)

    label = label_rank[0]
    # vectors = list(filter(lambda x: x[0]==label[0], vectors)) #### partial
    vectors_en = list(map(lambda x: x[1], vectors))
    vectors_de = list(map(lambda x: x[2], vectors))
    cluster_en = np.vstack(vectors_en).mean(axis=0)[np.newaxis,:]  # mean vectors of en
    cluster_de = np.vstack(vectors_de).mean(axis=0)[np.newaxis,:]  # mean vectors of de

    cluster_label = [label]
    cluster_entropy = [cluster_entropy]
    return cluster_en, cluster_de, cluster_label, cluster_entropy

def write_file(cluster_en, cluster_de, cluster_label, cluster_entropy, write_mode, file_name, write_file):
    with open(write_file , write_mode) as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for i in range(len(cluster_label)):
            f.writelines([file_name[:-4], '\t', cluster_label[i][0], '\t', str(cluster_label[i][1]), '\t', str(cluster_entropy[i]), '\t'])
            np.savetxt(f,cluster_en[i][np.newaxis,:], fmt='%.10f', newline="\t")
            np.savetxt(f,cluster_de[i][np.newaxis,:], fmt='%.10f', newline="\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def checkspecial(string):
    test_str = re.search(r"\W",string)
    if test_str==None:
        return False
    else:
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True) # file we are reading
    parser.add_argument("--write_file", type=str, required=True)
    parser.add_argument("--stopwords", default=None, type=str) # stopwords file name
    parser.add_argument("--min_threshold", type = int, default=10) # cluster the word whose appearence is larger than the threshold
    parser.add_argument("--min_num_words", type=int, default=3)    # discard the word whose appearence is less than the threshold
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    args = parser.parse_args()


    if args.stopwords:
        with open(args.stopwords, encoding = "utf-8") as f:
            stopwords = f.readlines()
            stopwords = [st.strip() for st in stopwords]
    else:
        stopwords = []

    rootdir = args.input_file  #'./word_vectors/4/'
    writefile = args.write_file
    list_dir = os.listdir(rootdir)

    if args.end == -1:
        args.end = len(list_dir)

    none_cluster = []
    discard_word = []
    for file_index, file_name in enumerate(tqdm(list_dir)):

        if not (file_index >= args.start and file_index< args.end):
            continue

        write_mode = "a"
        labels = []
        vectors = []

        if file_name[:-4] != '' and ".txt" in file_name:
            with open(rootdir+file_name) as f:
                # line = f.readline() #skip the first line which is the name of word
                line = f.readline()
                while(line):
                    line = line.split("\t")
                    labels.append(line[0])
                    vectors.append([line[0], np.array(line[1].split(" "), dtype = 'float'), np.array(line[2].split(" "), dtype = 'float')]) #[[label, vector_en, vector_de]]
                    line = f.readline()
                    
            if len(vectors) < args.min_num_words:
                discard_word.append(file_name[:-4])
                continue

            if len(vectors) <= args.min_threshold or checkspecial(file_name[:-4]) or file_name[:-4] in stopwords :
                cluster_en, cluster_de, cluster_label, cluster_entropy = get_mean_vector(vectors, labels)
            else:
                vectors_en_all = np.vstack(list(map(lambda x:x[1], vectors)))
                model = KElbowVisualizer(KMeans(), k=(1,8))
                model.fit(vectors_en_all)
                # print('Optimal k is: ', model.elbow_value_)
                if model.elbow_value_ == None:
                    none_cluster.append(file_name[:-4])
                    cluster_en, cluster_de, cluster_label, cluster_entropy = get_mean_vector(vectors, labels)
                else:
                    cluster_en, cluster_de, cluster_label, cluster_entropy = get_muti_mean_vector(vectors_en_all, vectors, labels, model.elbow_value_)
            write_file(cluster_en, cluster_de, cluster_label, cluster_entropy ,write_mode, file_name, writefile)

    print("Number of None in clustering:", len(none_cluster))
    print("Number of words that have been discarded:", len(discard_word))
    print("List of file that not been clusted:", none_cluster)
    print("List of words that have been discarded:", discard_word)

if __name__ == "__main__":
    main()


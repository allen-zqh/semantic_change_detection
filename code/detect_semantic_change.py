# -*- coding: utf-8 -*-

"""
This script detect words with meaning or usage change in two corpora
by 'clustering weighted concentration' (clustering weighted kappa) indicator.

----------------------------------------------------------------------
    
INPUT:  source_corpus, target_corpus
OUTPUT: intersected words in two corpora sorted by change score C(S,T)
    
----------------------------------------------------------------------
"""

import sys, codecs
import argparse
import numpy as np
from tqdm import tqdm
import os
import torch
import transformers
from transformers import AutoTokenizer
from transformers import BertModel
import json

import util
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import faiss

#os.environ["OMP_NUM_THREADS"] = "1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-source_corpus', default=r'source.txt')
    parser.add_argument('-target_corpus', default=r'target.txt')
    parser.add_argument('-m', '--bert_model', help='Language model to get word vectors', default='bert-large-cased')
    parser.add_argument('-c', '--cased', help='Use this to consider upper/lower case distinction', action='store_true')
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-f', '--freq_threshold', help='Words whose frequency is more than this value is considered', default=10, type=int)
    return parser.parse_args()


# getting contexual embeddings with word freqs
def get_contextual_embeddings_with_freqs(vectorizer, tokenizer, sentences, device='cpu'):
   
    # each vecctors are normalized
    token2vecs = util. tokenize_and_vectorize(vectorizer,
                                              tokenizer,
                                              sentences,
                                              device=device)
    token2freq = {}
    vector_size = None
    for token, vecs in token2vecs.items():
        token2freq[token] = len(vecs)
        sum_vec = np.sum(vecs, axis=0)
        vector_size = sum_vec.size

    return vector_size, token2freq, token2vecs

def cluster_vectors(vecs, max_clusters=5):
    vecs = np.array(vecs)
    n_samples = len(vecs)
    if n_samples < 2:
        gmm = GaussianMixture(n_components=1).fit(vecs)
        labels = gmm.predict(vecs)
        return gmm, labels
    
    best_n = 2
    best_score = -1
    best_labels = None
    best_gmm = None
    for n in range(2, min(max_clusters, n_samples) + 1):
        gmm = GaussianMixture(n_components=n)
        gmm.fit(vecs)
        labels = gmm.predict(vecs)
        if len(set(labels)) < 2 or len(set(labels)) >= n_samples:
            continue
        score = silhouette_score(vecs, labels)
        if score > best_score:
            best_n = n
            best_score = score
            best_gmm = gmm
            best_labels = labels

    if best_score == -1 or best_gmm is None:
        gmm = GaussianMixture(n_components=1).fit(vecs)
        labels = gmm.predict(vecs)
        return gmm, labels

    return best_gmm, best_labels

def estimate_kappa_mixture(gmm, vecs, labels, vector_size=1024):
    vecs = np.array(vecs)
    if gmm is None or gmm.n_components == 1:
        sum_vec = np.sum(vecs, axis=0)
        norm = np.linalg.norm(sum_vec) / float(len(vecs))
        d = vector_size
        kappa = (norm * (d - norm**2)) / (1 - norm**2)
        return kappa

    kappa = 0
    d = vector_size
    
    for i in range(gmm.n_components):
        weight = gmm.weights_[i]
        cluster_indices = np.where(labels == i)[0]
        cluster_size = len(cluster_indices)
        
        if cluster_size > 0:
            sum_vec = np.sum(vecs[cluster_indices], axis=0)
            norm = np.linalg.norm(sum_vec) / cluster_size
            kappa += weight * (norm * (d - norm**2)) / (1 - norm**2)
    
    return kappa

# using FAISS for accelerating clustering process for high dimensions
def faiss_kmeans(vecs, n_clusters=5, n_iter=20, verbose=False):
    d = vecs.shape[1]
    kmeans = faiss.Kmeans(d, n_clusters, niter=n_iter, verbose=verbose)
    kmeans.train(vecs)
    labels = kmeans.index.search(vecs, 1)[1].flatten()
    return kmeans, labels

def valid_clusters(labels, min_samples_per_cluster=2):
    unique, counts = np.unique(labels, return_counts=True)
    return all(count >= min_samples_per_cluster for count in counts)

def optimal_faiss_kmeans(vecs, max_clusters=10, n_iter=20, verbose=False):
    vecs = np.array(vecs)
    best_n = 2
    best_score = -1
    best_labels = None
    best_kmeans = None
    
    for n in range(2, max_clusters + 1):
        kmeans, labels = faiss_kmeans(vecs, n_clusters=n, n_iter=n_iter, verbose=verbose)
        if not valid_clusters(labels):
            continue
        score = silhouette_score(vecs, labels)
        if score > best_score:
            best_n = n
            best_score = score
            best_kmeans = kmeans
            best_labels = labels
    
    if best_score == -1 or best_kmeans is None:
        best_kmeans, best_labels = faiss_kmeans(vecs, n_clusters=1, n_iter=n_iter, verbose=verbose)
    
    return best_kmeans, best_labels

def estimate_kappa_faiss(kmeans, vecs, labels, vector_size=1024):
    vecs = np.array(vecs)
    if kmeans is None or kmeans.k == 1:
        sum_vec = np.sum(vecs, axis=0)
        norm = np.linalg.norm(sum_vec) / float(len(vecs))
        d = vector_size
        kappa = (norm * (d - norm**2)) / (1 - norm**2)
        return kappa

    kappa = 0
    d = vector_size
    
    for i in range(kmeans.k):
        weight = np.sum(labels == i) / len(labels)
        cluster_indices = np.where(labels == i)[0]
        cluster_size = len(cluster_indices)
        
        if cluster_size > 0:
            sum_vec = np.sum(vecs[cluster_indices], axis=0)
            norm = np.linalg.norm(sum_vec) / cluster_size
            kappa += weight * (norm * (d - norm**2)) / (1 - norm**2)
    
    return kappa

def cal_change_score(source_token2freq, 
              target_token2freq,
              source_token2vecs,
              target_token2vecs,
              source_vector_size=1024,
              target_vector_size=1024,
              freq_threshold=10):

    # Obtaining common vocabulary set 
    target_vocab = set(target_token2freq.keys())
    source_vocab = set(source_token2freq.keys())
    common_vocab = target_vocab & source_vocab

    # calculating change score
    token2score = {}
    for token in tqdm(common_vocab):
        source_freq = source_token2freq[token]
        target_freq = target_token2freq[token]
        source_vecs = source_token2vecs[token]
        target_vecs = target_token2vecs[token]
        
        # setting a proper threshold ensuring more stable esimated kappa
        # see more details in paper Section 4.3 'relationship between freq and norm of weighted mean vecs'
        if target_freq <= freq_threshold or source_freq <= freq_threshold:
            continue
        
        # getting best clustering number by optimazing silhouette
        source_gmm, source_labels = cluster_vectors(np.array(source_vecs))
        target_gmm, target_labels = cluster_vectors(np.array(target_vecs))
        
        # calculating clustering weighted concentration
        source_kappa = estimate_kappa_mixture(source_gmm, source_vecs, source_labels, source_vector_size)
        target_kappa = estimate_kappa_mixture(target_gmm, target_vecs, target_labels, target_vector_size)

        if target_kappa > 0.0 and source_kappa > 0.0:
            score = np.log(target_kappa / source_kappa)
            token2score[token] = score

    return token2score


def output(token2score, source_token2freq, target_token2freq, output_dir,
           delim='\t', digit=3):

    results = sorted(token2score.items(), key = lambda x: x[1], reverse=True)

    with open(output_dir, 'w', encoding='utf-8') as f:
        for token, score in tqdm(results):
            score = round(score, digit)
            output = delim.join((token,
                                 str(score),
                                 str(source_token2freq[token]),
                                 str(target_token2freq[token])))
            f.write(output + '\n')


def main():

    args = parse_args()

    # Preparing data
    source_sentences = util.load_sentences(args.source_corpus, args.cased)
    target_sentences = util.load_sentences(args.target_corpus, args.cased)
    batched_source_sentences = util.to_batches(source_sentences, batch_size=args.batch_size)
    batched_target_sentences = util.to_batches(target_sentences, batch_size=args.batch_size)


    # Preparing BERT model and tokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vectorizer = BertModel.from_pretrained(args.bert_model)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # getting contextual embeddings with token frequencies
    source_vector_size, source_token2freq, source_token2vecs = \
        get_contextual_embeddings_with_freqs(vectorizer, tokenizer, batched_source_sentences, device=device)
    target_vector_size, target_token2freq, target_token2vecs = \
        get_contextual_embeddings_with_freqs(vectorizer, tokenizer, batched_target_sentences, device=device)

    # calculating change score
    token2score = cal_change_score(source_token2freq, target_token2freq, source_token2vecs, target_token2vecs)

    output(token2score, source_token2freq, target_token2freq, output_dir=r'results.txt')

if __name__ == '__main__':
    main()
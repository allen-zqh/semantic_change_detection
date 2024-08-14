# -*- coding: utf-8 -*-

"""
This script aims at finding typical instances for detecteds word having wider meanings
in the source.txt than in the target.txt.

----------------------------------------------------------------------

INPUT:  source_corpus, target_corpus, target_word
OUTPUT: typical instances in source_corpus sorted by cosimilarity with
        difference of weighted mean vectors.

----------------------------------------------------------------------
"""

import sys, codecs
import random
import argparse
import numpy as np

import torch
from transformers import BertModel
from transformers import AutoTokenizer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import util
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    
    # C(S,T)>0 means word 'aaa' has wider meanings in source.txt
    parser.add_argument('-source_corpus', default=r'source.txt')
    parser.add_argument('-target_corpus', default=r'target.txt')

    parser.add_argument('-t',
                        '--target_phrase',
                        default='aaa',
                        help='Target phrase whose instances are scored')

    parser.add_argument('-m',
                        '--bert_model',
                        help='Language model to get word vectors',
                        default='bert-large-cased')

    parser.add_argument('-c',
                        '--cased',
                        help='Use this to consider upper/lower case distinction',
                        action='store_true')

    parser.add_argument('-b', '--batch_size', default=32, type=int)

    parser.add_argument('-n',
                        '--topN',
                        help='To show top N results',
                        default=10,
                        type=int)

    parser.add_argument('-a',
                        '--all_subwords',
                        help='To use all subwords in a token for\
                              its word vector; otherwise only first subword',
                        action='store_true')
        
    args = parser.parse_args()

    return args

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

def cal_similarity(source_vecs, target_vecs):

    # normalizing (or ensuring) to norm 1
    source_vecs = [v/np.linalg.norm(v) for v in source_vecs]
    target_vecs = [v/np.linalg.norm(v) for v in target_vecs]
    
    # clustering
    source_gmm, source_labels = cluster_vectors(np.array(source_vecs))
    target_gmm, target_labels = cluster_vectors(np.array(target_vecs))
    
    def get_weighted_mean_vector(vecs, labels, n_components):
        mean_vecs = []
        weights = []
        
        for i in range(n_components):
            cluster_vecs = vecs[labels == i]
            mean_vec = np.mean(cluster_vecs, axis=0)
            mean_vecs.append(mean_vec)
            weights.append(len(cluster_vecs))
            
        # Calculate weighted mean vector
        weighted_mean_vec = np.average(mean_vecs, axis=0, weights=weights)
        return weighted_mean_vec
    
    if source_gmm is None or source_gmm.n_components == 1:
        source_mean_vec = np.mean(source_vecs, axis=0)
    else:
        source_mean_vec = get_weighted_mean_vector(np.array(source_vecs), source_labels, source_gmm.n_components)
    
    if target_gmm is None or target_gmm.n_components == 1:
        target_mean_vec = np.mean(target_vecs, axis=0)
    else:
        target_mean_vec = get_weighted_mean_vector(np.array(target_vecs), target_labels, target_gmm.n_components)
    
    # Normalize the mean vectors
    source_mean_vec = source_mean_vec / np.linalg.norm(source_mean_vec)
    target_mean_vec = target_mean_vec / np.linalg.norm(target_mean_vec)
    
    # difference of two weighted mean vectors
    diff_vec = source_mean_vec - target_mean_vec

    similarities = []
    for i, v in enumerate(source_vecs):
        similarity = np.dot(v, diff_vec)
        similarities.append(similarity)

    return similarities


def output(results, topN=10, marker='*', delim='\t'):

    index = min(topN, len(results))

    for similarity, words, span in results[:index]:
        start, end = span
        words = ['[BOS]'] + words + ['[EOS]']
        before = ' '.join(words[:start+1])
        target = marker + ' '.join(words[start+1:end+1]) + marker
        after = ' '.join(words[end+1:])
        output = '\t'.join((str(similarity), before, target, after))
        print(output)
        

def main():
    
    args = parse_args()

    # Preparing data 
    # and finding sentences with the target word
    source_sentences, source_spans =\
        util.load_sentences_with_target_spans(args.source_corpus,
                                              args.target_phrase,
                                              cased=args.cased)
    target_sentences, target_spans =\
        util.load_sentences_with_target_spans(args.target_corpus,
                                              args.target_phrase,
                                              cased=args.cased)

    if len(source_spans)<1 or len(target_spans)<1:
        exit(0)


    batched_source_sentences = util.to_batches(source_sentences, batch_size=args.batch_size)
    batched_source_spans = util.to_batches(source_spans, batch_size=args.batch_size)
    batched_target_sentences = util.to_batches(target_sentences, batch_size=args.batch_size)
    batched_target_spans = util.to_batches(target_spans, batch_size=args.batch_size)


    # Preparing BERT model and tokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vectorizer = BertModel.from_pretrained(args.bert_model)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Obtaining contexual embeddings for target phrase
    source_vecs =\
        util.tokenize_and_vectorize_with_spans(batched_source_sentences,
                                               batched_source_spans,
                                               vectorizer,
                                               tokenizer,
                                               all_subwords=args.all_subwords,
                                               device=device)
    target_vecs =\
        util.tokenize_and_vectorize_with_spans(batched_target_sentences,
                                               batched_target_spans,
                                               vectorizer,
                                               tokenizer,
                                               all_subwords=args.all_subwords,
                                               device=device)

    # Calculating similarities
    scores = cal_similarity(source_vecs, target_vecs)

    # For output
    results =\
        [(s, source_sentences[i], source_spans[i]) for i, s in enumerate(scores)]
    results = sorted(results, key = lambda x: x[0], reverse=True)

    output(results, topN=args.topN)
    

if __name__ == '__main__':
    main()
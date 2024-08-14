# -*- coding: utf-8 -*-

"""
This script aims at giving an example of visualization for detected word
'signature' with different sematic sense clusters in source&target copora.

----------------------------------------------------------------------

INPUT:  contexual embeddings, target_word ('signature' in this case)
OUTPUT: figure of semantic evolution

----------------------------------------------------------------------
"""

#%% step1 - getting mean vectors for all words
import numpy as np

def compute_mean_vectors(token2vecs):
    mean_vectors = {}
    for word, vecs in token2vecs.items():
        mean_vectors[word] = np.mean(vecs, axis=0)
    return mean_vectors

source_meanvecs = compute_mean_vectors(source_token2vecs)
target_meanvecs = compute_mean_vectors(target_token2vecs)

#%% step2 - getting sense clusters
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def optimize_gmm_clusters(vectors, max_clusters=5):
    best_n_clusters = 2
    best_silhouette = -1
    best_labels = None
    vectors_np = np.array(vectors)
    for n_clusters in range(2, max_clusters + 1):
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
        gmm.fit(vectors_np)
        labels = gmm.predict(vectors_np)
        silhouette_avg = silhouette_score(vectors_np, labels)
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_n_clusters = n_clusters
            best_labels = labels
    return best_labels, best_n_clusters

# 'signature' only has 1 sense cluster in source corpus thus we omit the process
signature_vecs = target_token2vecs['signature']
labels, num_clusters = optimize_gmm_clusters(signature_vecs)

#%% step3 - finding most similar words for source
from sklearn.metrics.pairwise import cosine_similarity

def find_top_similar_vectors(vector, meanvecs, top_n=5):
    all_words = list(meanvecs.keys())
    all_vecs = np.array([meanvecs[word] for word in all_words])
    similarities = cosine_similarity([vector], all_vecs)[0]
    top_indices = np.argsort(similarities)[-top_n-1:-1][::-1]  # Exclude itself
    return [all_words[i] for i in top_indices if all_words[i] != 'signature']  # Exclude 'signature'

signature_vector = source_meanvecs['signature']
top_similar_words = find_top_similar_vectors(signature_vector, source_meanvecs)

#%% step4 - calculating mean vecs and finding most similar words for each cluster in target
def compute_cluster_means(vectors, labels, num_clusters):
    cluster_means = []
    for i in range(num_clusters):
        cluster_vecs = [vectors[j] for j in range(len(vectors)) if labels[j] == i]
        cluster_means.append(np.mean(cluster_vecs, axis=0))
    return cluster_means

signature_target_cluster_meanvecs = compute_cluster_means(signature_vecs, labels, num_clusters)

def find_top_similar_vectors_excluding_signature(vector, meanvecs, top_n=5):
    all_words = list(meanvecs.keys())
    all_vecs = np.array([meanvecs[word] for word in all_words])
    similarities = cosine_similarity([vector], all_vecs)[0]
    top_indices = np.argsort(similarities)[::-1]
    similar_words = []
    for idx in top_indices:
        if len(similar_words) >= top_n:
            break
        if all_words[idx] != 'signature':  # Exclude 'signature'
            similar_words.append(all_words[idx])
    return similar_words

cluster_similar_words = []
for cluster_mean in signature_target_cluster_meanvecs:
    similar_words = find_top_similar_vectors_excluding_signature(cluster_mean, target_meanvecs)
    cluster_similar_words.append(similar_words)

#%% step5 - plot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as patches

def plot_2d_rotated_planes(source_words, target_words, source_meanvecs, target_meanvecs, cluster_means, cluster_similar_words):
    source_vectors = np.array([source_meanvecs[word] for word in source_words])
    target_vectors = np.array([target_meanvecs[word] for word in target_words])
    all_vectors = np.concatenate([source_vectors, target_vectors])

    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(all_vectors)

    source_reduced = reduced_vectors[:len(source_vectors)]
    target_reduced = reduced_vectors[len(source_vectors):]

    cluster_means_reduced = pca.transform(cluster_means)

    # clustering weighted mean vec
    cluster_weights = [len(cluster) for cluster in cluster_similar_words]
    total_weight = sum(cluster_weights)
    weighted_mean_vec = np.average(cluster_means_reduced, axis=0, weights=cluster_weights)

    # rotating angle
    angle = 15
    angle_rad = np.radians(angle)

    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    # rotate source plane
    source_reduced_rotated = np.dot(source_reduced, rotation_matrix.T)

    # rotate target plane
    distance = 1  # distance between two planes
    target_reduced_rotated = np.dot(target_reduced, rotation_matrix.T) + np.array([distance, 0])
    cluster_means_reduced_rotated = np.dot(cluster_means_reduced, rotation_matrix.T) + np.array([distance, 0])
    weighted_mean_vec_rotated = np.dot(weighted_mean_vec, rotation_matrix.T) + np.array([distance, 0])

    fig, ax = plt.subplots(figsize=(12, 8))

    # setting line&color for source plane
    ax.scatter(source_reduced_rotated[:, 0], source_reduced_rotated[:, 1], c='black', marker='o', s=30, label='Neighbor Word')
    ax.scatter(source_reduced_rotated[0, 0], source_reduced_rotated[0, 1], color='black', marker='s', s=30, label='Center of Sense')
    for vec in source_reduced_rotated:
        ax.plot([vec[0], source_reduced_rotated[0, 0]], [vec[1], source_reduced_rotated[0, 1]], 'black', linewidth=0.8)
    for i, word in enumerate(source_words):
        if word != 'signature':
            ax.text(source_reduced_rotated[i, 0], source_reduced_rotated[i, 1], word, fontsize=12, ha='right')

    # setting line&color for target plane
    cluster_colors = ['black', 'lightblue']

    for i, (cluster_mean, cluster_points) in enumerate(zip(cluster_means_reduced_rotated, cluster_similar_words)):
        cluster_indices = [target_words.index(word) for word in cluster_points]
        cluster_vectors = target_reduced_rotated[cluster_indices]

        ax.scatter(cluster_vectors[:, 0], cluster_vectors[:, 1], c=cluster_colors[i], marker='o', s=30, label=f'Cluster {i} Points')
        ax.scatter(cluster_mean[0], cluster_mean[1], c=cluster_colors[i], marker='s', s=30, label=f'Cluster {i} Mean')
        for vec in cluster_vectors:
            ax.plot([vec[0], cluster_mean[0]], [vec[1], cluster_mean[1]], color=cluster_colors[i], linewidth=0.8)
        for j, word in enumerate(cluster_points):
            offset = 0
            if cluster_colors[i] == 'lightblue':
                ax.text(cluster_vectors[j, 0] + offset, cluster_vectors[j, 1] + offset, word, fontsize=12, ha='right')
            else:
                ax.text(cluster_vectors[j, 0], cluster_vectors[j, 1], word, fontsize=12, ha='right')

    # weighted center for target
    ax.scatter(weighted_mean_vec_rotated[0], weighted_mean_vec_rotated[1], color='black', marker='s', s=30, label='Signature (Target)')

    # line for center to center (S->T)
    ax.plot([source_reduced_rotated[0, 0], weighted_mean_vec_rotated[0]], [source_reduced_rotated[0, 1], weighted_mean_vec_rotated[1]], 'black', linewidth=0.8)

    # lines in target
    for i, cluster_mean in enumerate(cluster_means_reduced_rotated):
        ax.plot([weighted_mean_vec_rotated[0], cluster_mean[0]], [weighted_mean_vec_rotated[1], cluster_mean[1]], 'black', linewidth=0.8)
        cluster_indices = [target_words.index(word) for word in cluster_similar_words[i]]
        cluster_vectors = target_reduced_rotated[cluster_indices]
        for vec in cluster_vectors:
            ax.plot([vec[0], cluster_mean[0]], [vec[1], cluster_mean[1]], color=cluster_colors[i], linewidth=0.8)

    # rotated planes
    def draw_rotated_rectangle(ax, center, width, height, angle, **kwargs):
        theta = np.radians(angle)
        rect = patches.Rectangle((-width/2, -height/2), width, height, **kwargs)
        t = plt.matplotlib.transforms.Affine2D().rotate(theta) + plt.matplotlib.transforms.Affine2D().translate(*center)
        rect.set_transform(t + ax.transData)
        ax.add_patch(rect)

    # original planes
    max_width = max(max(source_reduced[:, 0]) - min(source_reduced[:, 0]), max(target_reduced[:, 0]) - min(target_reduced[:, 0])) * 1.5
    max_height = max(max(source_reduced[:, 1]) - min(source_reduced[:, 1]), max(target_reduced[:, 1]) - min(target_reduced[:, 1])) * 1.5

    source_center = [np.mean(source_reduced_rotated[:, 0]), np.mean(source_reduced_rotated[:, 1])]
    draw_rotated_rectangle(ax, source_center, max_width, max_height, angle, edgecolor='none', facecolor='gray', alpha=0.05)

    target_center = [np.mean(target_reduced_rotated[:, 0]), np.mean(target_reduced_rotated[:, 1])]
    draw_rotated_rectangle(ax, target_center, max_width, max_height, angle, edgecolor='none', facecolor='gray', alpha=0.05)
    
    ax.axis('off')

    # legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=2, frameon=False)
    plt.setp(legend.get_texts(), fontsize='12')

    plt.show()

source_words = ['signature'] + top_similar_words
target_words = []
for cluster_words in cluster_similar_words:
    target_words.extend(cluster_words)

plot_2d_rotated_planes(source_words, target_words, source_meanvecs, target_meanvecs, signature_target_cluster_meanvecs, cluster_similar_words)
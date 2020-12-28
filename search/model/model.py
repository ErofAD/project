"""
This example uses Approximate Nearest Neighbor Search (ANN) with Annoy (https://github.com/spotify/annoy).
Searching a large corpus with Millions of embeddings can be time-consuming. To speed this up,
ANN can index the existent vectors. For a new query vector, this index can be used to find the nearest neighbors.
This nearest neighbor search is not perfect, i.e., it might not perfectly find all top-k nearest neighbors.
In this example, we use Annoy. It learns to a tree that partitions embeddings into smaller sections. For our query embeddings,
we can efficiently check which section matches and only search that section for nearest neighbor.
Selecting the n_trees parameter is quite important. With more trees, we get a better recall, but a worse run-time.
This script will compare the result from ANN with exact nearest neighbor search and output a Recall@k value
as well as the missing results in the top-k hits list.
See the Annoy repository, how to install Annoy.
For details how Annoy works, see: https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html
As dataset, we use the Quora Duplicate Questions dataset, which contains about 500k questions:
https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs
As embeddings model, we use the SBERT model 'distilbert-multilingual-nli-stsb-quora-ranking',
that it aligned for 100 languages. I.e., you can type in a question in various languages and it will
return the closest questions in the corpus (questions in the corpus are mainly in English).
"""
from sentence_transformers import SentenceTransformer, util
import os
import csv
import pickle
import time
import torch
from annoy import AnnoyIndex



model_name = 'distilbert-multilingual-nli-stsb-quora-ranking'
model = SentenceTransformer(model_name)

url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
dataset_path = "quora_duplicate_questions.tsv"
max_corpus_size = 100000

n_trees = 256           #Number of trees used for Annoy. More trees => better recall, worse run-time
embedding_size = 768    #Size of embeddings
top_k_hits = 10         #Output k hits

annoy_index_path = 'quora-embeddings-{}-size-{}-annoy_index-trees-{}.ann'.format(model_name.replace('/', '_'), max_corpus_size,n_trees)
embedding_cache_path = 'quora-embeddings-{}-size-{}.pkl'.format(model_name.replace('/', '_'), max_corpus_size)


#Check if embedding cache path exists
if not os.path.exists(embedding_cache_path):
    # Check if the dataset exists. If not, download and extract
    # Download dataset if needed
    if not os.path.exists(dataset_path):
        print("Download dataset")
        util.http_get(url, dataset_path)

    # Get all unique sentences from the file
    corpus_sentences = set()
    with open(dataset_path, encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            corpus_sentences.add(row['question1'])
            if len(corpus_sentences) >= max_corpus_size:
                break

            corpus_sentences.add(row['question2'])
            if len(corpus_sentences) >= max_corpus_size:
                break

    corpus_sentences = list(corpus_sentences)
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

    print("Store file on disc")
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
else:
    print("Load pre-computed embeddings from disc")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['sentences']
        corpus_embeddings = cache_data['embeddings']

if not os.path.exists(annoy_index_path):
    # Create Annoy Index
    print("Create Annoy index with {} trees. This can take some time.".format(n_trees))
    annoy_index = AnnoyIndex(embedding_size, 'angular')

    for i in range(len(corpus_embeddings)):
        annoy_index.add_item(i, corpus_embeddings[i])

    annoy_index.build(n_trees)
    annoy_index.save(annoy_index_path)
else:
    #Load Annoy Index from disc
    annoy_index = AnnoyIndex(embedding_size, 'angular')
    annoy_index.load(annoy_index_path)


corpus_embeddings = torch.from_numpy(corpus_embeddings)

######### Search in the index ###########

print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))


class Search:
    def searching(inp_question):
        start_time = time.time()
        question_embedding = model.encode(inp_question)

        corpus_ids, scores = annoy_index.get_nns_by_vector(question_embedding, top_k_hits, include_distances=True)
        hits = []
        for id, score in zip(corpus_ids, scores):
            hits.append({'corpus_id': id, 'score': 1-((score**2) / 2)})

        end_time = time.time()

        print("Input question:", inp_question)
        print("Results (after {:.3f} seconds):".format(end_time-start_time))
        return_results = []
        for hit in hits[0:top_k_hits]:
            return_results.append(corpus_sentences[hit['corpus_id']])

        return return_results

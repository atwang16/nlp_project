import numpy as np
import torch
import pdb
import gzip

# maps word to embedding tensor
def create_embedding_dict(filename, glove=False):
	word_to_embedding = {}
	if glove:
		f = open(filename, 'r')
	else:
		f = gzip.open(filename, 'r')
	for l in f.readlines():
		line = l.split()
		word = line[0]
		embedding = torch.Tensor([float(v) for v in line[1:]])
		word_to_embedding[word] = embedding
	return word_to_embedding


# maps question id to:
# 1) title embedding tensors (padded to 25)
# 2) mask for padding titles
# 3) body embedding tensors (padded to 75)
# 4) mask for padding bodies
def create_question_dict(filename, embedding, hidden_size):
	title_embeddings_dict = {}
	title_mask_dict = {}
	body_embeddings_dict = {}
	body_mask_dict = {}

	f = gzip.open(filename, 'r')
	for l in f.readlines():
		line = l.split('\t')
		id, title, body = line

		title_words = torch.zeros(25, len(embedding.itervalues().next()))
		title_mask = torch.zeros(25, hidden_size*2)
		body_words = torch.zeros(75, len(embedding.itervalues().next()))
		body_mask = torch.zeros(75, hidden_size*2)

		i = 0
		for w in title.split():
			if i < 25 and w.lower() in embedding:
				title_words[i] = embedding[w.lower()]
				title_mask[i] = torch.ones(hidden_size*2)
			i += 1

		i = 0
		for w in body.split():
			if i < 75 and w.lower() in embedding:
				body_words[i] = embedding[w.lower()]
				body_mask[i] = torch.ones(hidden_size*2)
			i += 1

		title_embeddings_dict[int(id)] = title_words
		title_mask_dict[int(id)] = title_mask
		body_embeddings_dict[int(id)] = body_words
		body_mask_dict[int(id)] = body_mask

	return (title_embeddings_dict, title_mask_dict, body_embeddings_dict, body_mask_dict)


# creates training samples, each with 1 query, 1 positive, 20 negative question ids
def read_training_data(filename):
	train_matrix = []
	f = open(filename, 'r')
	for l in f.readlines():
		line = l.split('\t')
		query, positives, negatives = line

		for pos in positives.split():
			twenty_neg = np.take([int(id) for id in negatives.split()], np.random.choice(100, 20, replace=False))
			ids = np.concatenate(([int(query), int(pos)], twenty_neg))

			sample_id_tensor = torch.Tensor(ids)
			train_matrix.append(sample_id_tensor)

	train_id_tensor = torch.functional.stack(train_matrix)
	return train_id_tensor


# reads eval input data and creates:
# 1) eval matrix, each row with 1 query, 20 candidate questions
# 2) label dict, maps query id to candidate labels (1 for positive, 0 for negative)
# 3) score dict, maps query id to BM25 scores
def read_eval_data(filename):
	eval_matrix = []
	label_dict = {}
	score_dict = {}
	f = open(filename, 'r')
	for l in f.readlines():
		line = l.split('\t')
		query, positives, candidates, scores = line

		golden_set = set([int(v) for v in positives.split()])
		candidates = [int(v) for v in candidates.split()]
		sample = torch.Tensor(np.concatenate(([int(query)], candidates)))

		label_dict[int(query)] = torch.Tensor([1 if v in golden_set else 0 for v in candidates])
		score_dict[int(query)] = torch.Tensor([float(v) for v in scores.split()])
		eval_matrix.append(sample)

	eval_tensor = torch.functional.stack(eval_matrix)
	return eval_tensor, label_dict, score_dict


# computes MAP metric
def MAP(data):
	scores = []
	missing_MAP = 0
	for item in data:
		temp = []
		count = 0.0
		for i,val in enumerate(item):
			if val == 1:
				count += 1.0
				temp.append(count/(i+1))
		if len(temp) > 0:
			scores.append(sum(temp) / len(temp))
		else:
			missing_MAP += 1
	return sum(scores)/len(scores) if len(scores) > 0 else 0.0


# computes MRR metric
def MRR(data):
	scores = []
	for item in data:
		for i,val in enumerate(item):
			if val == 1:
				scores.append(1.0/(i+1))
				break
	return sum(scores)/len(scores) if len(scores) > 0 else 0.0


# computes precision metric
def precision(data, precision_at):
	scores = []
	for item in data:
		temp = item[:precision_at]
		if any(val==1 for val in item):
			scores.append(sum([1 if val==1 else 0 for val in temp])*1.0 / len(temp) if len(temp) > 0 else 0.0)
	return sum(scores)/len(scores) if len(scores) > 0 else 0.0


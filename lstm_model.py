import numpy as np
import math, sys
import torch.nn as nn
import torch
import time
from torch.autograd import Variable
import torch.optim as optim
import pdb
from lstm_model_utils import *
from optparse import OptionParser

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, batch_size):
        super(RNN, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=n_layers, batch_first=False, dropout=0.1, bidirectional=True)
        return

    def forward(self, input_tensor, hidden, state):
        output, hc_n = self.rnn(input_tensor, (hidden, state))
        h_n, c_n = hc_n[0], hc_n[1]
        return output


def mean_pooling(rnn, hidden, state, question_embedding, question_mask, sample):
	input_minibatch = torch.functional.stack([question_embedding[s] for s in sample]).permute(1,0,2)
	mask_minibatch = torch.functional.stack([question_mask[s] for s in sample]).permute(1,0,2)
	output_matrix = rnn(Variable(input_minibatch, requires_grad=True), hidden, state).data*mask_minibatch
	sum_matrix = torch.sum(output_matrix, 0)
	num_words_matrix = torch.sum(mask_minibatch, 0)
	return sum_matrix/num_words_matrix


def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def train(rnn, criterion, optimizer, train_data, question_data, hidden_size, num_layers, batch_size):
	hidden = Variable(torch.zeros(num_layers*2, batch_size, hidden_size))
	state = Variable(torch.zeros(num_layers*2, batch_size, hidden_size))
	rnn.zero_grad()

	train_data = train_data[:300] # ONLY FOR DEBUGGING, REMOVE LINE TO RUN ON ALL TRAINING DATA

	TITLE_EMB, TITLE_MASK, BODY_EMB, BODY_MASK = question_data
	num_samples = train_data.size(0)
	outer_batch_size = 25
		
	for i in range(num_samples / outer_batch_size):
		input_batch = train_data[i*outer_batch_size:(i+1)*outer_batch_size]
		# input_batch = train_data[:outer_batch_size] # for testing if weights or loss change
		X_scores = []

		for sample in input_batch:
			encoded_title_matrix = mean_pooling(rnn, hidden, state, TITLE_EMB, TITLE_MASK, sample)
			encoded_body_matrix = mean_pooling(rnn, hidden, state, BODY_EMB, BODY_MASK, sample)
			encoded_matrix = (encoded_title_matrix + encoded_body_matrix) / 2.0

			query_matrix = encoded_matrix[0].expand(21,hidden_size*2)
			candidate_matrix = encoded_matrix[1:]
			cos = nn.CosineSimilarity(dim=1)(query_matrix, candidate_matrix)
			X_scores.append(cos)

		X_scores = Variable(torch.functional.stack(X_scores), requires_grad=True)
		targets = Variable(torch.zeros(X_scores.size(0)).type(torch.LongTensor))

		# print(X_scores, targets)
		# a = list(rnn.parameters())[0].clone()

		optimizer.zero_grad()
		loss = criterion(X_scores, targets)
		loss.backward()
		optimizer.step()

		# b = list(rnn.parameters())[0].clone()
		# print(torch.equal(a.data, b.data))

		# for p in rnn.parameters():
		# 	print(p)
		# 	print(p.data)
		# 	print(p.grad.data)

		# print(loss.data[0])

	return loss.data[0]


def compute_metrics(data):
	return (MAP(data), MRR(data), precision(data, 1), precision(data, 5))


def evaluate(rnn, eval_data, label_dict, question_data, hidden_size, num_layers, batch_size):
	hidden = Variable(torch.zeros(num_layers*2, batch_size, hidden_size))
	state = Variable(torch.zeros(num_layers*2, batch_size, hidden_size))

	TITLE_EMB, TITLE_MASK, BODY_EMB, BODY_MASK = question_data
	sorted_output_data = []

	for sample in eval_data:
		encoded_title_matrix = mean_pooling(rnn, hidden, state, TITLE_EMB, TITLE_MASK, sample)
		encoded_body_matrix = mean_pooling(rnn, hidden, state, BODY_EMB, BODY_MASK, sample)
		encoded_matrix = (encoded_title_matrix + encoded_body_matrix) / 2.0

		query_matrix = encoded_matrix[0].expand(20,hidden_size*2)
		candidate_matrix = encoded_matrix[1:]
		cos = nn.CosineSimilarity(dim=1)(query_matrix, candidate_matrix)
		golden_tags = label_dict[sample[0]]

		sorted_output = [golden_tags for _,golden_tags in sorted(zip(cos,golden_tags),reverse=True)]
		sorted_output_data.append(sorted_output)
	metrics = compute_metrics(sorted_output_data)
	return metrics


if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option("--batch_size", dest="batch_size", default="22")
	parser.add_option("--hidden_size", dest="hidden_size", default="125")
	parser.add_option("--num_epochs", dest="num_epochs", default="5")
	parser.add_option("--learning_rate", dest="learning_rate", default="1e-4")
	parser.add_option("--print_epochs", dest="print_epochs", default="1")
	opts,args = parser.parse_args()

	batch_size = int(opts.batch_size)
	hidden_size = int(opts.hidden_size)
	n_layers = 1
	n_features = 200
	learning_rate = float(opts.learning_rate)
	n_epochs = int(opts.num_epochs)
	print_every = int(opts.print_epochs)

	EMBEDDING = create_embedding_dict("askubuntu-master/vector/vectors_pruned.200.txt.gz")
	QUESTIONS = create_question_dict("askubuntu-master/text_tokenized.txt.gz", EMBEDDING, hidden_size)
	TRAIN_DATA = read_training_data("askubuntu-master/train_random.txt")
	DEV_DATA, DEV_LABEL_DICT, DEV_SCORES = read_eval_data("askubuntu-master/dev.txt")
	TEST_DATA, TEST_LABEL_DICT, TEST_SCORES = read_eval_data("askubuntu-master/test.txt")

	rnn = RNN(n_features, hidden_size, n_layers, batch_size)
	optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
	criterion = nn.MultiMarginLoss(margin=0.2)
	print("Starting run with batch_size: %d, hidden size: %d, learning rate: %.4f"%(batch_size, hidden_size, learning_rate))
	
	start = time.time()
	current_loss = 0

	for iter in range(1, n_epochs + 1):
		avg_loss = train(rnn, criterion, optimizer, TRAIN_DATA, QUESTIONS, hidden_size, n_layers, batch_size)
		current_loss += avg_loss

		if iter % print_every == 0:
			d_MAP, d_MRR, d_P_1, d_P_5 = evaluate(rnn, DEV_DATA, DEV_LABEL_DICT, QUESTIONS, hidden_size, n_layers, batch_size=21)
			t_MAP, t_MRR, t_P_1, t_P_5 = evaluate(rnn, TEST_DATA, TEST_LABEL_DICT, QUESTIONS, hidden_size, n_layers, batch_size=21)
			print("Epoch %d: Average Train Loss: %.5f, Time: %s"%(iter, (current_loss / print_every), timeSince(start)))
			print("Dev MAP: %.1f, MRR: %.1f, P@1: %.1f, P@5: %.1f"%(d_MAP*100, d_MRR*100, d_P_1*100, d_P_5*100))
			print("Test MAP: %.1f, MRR: %.1f, P@1: %.1f, P@5: %.1f"%(t_MAP*100, t_MRR*100, t_P_1*100, t_P_5*100))
			current_loss = 0
	
import math
import torch.nn as nn
import time
from torch.autograd import Variable
import torch.optim as optim
from lstm_model_utils import *
from optparse import OptionParser

# PATHS
word_embedding_path = "../askubuntu-master/vector/vectors_pruned.200.txt.gz"
question_path = "../askubuntu-master/text_tokenized.txt.gz"
train_data_path = "../askubuntu-master/train_random.txt"
dev_data_path = "../askubuntu-master/dev.txt"
test_data_path = "../askubuntu-master/test.txt"

# CONSTANTS
OUTER_BATCH_SIZE = 25
HIDDEN_SIZE = 120
DFT_NUM_EPOCHS = 5
DFT_LEARNING_RATE = 7e-4
DFT_PRINT_EPOCHS = 1
DEBUG = True

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
	input_minibatch = Variable(torch.functional.stack([question_embedding[s] for s in sample]).permute(1,0,2), requires_grad=True)
	mask_minibatch = Variable(torch.functional.stack([question_mask[s] for s in sample]).permute(1,0,2), requires_grad=True)
	output_matrix = rnn(input_minibatch, hidden, state)*mask_minibatch
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
	cum_loss = 0
	hidden = Variable(torch.zeros(num_layers*2, batch_size, hidden_size))
	state = Variable(torch.zeros(num_layers*2, batch_size, hidden_size))
	rnn.zero_grad()

	TITLE_EMB, TITLE_MASK, BODY_EMB, BODY_MASK = question_data
	num_samples = train_data.size(0)
		
	for i in range(num_samples / OUTER_BATCH_SIZE):
		input_batch = train_data[i*OUTER_BATCH_SIZE:(i+1)*OUTER_BATCH_SIZE]
		X_scores = []

		for sample in input_batch:
			encoded_title_matrix = mean_pooling(rnn, hidden, state, TITLE_EMB, TITLE_MASK, sample)
			encoded_body_matrix = mean_pooling(rnn, hidden, state, BODY_EMB, BODY_MASK, sample)
			encoded_matrix = (encoded_title_matrix + encoded_body_matrix) / 2.0

			query_matrix = encoded_matrix[0].expand(21,hidden_size*2)
			candidate_matrix = encoded_matrix[1:]
			cos = nn.CosineSimilarity(dim=1)(query_matrix, candidate_matrix)
			X_scores.append(cos)

		X_scores = torch.stack(X_scores)
		targets = Variable(torch.zeros(X_scores.size(0)).type(torch.LongTensor))

		optimizer.zero_grad()
		loss = criterion(X_scores, targets)
		loss.backward()
		optimizer.step()
		cum_loss += loss.data[0]
	return cum_loss


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

		sorted_output = [golden_tags for _,golden_tags in sorted(zip(cos.data,golden_tags),reverse=True)]
		sorted_output_data.append(sorted_output)
	metrics = compute_metrics(sorted_output_data)
	return metrics


if __name__ == '__main__':
	# Create parser and extract arguments
	parser = OptionParser()
	parser.add_option("--batch_size", dest="batch_size", default=str(OUTER_BATCH_SIZE))
	parser.add_option("--hidden_size", dest="hidden_size", default=str(HIDDEN_SIZE))
	parser.add_option("--num_epochs", dest="num_epochs", default=str(DFT_NUM_EPOCHS))
	parser.add_option("--learning_rate", dest="learning_rate", default=str(DFT_LEARNING_RATE))
	parser.add_option("--print_epochs", dest="print_epochs", default=str(DFT_PRINT_EPOCHS))
	opts,args = parser.parse_args()

	# Set parameters
	n_layers = 1
	n_features = 200
	outer_batch_size = int(opts.batch_size)
	hidden_size = int(opts.hidden_size)
	learning_rate = float(opts.learning_rate)
	n_epochs = int(opts.num_epochs)
	print_every = int(opts.print_epochs)

	# Load data
	print("LOADING DATA...")
	embedding = create_embedding_dict(word_embedding_path)
	questions = create_question_dict(question_path, embedding, hidden_size)
	train_data = read_training_data(train_data_path)
	dev_data, dev_label_dict, dev_scores = read_eval_data(dev_data_path)
	test_data, test_label_dict, test_scores = read_eval_data(test_data_path)

	if DEBUG:
		train_data = train_data[:300]  # ONLY FOR DEBUGGING, REMOVE LINE TO RUN ON ALL TRAINING DATA

	# Create model
	rnn = RNN(n_features, hidden_size, n_layers, batch_size=22)
	optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
	criterion = nn.MultiMarginLoss(margin=0.2)

	# Training
	print("Starting run with batch_size: %d, hidden size: %d, learning rate: %.4f"%(outer_batch_size, hidden_size, learning_rate))
	start = time.time()
	current_loss = 0

	for iter in range(1, n_epochs + 1):
		avg_loss = train(rnn, criterion, optimizer, train_data, questions, hidden_size, n_layers, batch_size=22)
		current_loss += avg_loss

		if iter % print_every == 0:
			d_MAP, d_MRR, d_P_1, d_P_5 = evaluate(rnn, dev_data, dev_label_dict, questions, hidden_size, n_layers, batch_size=21)
			t_MAP, t_MRR, t_P_1, t_P_5 = evaluate(rnn, test_data, test_label_dict, questions, hidden_size, n_layers, batch_size=21)
			print("Epoch %d: Average Train Loss: %.5f, Time: %s"%(iter, (current_loss / print_every), timeSince(start)))
			print("Dev MAP: %.1f, MRR: %.1f, P@1: %.1f, P@5: %.1f"%(d_MAP*100, d_MRR*100, d_P_1*100, d_P_5*100))
			print("Test MAP: %.1f, MRR: %.1f, P@1: %.1f, P@5: %.1f"%(t_MAP*100, t_MRR*100, t_P_1*100, t_P_5*100))
			current_loss = 0
	
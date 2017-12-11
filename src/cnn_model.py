import time
import math
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
from optparse import OptionParser
from utils import *

# PATHS
word_embedding_path = "../askubuntu/vector/vectors_pruned.200.txt.gz"
question_path = "../askubuntu/text_tokenized.txt.gz"
train_data_path = "../askubuntu/train_random.txt"
dev_data_path = "../askubuntu/dev.txt"
test_data_path = "../askubuntu/test.txt"

# CONSTANTS
EMBEDDING_SIZE = 200
HIDDEN_SIZE = 667
LOSS_MARGIN = 0.2
KERNEL_SIZE = 4 # number of words to include in each feature map
DFT_LEARNING_RATE = 0.001
DFT_NUM_EPOCHS = 4
DFT_BATCH_SIZE = 20
DFT_PRINT_EPOCHS = 1
MAX_OR_MEAN_POOL = "MAX"
DEBUG = True

class CNN(nn.Module):

    def __init__(self, input_size, hidden_size, n, max_or_mean):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.n = n
        self.conv_layer = nn.Conv1d(in_channels=input_size,
                                    out_channels=hidden_size,
                                    kernel_size=n,
                                    stride=1,
                                    padding=0)
        self.activation_layer = nn.Tanh()
        self.max_or_mean = max_or_mean

    def forward(self, input):
        out = self.conv_layer(input)
        out = self.activation_layer(out)
        if self.max_or_mean == "MAX":
            out = nn.MaxPool1d(input.size()[2] - self.n + 1)(out)
        else:
            out = nn.AvgPool1d(input.size()[2] - self.n + 1)(out)
        return out


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def compute(model, question_embedding, sample):
    input_minibatch = Variable(torch.stack([question_embedding[s] for s in sample], dim=0), requires_grad=True)
    return model(input_minibatch)


def evaluate_sample(model, title_embedding, body_embedding, sample, num_comparisons):
    encoded_title_matrix = compute(model, title_embedding, sample)
    encoded_body_matrix = compute(model, body_embedding, sample)
    encoded_matrix = 0.5 * (encoded_title_matrix + encoded_body_matrix)
    encoded_matrix = encoded_matrix.squeeze(dim=2)
    query_matrix = encoded_matrix[0].expand(num_comparisons, model.hidden_size)
    candidate_matrix = encoded_matrix[1:]
    return nn.CosineSimilarity(dim=1)(query_matrix, candidate_matrix)


def train(model, criterion, optimizer, train_data, question_data, batch_size):
    cum_loss = 0
    model.zero_grad()

    TITLE_EMB, TITLE_MASK, BODY_EMB, BODY_MASK = question_data
    num_samples = train_data.size(0)

    for i in range(num_samples / batch_size): # loop through all samples, by batch
        optimizer.zero_grad()
        input_batch = train_data[i * batch_size:(i + 1) * batch_size]
        X_scores = []

        for sample in input_batch: # loop through each sample in batch
            cos = evaluate_sample(model, TITLE_EMB, TITLE_MASK, BODY_EMB, BODY_MASK, sample, 21)
            # 21 is the number of comparisons, 1 positive and 20 negative
            X_scores.append(cos)

        X_scores = torch.stack(X_scores)
        targets = Variable(torch.zeros(X_scores.size(0)).type(torch.LongTensor))

        loss = criterion(X_scores, targets)
        loss.backward()
        optimizer.step()
        cum_loss += loss.data[0]

    return cum_loss

def evaluate(model, eval_data, eval_label_dict, question_data):
    global print_result
    TITLE_EMB, TITLE_MASK, BODY_EMB, BODY_MASK = question_data
    sorted_output_data = []

    for sample in eval_data:
        cos = evaluate_sample(model, TITLE_EMB, BODY_EMB, sample, 20)
        golden_tags = eval_label_dict[sample[0]]

        sorted_output = [golden_tags for _, golden_tags in sorted(zip(cos.data, golden_tags), reverse=True)]
        sorted_output_data.append(sorted_output)
    metrics = compute_metrics(sorted_output_data)
    return metrics

if __name__ == '__main__':
    # Create parser and extract arguments
    parser = OptionParser()
    parser.add_option("--batch_size", dest="batch_size", default=str(DFT_BATCH_SIZE))
    parser.add_option("--hidden_size", dest="hidden_size", default=str(HIDDEN_SIZE))
    parser.add_option("--num_epochs", dest="num_epochs", default=str(DFT_NUM_EPOCHS))
    parser.add_option("--learning_rate", dest="learning_rate", default=str(DFT_LEARNING_RATE))
    parser.add_option("--print_epochs", dest="print_epochs", default=str(DFT_PRINT_EPOCHS))
    opts,args = parser.parse_args()

    # Set parameters
    batch_size = int(opts.batch_size)
    hidden_size = int(opts.hidden_size)
    learning_rate = float(opts.learning_rate)
    max_num_epochs = int(opts.num_epochs)
    print_epochs = int(opts.print_epochs)

    # Load data
    print("LOADING DATA...")
    embedding_dict = create_embedding_dict(word_embedding_path)
    questions = create_question_dict("CNN", question_path, embedding_dict, HIDDEN_SIZE)
    train_data = read_training_data(train_data_path)
    dev_data, dev_label_dict, dev_scores = read_eval_data(dev_data_path)
    test_data, test_label_dict, test_scores = read_eval_data(test_data_path)

    if DEBUG:
        train_data = train_data[:300]  # ONLY FOR DEBUGGING, REMOVE LINE TO RUN ON ALL TRAINING DATA

    # Create model
    cnn = CNN(EMBEDDING_SIZE, hidden_size, KERNEL_SIZE, MAX_OR_MEAN_POOL)
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    criterion = nn.MultiMarginLoss(margin=LOSS_MARGIN)

    # Training
    print("Starting run with batch_size: %d, hidden size: %d, learning rate: %.4f" % (
    batch_size, hidden_size, learning_rate))
    start = time.time()
    current_loss = 0

    d_MAP, d_MRR, d_P_1, d_P_5 = evaluate(cnn, dev_data, dev_label_dict, questions)
    print("Epoch %d: Average Train Loss: %.5f, Time: %s" % (0, (current_loss / print_epochs), timeSince(start)))
    print("Dev MAP: %.1f, MRR: %.1f, P@1: %.1f, P@5: %.1f" % (d_MAP * 100, d_MRR * 100, d_P_1 * 100, d_P_5 * 100))
    current_loss = 0

    for iter in range(1, max_num_epochs + 1):
        current_loss += train(cnn, criterion, optimizer, train_data, questions, batch_size)
        if iter % print_epochs == 0:
            d_MAP, d_MRR, d_P_1, d_P_5 = evaluate(cnn, dev_data, dev_label_dict, questions)
            t_MAP, t_MRR, t_P_1, t_P_5 = evaluate(cnn, test_data, test_label_dict, questions)
            print("Epoch %d: Average Train Loss: %.5f, Time: %s"%(iter, (current_loss / print_epochs), timeSince(start)))
            print("Dev MAP: %.1f, MRR: %.1f, P@1: %.1f, P@5: %.1f"%(d_MAP*100, d_MRR*100, d_P_1*100, d_P_5*100))
            print("Test MAP: %.1f, MRR: %.1f, P@1: %.1f, P@5: %.1f" % (t_MAP * 100, t_MRR * 100, t_P_1 * 100, t_P_5 * 100))
            current_loss = 0

# TODO: alternative to mask
# TODO: consolidate utils files

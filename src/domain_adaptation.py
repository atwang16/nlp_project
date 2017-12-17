import time
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from optparse import OptionParser
from utils import *
import os
import sys

# PATHS
word_embedding_path = "../glove.840B.300d.txt"
ubuntu_question_path = "../askubuntu-master/text_tokenized.txt.gz"
train_data_path = "../askubuntu-master/train_random.txt"
dev_data_path = "../askubuntu-master/dev.txt"
test_data_path = "../askubuntu-master/test.txt"
android_question_path = "../Android-master/corpus.tsv.gz"
dev_pos_data_path = "../Android-master/dev.pos.txt"
dev_neg_data_path = "../Android-master/dev.neg.txt"
test_pos_data_path = "../Android-master/test.pos.txt"
test_neg_data_path = "../Android-master/test.neg.txt"

# CONSTANTS
DFT_LAMBDA = 1e-4
DFT_EMBEDDING_SIZE = 200
DFT_HIDDEN_SIZE = 667
DFT_LOSS_MARGIN = 0.2
DFT_KERNEL_SIZE = 3 # number of words to include in each feature map
DFT_DROPOUT_PROB = 0.1
DFT_LEARNING_RATE_1 = 0.0001
DFT_LEARNING_RATE_2 = -0.0001
DFT_NUM_EPOCHS = 5
DFT_BATCH_SIZE = 20
DFT_PRINT_EPOCHS = 1
MAX_OR_MEAN_POOL = "MEAN"
DEBUG = True
DFT_SAVE_MODEL_PATH = os.path.join("..", "models", "cnn")
TRAIN_HYPER_PARAM = False
SAVE_MODEL = False

# HYPERPARAMETER TESTING
lambda_arr = [1e-3, 1e-5, 1e-7]
hidden_size_arr = [667, 333]
filter_width_arr = [2, 3, 4]
loss_margin_arr = [0.1, 0.2, 0.4]
dropout_prob_arr = [0.1, 0.2, 0.3]
learning_rate_arr_1 = [0.0002]
learning_rate_arr_2 = [-0.0002]
batch_size_arr = [20]
max_mean_arr = ["MAX", "MEAN"]
opt_model_params = {}
opt_mrr = (0.0, 0.0)

class Logger(object):
    """
    Stack Overflow Logger class: https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    """
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

sys.stdout = Logger()

class CNN(nn.Module):

    def __init__(self, input_size, hidden_size, n, max_or_mean, dropout_prob):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.n = n
        self.conv_layer = nn.Conv1d(in_channels=input_size,
                                    out_channels=hidden_size,
                                    kernel_size=n,
                                    stride=1,
                                    padding=0)
        self.batch_norm_layer = nn.BatchNorm1d(hidden_size)
        self.dropout_layer = nn.Dropout(p=dropout_prob)
        self.activation_layer = nn.Tanh()
        self.max_or_mean = max_or_mean

    def forward(self, input, mask, is_training):
        out = self.conv_layer(input)
        out = self.activation_layer(out)
        out = self.batch_norm_layer(out)
        if is_training:
            out = self.dropout_layer(out)
        if self.max_or_mean == "MAX":
            out = nn.MaxPool1d(input.size()[2] - self.n + 1)(out)
        else:
            mask_trimmed = mask.narrow(2, 0, out.size(2))
            out = out * mask_trimmed
            # out = nn.AvgPool1d(input.size()[2] - self.n + 1)(out)
            # print out.size()
            out = torch.sum(out, 2)
            # print out.size()
            out = out / torch.sum(mask_trimmed, 2)
            out = out.unsqueeze(2)
            # print out.size()
        return out


class FFN(nn.Module):
    def __init__(self, input_dim):
        super(NN, self).__init__()
        self.W_hidden_1 = nn.Linear(input_dim, 300)
        self.W_hidden_2 = nn.Linear(300, 150)
        self.W_out = nn.Linear(150, 2)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        hidden_1 = F.relu(self.W_hidden_1(input))
        hidden_2 = F.relu(self.W_hidden_1(hidden_1))
        out = self.softmax(self.W_out(hidden_2))
        return out


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def compute(model, question_embedding, question_mask, sample, is_training):
    input_minibatch = Variable(torch.stack([question_embedding[s] for s in sample], dim=0), requires_grad=True)
    mask_minibatch = Variable(torch.stack([question_mask[s] for s in sample], dim=0), requires_grad=False)
    return model(input_minibatch, mask_minibatch, is_training)


def evaluate_sample(model, question_data, sample, num_comparisons, is_training):
    title_embedding, title_mask, body_embedding, body_mask = question_data
    encoded_title_matrix = compute(model, title_embedding, title_mask, sample, is_training)
    encoded_body_matrix = compute(model, body_embedding, body_mask, sample, is_training)
    encoded_matrix = 0.5 * (encoded_title_matrix + encoded_body_matrix)

    encoded_matrix = encoded_matrix.squeeze(dim=2)
    query_matrix = encoded_matrix[0].expand(num_comparisons, model.hidden_size)
    candidate_matrix = encoded_matrix[1:]
    return nn.CosineSimilarity(dim=1)(query_matrix, candidate_matrix)


def train(model_1, model_2, criterion_1, criterion_2, optimizer_1, optimizer_2, train_data, question_data, batch_size, lambda_val):
    cum_loss = 0
    model.zero_grad()

    # Randomly shuffle training data
    rand_train_data = train_data[torch.LongTensor(np.random.permutation(train_data.size(0)))]

    num_samples = rand_train_data.size(0)

    for i in range(num_samples / batch_size): # loop through all samples, by batch
        ###############
        ### ENCODER ###
        ###############
        optimizer_1.zero_grad()
        input_batch = rand_train_data[i * batch_size:(i + 1) * batch_size]
        X_scores = []

        for sample in input_batch: # loop through each sample in batch
            cos = evaluate_sample(model, question_data, sample, 21, True)
            # 21 is the number of comparisons, 1 positive and 20 negative
            X_scores.append(cos)

        X_scores = torch.stack(X_scores)
        targets = Variable(torch.zeros(X_scores.size(0)).type(torch.LongTensor))

        #########################
        ### DOMAIN CLASSIFIER ###
        #########################
        optimizer_2.zero_grad()

        #TODO


        # backpropagation
        loss_1 = criterion_1(X_scores, targets)
        loss_2 = criterion_2()
        total_loss = loss_1 - lambda_val*loss_2
        total_loss.backward()
        optimizer_1.step()
        optimizer_2.step()
        cum_loss += total_loss.data[0]

    return cum_loss


def train_model(lambda_val, embedding_size, hidden_size, filter_width, max_or_mean, max_num_epochs, batch_size, learning_rate_1, learning_rate_2,
                loss_margin, training_checkpoint, dropout_prob):
    global load_model_path, train_data, questions, dev_data, dev_label_dict, test_data, test_label_dict, opt_mrr, opt_model_params

    # Generate model
    cnn = CNN(embedding_size, hidden_size, filter_width, max_or_mean, dropout_prob)
    optimizer_1 = optim.Adam(cnn.parameters(), lr=learning_rate_1)
    criterion_1 = nn.MultiMarginLoss(margin=loss_margin)
    ffn = FFN(hidden_size)
    optimizer_2 = optim.Adam(ffn.parameters(), lr=learning_rate_2)
    criterion_2 = nn.functional.cross_entropy()
    init_epoch = 1

    # Load model
    if load_model_path is not None:
        print("Loading model from \"" + load_model_path + "\"...")
        init_epoch = load_model(load_model_path, cnn, optimizer)

    # Training
    print("***************************************")
    print("Starting run with following parameters:")
    print(" --lambda:           %d" % (lambda_val))
    print(" --embedding size:   %d" % (cnn.input_size))
    print(" --hidden size:      %d" % (cnn.hidden_size))
    print(" --filter width:     %d" % (cnn.n))
    print(" --dropout:          %f" % (cnn.dropout_prob))
    print(" --pooling:          %s" % (cnn.max_or_mean))
    print(" --initial epoch:    %d" % (init_epoch))
    print(" --number of epochs: %d" % (max_num_epochs))
    print(" --batch size:       %d" % (batch_size))
    print(" --learning rate 1:  %f" % (learning_rate_1))
    print(" --learning rate 2:  %f" % (learning_rate_2))
    print(" --loss margin:      %f" % (loss_margin))

    start = time.time()
    current_loss = 0

    for iter in range(init_epoch, max_num_epochs + 1):
        current_loss += train(cnn, ffn, criterion_1, criterion_2, optimizer_1, optimizer_2, train_data, questions, batch_size, lambda_val)
        if iter % training_checkpoint == 0:
            d_MAP, d_MRR, d_P_1, d_P_5 = evaluate(cnn, dev_data, dev_label_dict, questions)
            t_MAP, t_MRR, t_P_1, t_P_5 = evaluate(cnn, test_data, test_label_dict, questions)
            print("Epoch %d: Average Train Loss: %.5f, Time: %s" % (
            iter, (current_loss / training_checkpoint), timeSince(start)))
            print("Dev MAP: %.1f, MRR: %.1f, P@1: %.1f, P@5: %.1f" % (
            d_MAP * 100, d_MRR * 100, d_P_1 * 100, d_P_5 * 100))
            print("Test MAP: %.1f, MRR: %.1f, P@1: %.1f, P@5: %.1f" % (
            t_MAP * 100, t_MRR * 100, t_P_1 * 100, t_P_5 * 100))
            current_loss = 0

            if SAVE_MODEL:
                state = {}
                state["model"] = cnn.state_dict()
                state["optimizer"] = optimizer.state_dict()
                state["epoch"] = iter
                save_model(save_model_path, "cnn", state, iter == max_num_epochs)

    # Compute final results
    print("-------")
    print("FINAL RESULTS:")
    d_MAP, d_MRR, d_P_1, d_P_5 = evaluate(cnn, dev_data, dev_label_dict, questions)
    t_MAP, t_MRR, t_P_1, t_P_5 = evaluate(cnn, test_data, test_label_dict, questions)
    print("Training time: %s" % (timeSince(start)))
    print("Dev MAP: %.1f, MRR: %.1f, P@1: %.1f, P@5: %.1f" % (
        d_MAP * 100, d_MRR * 100, d_P_1 * 100, d_P_5 * 100))
    print("Test MAP: %.1f, MRR: %.1f, P@1: %.1f, P@5: %.1f" % (
        t_MAP * 100, t_MRR * 100, t_P_1 * 100, t_P_5 * 100))

    if SAVE_MODEL:
        state = {}
        state["model"] = cnn.state_dict()
        state["optimizer"] = optimizer.state_dict()
        state["epoch"] = max_num_epochs if init_epoch < max_num_epochs else init_epoch
        save_model(save_model_path, "cnn", state, True)

    if TRAIN_HYPER_PARAM and d_MRR > opt_mrr[0]:
        opt_mrr = d_MRR, t_MRR
        opt_model_params = {"lambda": lambda_val,
                            "embedding_size": cnn.input_size,
                            "hidden_size": cnn.hidden_size,
                            "filter_width": cnn.n,
                            "dropout_prob": cnn.dropout_prob,
                            "pooling": cnn.max_or_mean,
                            "number of epochs": max_num_epochs,
                            "batch size": batch_size,
                            "learning rate 1": learning_rate_1,
                            "learning rate 2": learning_rate_2,
                            "loss margin": loss_margin}

    return (d_MAP, d_MRR, d_P_1, d_P_5)


def evaluate(model, eval_data, eval_label_dict, question_data):
    global print_result
    sorted_output_data = []

    for sample in eval_data:
        cos = evaluate_sample(model, question_data, sample, 20, False)
        golden_tags = eval_label_dict[sample[0]]

        sorted_output = [golden_tags for _, golden_tags in sorted(zip(cos.data, golden_tags), reverse=True)]
        sorted_output_data.append(sorted_output)
    metrics = compute_metrics(sorted_output_data)
    return metrics

if __name__ == '__main__':
    # Create parser and extract arguments
    parser = OptionParser()
    parser.add_option("--lambda", dest="lambda_val", default=str(DFT_LAMBDA))
    parser.add_option("--batch_size", dest="batch_size", default=str(DFT_BATCH_SIZE))
    parser.add_option("--hidden_size", dest="hidden_size", default=str(DFT_HIDDEN_SIZE))
    parser.add_option("--num_epochs", dest="num_epochs", default=str(DFT_NUM_EPOCHS))
    parser.add_option("--learning_rate_1", dest="learning_rate_1", default=str(DFT_LEARNING_RATE_1))
    parser.add_option("--learning_rate_2", dest="learning_rate_2", default=str(DFT_LEARNING_RATE_2))
    parser.add_option("--filter_width", dest="filter_width", default=str(DFT_KERNEL_SIZE))
    parser.add_option("--loss_margin", dest="loss_margin", default=str(DFT_LOSS_MARGIN))
    parser.add_option("--dropout", dest="dropout_prob", default=str(DFT_DROPOUT_PROB))
    parser.add_option("--max_or_mean", dest="max_or_mean", default=MAX_OR_MEAN_POOL)
    parser.add_option("--print_epochs", dest="print_epochs", default=str(DFT_PRINT_EPOCHS))
    parser.add_option("--load_model", dest="load_model_path", default=None)
    parser.add_option("--save_model", dest="save_model_path", default=DFT_SAVE_MODEL_PATH)
    opts,args = parser.parse_args()

    # Set parameters
    lambda_val = float(opts.lambda_val)
    batch_size = int(opts.batch_size)
    hidden_size = int(opts.hidden_size)
    learning_rate_1 = float(opts.learning_rate_1)
    learning_rate_2 = float(opts.learning_rate_2)
    max_num_epochs = int(opts.num_epochs)
    training_checkpoint = int(opts.print_epochs)
    load_model_path = opts.load_model_path
    save_model_path = opts.save_model_path
    filter_width = int(opts.filter_width)
    loss_margin = float(opts.loss_margin)
    dropout_prob = float(opts.dropout_prob)
    max_or_mean = opts.max_or_mean

    # Load data
    print("LOADING DATA...")
    embedding_dict = create_embedding_dict(word_embedding_path)
    questions = create_question_dict("CNN", question_path, embedding_dict, DFT_HIDDEN_SIZE, init_padding=filter_width - 1)
    train_data = read_training_data(train_data_path)
    dev_data, dev_label_dict, dev_scores = read_eval_data(dev_data_path)
    test_data, test_label_dict, test_scores = read_eval_data(test_data_path)

    if DEBUG:
        train_data = train_data[:300]  # ONLY FOR DEBUGGING, REMOVE LINE TO RUN ON ALL TRAINING DATA

    if TRAIN_HYPER_PARAM:
        i = 1
        total = len(lambda_arr) * len(hidden_size_arr) * len(filter_width_arr) * len(loss_margin_arr) * len(learning_rate_arr_1) * \
                len(learning_rate_arr_2) * len(batch_size_arr) * len(max_mean_arr) * len(dropout_prob_arr)
        for l in lambda_arr:
            for hs in hidden_size_arr:
                for fw in filter_width_arr:
                    for lm in loss_margin_arr:
                        for lr1 in learning_rate_arr_1:
                            for lr2 in learning_rate_arr_2:
                                for bs in batch_size_arr:
                                    for m in max_mean_arr:
                                        for dp in dropout_prob_arr:
                                            train_model(l,
                                                        DFT_EMBEDDING_SIZE,
                                                        hs,
                                                        fw,
                                                        m,
                                                        max_num_epochs,
                                                        bs,
                                                        lr1,
                                                        lr2,
                                                        lm,
                                                        training_checkpoint,
                                                        dp)
                                            print "Model " + str(i) + "/" + str(total)
                                            i += 1
        print("-------------")
        print("OPTIMAL MODEL:")
        print opt_mrr
        print opt_model_params

    else:
        train_model(lambda_val,
                    DFT_EMBEDDING_SIZE,
                    hidden_size,
                    filter_width,
                    max_or_mean,
                    max_num_epochs,
                    batch_size,
                    learning_rate_1,
                    learning_rate_2,
                    loss_margin,
                    training_checkpoint,
                    dropout_prob)


# TODO: retrain hyperparameters
# TODO: consolidate utils files
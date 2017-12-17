import torch.nn as nn
import torch.optim as optim
from meter import *
from optparse import OptionParser
from cnn_model import CNN, train, evaluate
from utils import *
import os
import sys
from torch.autograd import Variable

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
DFT_EMBEDDING_SIZE = 300
DFT_HIDDEN_SIZE = 667
DFT_LOSS_MARGIN = 0.2
DFT_KERNEL_SIZE = 3 # number of words to include in each feature map
DFT_DROPOUT_PROB = 0.3
DFT_LEARNING_RATE = 0.0002
DFT_NUM_EPOCHS = 5
DFT_BATCH_SIZE = 20
DFT_PRINT_EPOCHS = 1
MAX_OR_MEAN_POOL = "MEAN"
DFT_EVAL_BATCH_SIZE = 200
DEBUG = True
DFT_SAVE_MODEL_PATH = os.path.join("..", "models", "cnn")
TRAIN_HYPER_PARAM = True
SAVE_MODEL = False

#HYPERPARAMETER TESTING
hidden_size_arr = [667]
filter_width_arr = [3, 4]
loss_margin_arr = [0.1, 0.2]
dropout_prob_arr = [0.2, 0.3]
learning_rate_arr = [0.0002]
batch_size_arr = [20]
max_mean_arr = ["MEAN"]


class Logger(object):
    """
    Stack Overflow Logger class: https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

sys.stdout = Logger("../models/cnn/direct_transfer_results.txt")


def train_model(embedding_size, hidden_size, filter_width, max_or_mean, max_num_epochs, batch_size, learning_rate,
                loss_margin, training_checkpoint, dropout_prob, eval_batch_size):
    global load_model_path, train_data, source_questions
    global dev_data, dev_label_dict, test_data, test_label_dict
    global dev_pos_data, dev_neg_data, test_pos_data, test_neg_data, target_questions

    # Generate model
    cnn = CNN(embedding_size, hidden_size, filter_width, max_or_mean, dropout_prob)
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    criterion = nn.MultiMarginLoss(margin=loss_margin)
    init_epoch = 1

    # Load model
    if load_model_path is not None:
        print("Loading model from \"" + load_model_path + "\"...")
        init_epoch = load_model(load_model_path, cnn, optimizer)

    # Training
    print("***************************************")
    print("Starting run with following parameters:")
    print(" --embedding size:   %d" % (cnn.input_size))
    print(" --hidden size:      %d" % (cnn.hidden_size))
    print(" --filter width:     %d" % (cnn.n))
    print(" --dropout:          %f" % (cnn.dropout_prob))
    print(" --pooling:          %s" % (cnn.max_or_mean))
    print(" --initial epoch:    %d" % (init_epoch))
    print(" --number of epochs: %d" % (max_num_epochs))
    print(" --batch size:       %d" % (batch_size))
    print(" --learning rate:    %f" % (learning_rate))
    print(" --loss margin:      %f" % (loss_margin))

    start = time.time()
    current_loss = 0

    for iter in range(init_epoch, max_num_epochs + 1):
        current_loss += train(cnn, criterion, optimizer, train_data, source_questions, batch_size, 21)
        if iter % training_checkpoint == 0:
            d_MAP, d_MRR, d_P_1, d_P_5 = evaluate(cnn, dev_data, dev_label_dict, source_questions)
            t_MAP, t_MRR, t_P_1, t_P_5 = evaluate(cnn, test_data, test_label_dict, source_questions)
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
    d_auc = evaluate_auc(cnn, dev_pos_data, dev_neg_data, target_questions, eval_batch_size)
    t_auc = evaluate_auc(cnn, test_pos_data, test_neg_data, target_questions, eval_batch_size)
    print("Training time: %s" % (timeSince(start)))
    print("Dev AUC(0.05): %.2f" % (d_auc))
    print("Test AUC(0.05): %.2f" % (t_auc))

    if SAVE_MODEL:
        state = {}
        state["model"] = cnn.state_dict()
        state["optimizer"] = optimizer.state_dict()
        state["epoch"] = max_num_epochs if init_epoch < max_num_epochs else init_epoch
        save_model(save_model_path, "cnn", state, True)

    return (d_auc, t_auc)


def evaluate_auc(model, pos_data, neg_data, question_data, batch_size):
    auc = AUCMeter()

    evaluate_pair_set(model, pos_data, 1, question_data, auc, batch_size)
    evaluate_pair_set(model, neg_data, 0, question_data, auc, batch_size)

    return auc.value(max_fpr=0.05)

def evaluate_pair_set(model, pairs, label, question_data, auc, batch_size):
    for i in range(pairs.size(0) / batch_size):
        input_batch = pairs[i * batch_size:(i + 1) * batch_size]

        encoding_1 = compute_encoding(model, question_data, input_batch[:, 0])
        encoding_2 = compute_encoding(model, question_data, input_batch[:, 1])

        score = nn.CosineSimilarity(dim=1)(encoding_1, encoding_2)

        auc.add(score.data, torch.LongTensor([label]*batch_size))

def compute_encoding(model, question_data, list_of_ids):
    TITLE_EMB, TITLE_MASK, BODY_EMB, BODY_MASK = question_data

    title_embeddings = Variable(torch.stack([TITLE_EMB[id] for id in list_of_ids], dim=0), requires_grad=True)
    title_mask = Variable(torch.stack([TITLE_MASK[id] for id in list_of_ids], dim=0), requires_grad=False)
    body_embeddings = Variable(torch.stack([BODY_EMB[id] for id in list_of_ids], dim=0), requires_grad=True)
    body_mask = Variable(torch.stack([BODY_MASK[id] for id in list_of_ids], dim=0), requires_grad=False)

    title_matrix = model(title_embeddings, title_mask, False)
    body_matrix = model(body_embeddings, body_mask, False)
    encoded_matrix = 0.5 * (title_matrix + body_matrix)

    return encoded_matrix.squeeze(dim=2)


if __name__ == '__main__':
    # Create parser and extract arguments
    parser = OptionParser()
    parser.add_option("--batch_size", dest="batch_size", default=str(DFT_BATCH_SIZE))
    parser.add_option("--eval_batch_size", dest="eval_batch_size", default=str(DFT_EVAL_BATCH_SIZE))
    parser.add_option("--hidden_size", dest="hidden_size", default=str(DFT_HIDDEN_SIZE))
    parser.add_option("--num_epochs", dest="num_epochs", default=str(DFT_NUM_EPOCHS))
    parser.add_option("--learning_rate", dest="learning_rate", default=str(DFT_LEARNING_RATE))
    parser.add_option("--filter_width", dest="filter_width", default=str(DFT_KERNEL_SIZE))
    parser.add_option("--loss_margin", dest="loss_margin", default=str(DFT_LOSS_MARGIN))
    parser.add_option("--dropout", dest="dropout_prob", default=str(DFT_DROPOUT_PROB))
    parser.add_option("--max_or_mean", dest="max_or_mean", default=MAX_OR_MEAN_POOL)
    parser.add_option("--print_epochs", dest="print_epochs", default=str(DFT_PRINT_EPOCHS))
    parser.add_option("--load_model", dest="load_model_path", default=None)
    parser.add_option("--save_model", dest="save_model_path", default=DFT_SAVE_MODEL_PATH)
    opts, args = parser.parse_args()

    # Set parameters
    batch_size = int(opts.batch_size)
    hidden_size = int(opts.hidden_size)
    learning_rate = float(opts.learning_rate)
    max_num_epochs = int(opts.num_epochs)
    training_checkpoint = int(opts.print_epochs)
    load_model_path = opts.load_model_path
    save_model_path = opts.save_model_path
    filter_width = int(opts.filter_width)
    loss_margin = float(opts.loss_margin)
    dropout_prob = float(opts.dropout_prob)
    max_or_mean = opts.max_or_mean
    eval_batch_size = int(opts.eval_batch_size)

    # Load data
    print("LOADING DATA...")
    embedding = create_embedding_dict(word_embedding_path, True)
    source_questions = create_question_dict("CNN", ubuntu_question_path, embedding, hidden_size, embedding_size=DFT_EMBEDDING_SIZE, init_padding=filter_width - 1)
    target_questions = create_question_dict("CNN", android_question_path, embedding, hidden_size, embedding_size=DFT_EMBEDDING_SIZE, init_padding=filter_width - 1)
    train_data = read_training_data(train_data_path)
    dev_data, dev_label_dict, dev_scores = read_eval_data(dev_data_path)
    test_data, test_label_dict, test_scores = read_eval_data(test_data_path)
    dev_pos_data = read_android_eval_data(dev_pos_data_path)
    dev_neg_data = read_android_eval_data(dev_neg_data_path)
    test_pos_data = read_android_eval_data(test_pos_data_path)
    test_neg_data = read_android_eval_data(test_neg_data_path)

    if DEBUG:
        train_data = train_data[:400]  # ONLY FOR DEBUGGING, REMOVE LINE TO RUN ON ALL TRAINING DATA
        dev_neg_data = dev_neg_data[:20000]
        test_neg_data = dev_neg_data[:20000]

    if TRAIN_HYPER_PARAM:
        i = 1
        total = len(hidden_size_arr) * len(filter_width_arr) * len(loss_margin_arr) * len(learning_rate_arr) * \
                len(batch_size_arr) * len(max_mean_arr) * len(dropout_prob_arr)
        for hs in hidden_size_arr:
            for fw in filter_width_arr:
                for lm in loss_margin_arr:
                    for lr in learning_rate_arr:
                        for bs in batch_size_arr:
                            for m in max_mean_arr:
                                for dp in dropout_prob_arr:
                                    train_model(DFT_EMBEDDING_SIZE,
                                                hs,
                                                fw,
                                                m,
                                                max_num_epochs,
                                                bs,
                                                lr,
                                                lm,
                                                training_checkpoint,
                                                dp,
                                                eval_batch_size)
                                    print "Model " + str(i) + "/" + str(total)
                                    i += 1

    else:
        train_model(DFT_EMBEDDING_SIZE,
                    hidden_size,
                    filter_width,
                    max_or_mean,
                    max_num_epochs,
                    batch_size,
                    learning_rate,
                    loss_margin,
                    training_checkpoint,
                    dropout_prob,
                    eval_batch_size)

    
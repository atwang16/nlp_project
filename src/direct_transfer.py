import torch.nn as nn
import time
from torch.autograd import Variable
import torch.optim as optim
import pdb
from lstm_model_utils import *
from transfer_utils import *
from meter import *
from optparse import OptionParser
from cnn_model import CNN, train, compute
from utils import *
import os

# PATHS
word_embedding_path = "../glove.840B.300d.txt"
ubuntu_question_path = "../askubuntu-master/text_tokenized.txt.gz"
android_question_path = "../Android-master/corpus.tsv.gz"
train_data_path = "../askubuntu-master/train_random.txt"
dev_pos_data_path = "../Android-master/dev.pos.txt"
dev_neg_data_path = "../Android-master/dev.neg.txt"
test_pos_data_path = "../Android-master/test.pos.txt"
test_neg_data_path = "../Android-master/test.neg.txt"

# CONSTANTS
DFT_EMBEDDING_SIZE = 300
DFT_HIDDEN_SIZE = 667
DFT_LOSS_MARGIN = 0.1
DFT_KERNEL_SIZE = 3 # number of words to include in each feature map
DFT_DROPOUT_PROB = 0.3
DFT_LEARNING_RATE = 0.0002
DFT_NUM_EPOCHS = 5
DFT_BATCH_SIZE = 20
DFT_PRINT_EPOCHS = 1
MAX_OR_MEAN_POOL = "MEAN"
EVAL_BATCH_SIZE = 100
DEBUG = True
DFT_SAVE_MODEL_PATH = os.path.join("..", "models", "cnn")
TRAIN_HYPER_PARAM = True
SAVE_MODEL = False


def train_model(embedding_size, hidden_size, filter_width, max_or_mean, max_num_epochs, batch_size, learning_rate,
                loss_margin, training_checkpoint, dropout_prob):
    global load_model_path, train_data, source_questions
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
        current_loss += train(cnn, criterion, optimizer, train_data, source_questions, batch_size)
        if iter % training_checkpoint == 0:
            d_auc = evaluate(cnn, (dev_pos_data, dev_neg_data), target_questions)
            t_auc = evaluate(cnn, (test_pos_data, test_neg_data), target_questions)
            print("Epoch %d: Average Train Loss: %.5f, Time: %s" % (
            iter, (current_loss / training_checkpoint), timeSince(start)))
            print("Dev AUC(0.05): %.2f" % (d_auc))
            print("Test AUC(0.05): %.2f" % (t_auc))
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
    d_auc = evaluate(cnn, (dev_pos_data, dev_neg_data), target_questions)
    t_auc = evaluate(cnn, (test_pos_data, test_neg_data), target_questions)
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


def evaluate_batch(model, emb, mask, sample):
    encoded_title_matrix = compute(model, emb[0], mask[0], sample, False)
    encoded_body_matrix = compute(model, emb[1], mask[1], sample, False)
    encoded_matrix = 0.5 * (encoded_title_matrix + encoded_body_matrix)
    encoded_matrix = encoded_matrix.squeeze(dim=2)
    return encoded_matrix


def evaluate_data(model, data, emb, mask, meter, pos):
    query_tensor = data[:,0]
    candidate_tensor = data[:,1]
    encoded_queries = evaluate_batch(model, emb, mask, query_tensor)
    encoded_candidates = evaluate_batch(model, emb, mask, candidate_tensor)
    scores = nn.CosineSimilarity(dim=1)(encoded_queries, encoded_candidates)
    print scores.size()
    if pos:
        expected = torch.ones(EVAL_BATCH_SIZE).type(torch.LongTensor)
    else:
        expected = torch.zeros(EVAL_BATCH_SIZE).type(torch.LongTensor)
    print expected.size()
    meter.add(scores.data, expected)


def evaluate(model, eval_data, question_data):
    pos_data, neg_data = eval_data
    TITLE_EMB, TITLE_MASK, BODY_EMB, BODY_MASK = question_data
    meter = AUCMeter()

    # Evaluate positive data
    evaluate_data(model, pos_data, (TITLE_EMB, BODY_EMB), (TITLE_MASK, BODY_MASK), meter, True)

    # Evaluate negative data
    for i in range(neg_data.size(0) / EVAL_BATCH_SIZE):
        input_data = neg_data[i*EVAL_BATCH_SIZE:(i+1)*EVAL_BATCH_SIZE]
        evaluate_data(model, input_data, (TITLE_EMB, BODY_EMB), (TITLE_MASK, BODY_MASK), meter, False)

    score = meter.value(max_fpr=0.05)
    return score


if __name__ == '__main__':
    # Create parser and extract arguments
    parser = OptionParser()
    parser.add_option("--batch_size", dest="batch_size", default=str(DFT_BATCH_SIZE))
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

    # Load data
    print("LOADING DATA...")
    embedding = create_embedding_dict(word_embedding_path, True)
    source_questions = create_question_dict("CNN", ubuntu_question_path, embedding, hidden_size, embedding_size=DFT_EMBEDDING_SIZE)
    target_questions = create_question_dict("CNN", android_question_path, embedding, hidden_size, embedding_size=DFT_EMBEDDING_SIZE)
    train_data = read_training_data(train_data_path)
    dev_pos_data = read_android_eval_data(dev_pos_data_path)
    dev_neg_data = read_android_eval_data(dev_neg_data_path)
    test_pos_data = read_android_eval_data(test_pos_data_path)
    test_neg_data = read_android_eval_data(test_neg_data_path)
    
    if DEBUG:
        train_data = train_data[:300]  # ONLY FOR DEBUGGING, REMOVE LINE TO RUN ON ALL TRAINING DATA

    train_model(DFT_EMBEDDING_SIZE,
                hidden_size,
                filter_width,
                max_or_mean,
                max_num_epochs,
                batch_size,
                learning_rate,
                loss_margin,
                training_checkpoint,
                dropout_prob)

    
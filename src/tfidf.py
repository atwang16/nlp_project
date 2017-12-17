from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gzip
import sys
from meter import AUCMeter
import torch

# PATHS
CORPUS_PATH = "../Android-master/corpus.tsv.gz"
DEV_POS_PATH = "../Android-master/dev.pos.txt"
DEV_NEG_PATH = "../Android-master/dev.neg.txt"
TEST_POS_PATH = "../Android-master/test.pos.txt"
TEST_NEG_PATH = "../Android-master/test.neg.txt"

def read_raw_data(filename):
    question_to_index = {}
    text = []
    questions = gzip.open(filename, 'rb').read().strip().split('\n')
    for line in questions:
        id_question_split = line.split('\t')
        question_to_index[int(id_question_split[0])] = len(question_to_index)
        text.append(id_question_split[1] + " " + id_question_split[2])
    return (question_to_index, text)

def compute_tfidf(text):
    vectorizer = TfidfVectorizer()
    rep = vectorizer.fit_transform(text)
    return rep

def import_pairs(filename):
    pairs = []
    raw_pairs = open(filename, 'rb').read().strip().split('\n')
    for line in raw_pairs:
        line_split = line.split(' ')
        pairs.append([int(line_split[0]), int(line_split[1])])
    return pairs

def evaluate(pairs, label, text_data, question_lookup, auc):
    for p in pairs:
        id_1, id_2 = p
        cos = cosine_similarity(text_data.getrow(question_lookup[id_1]),
                                text_data.getrow(question_lookup[id_2]))
        cos = float(cos)

        auc.add(torch.DoubleTensor([cos]), torch.LongTensor([label]))

print >> sys.stderr, "LOADING DATA..."
question_lookup, text_raw = read_raw_data(CORPUS_PATH)
dev_pos_pairs = import_pairs(DEV_POS_PATH)
dev_neg_pairs = import_pairs(DEV_NEG_PATH)
test_pos_pairs = import_pairs(TEST_POS_PATH)
test_neg_pairs = import_pairs(TEST_NEG_PATH)

print >> sys.stderr, "COMPUTING TF-IDF FEATURES..."
text_tfidf = compute_tfidf(text_raw)

print >> sys.stderr, "COMPUTING AUC..."
d_auc = AUCMeter()
t_auc = AUCMeter()
evaluate(dev_pos_pairs, 1, text_tfidf, question_lookup, d_auc)
evaluate(dev_neg_pairs, 0, text_tfidf, question_lookup, d_auc)
evaluate(test_pos_pairs, 1, text_tfidf, question_lookup, t_auc)
evaluate(test_neg_pairs, 0, text_tfidf, question_lookup, t_auc)

print "Dev AUC: %.2f" % (d_auc.value(max_fpr=0.05))
print "Test AUC: %.2f" % (t_auc.value(max_fpr=0.05))

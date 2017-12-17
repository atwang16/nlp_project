import numpy as np
import torch
import pdb
import gzip

# reads eval input data and creates:
# 1) eval matrix, each row with 2 candidate questions
def read_android_eval_data(filename):
	eval_matrix = []
	f = open(filename, 'r')
	for line in f.readlines():
		candidates = [int(v) for v in line.split()]
		eval_matrix.append(torch.Tensor(candidates))

	eval_tensor = torch.functional.stack(eval_matrix)
	return eval_tensor


def read_android_eval_data_for_transfer(filename):
	eval_matrix = []
	question_set = set()
	f = open(filename, 'r')
	for line in f.readlines():
		candidates = [int(v) for v in line.split()]
		eval_matrix.append(torch.Tensor(candidates))
		question_set = question_set.union(set(candidates))

	eval_tensor = torch.functional.stack(eval_matrix)
	return eval_tensor, question_set
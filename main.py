import os
import sys
import argparse
from src.utils import *
from src.utils_data import *


from model import HGSL


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./data')
parser.add_argument('--dataset_name', type=str, default='acm')
parser.add_argument('--num_node', type=float, default=508)
parser.add_argument('--alpha', type=float, default=1000)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--step_size', type=float, default=0.01)
parser.add_argument('--threshold', type=float, default=0.0034)
parser.add_argument('--log_save_dir', type=str, default='./log')
args = parser.parse_args()

device = 'cpu'

#build log
check_folder(args.log_save_dir)
log_file_name = os.path.join(args.log_save_dir, 'log.txt')
saver = open(log_file_name, "w")

# Logging the details for this experiment
saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
saver.write(args.__repr__() + "\n\n")
saver.flush()

#get dataset
dataset_file_name = '{}.pickle'.format(args.dataset_name)
dir_dataset = '{}/{}'.format(args.dataset_path, dataset_file_name)
node_feature, gt_inc_matrix = data_loading(dir_dataset)
gt_inc_set = icm_2_set(gt_inc_matrix)
gt_inc_matrix0 = gt_inc_matrix.sum(0)
gt_inc_matrix1 = gt_inc_matrix.sum(1)
#test model
hgsl = HGSL(device, args)
learnt_inc_matrix = hgsl.forward(node_feature)
learnt_inc_matrix_set = icm_2_set(learnt_inc_matrix)
f1, presicion, recall = compute_metric_set(learnt_inc_matrix_set, gt_inc_set)

results = 'f1:{}\npresicion:{}\nrecall:{}'.format(f1, presicion, recall)
print(results)
saver.write(results)
saver.flush()
saver.close()
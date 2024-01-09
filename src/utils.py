import numpy as np
import math
import os
import torch
import scipy.sparse as sparse

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    return folder_path


def torch_sqaureform_to_matrix_v(w, device):
    """
    from half vectorisation to matrix in non-batch way.
    """

    l = w.shape[0]
    m = int((1 / 2) * (1 + math.sqrt(1 + 8 * l)))

    E = torch.zeros((m, m), dtype = w.dtype).to(device)

    E[:, :][np.triu_indices(m, 1)] = w.clone().detach()
    E[:, :] = E[:, :].T + E[:, :]

    return E

def torch_squareform_to_vector_v(A, device):
    m, _ = A.shape
    l = int(m * (m - 1) / 2)

    w = torch.zeros((l), dtype = A.dtype).to(device)

    w[:] = A[:,:][np.triu_indices(m, 1)].clone().detach()

    return w

def torch_get_distance_halfvector(x, device):
    dist_x = torch.cdist(x,x)
    z = torch_squareform_to_vector(dist_x, device)
    
    return z

def torch_sqaureform_to_matrix(w, device):
    """
    from half vectorisation to matrix in batch way.
    """

    batch_size, l = w.size()
    m = int((1 / 2) * (1 + math.sqrt(1 + 8 * l)))

    E = torch.zeros((batch_size, m, m), dtype = w.dtype).to(device)

    for i in range(batch_size):
        E[i, :, :][np.triu_indices(m, 1)] = w[i].clone().detach()
        E[i, :, :] = E[i, :, :].T + E[i, :, :]

    return E

def torch_squareform_to_vector(A, device):
    batch_size, m, _ = A.size()
    l = int(m * (m - 1) / 2)

    w = torch.zeros((batch_size, l), dtype = A.dtype).to(device)

    for i in range(batch_size):
        w[i, :] = A[i,:,:][np.triu_indices(m, 1)].clone().detach()

    return w

def check_tensor(x, device):
    if isinstance(x, np.ndarray) or type(x) in [int, float]:
        x = torch.Tensor(x)
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x

def coo_to_sparseTensor(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

def get_degree_operator(m):
    ncols =int(m*(m - 1)/2)

    I = np.zeros(ncols)
    J = np.zeros(ncols)

    k = 0
    for i in np.arange(1, m):
        I[k:(k + m - i)] = np.arange(i, m)
        k = k + (m - i)

    k = 0
    for i in np.arange(1, m):
        J[k: (k + m - i)] = i - 1
        k = k + m - i

    Row = np.tile(np.arange(0, ncols), 2)
    Col = np.append(I, J)
    Data = np.ones(Col.size)
    St = sparse.coo_matrix((Data, (Row, Col)), shape=(ncols, m))
    return St.T

def compute_TP(gt_im, est_im):
    TP = 0
    gt_node_num, gt_hyper_num = gt_im.shape
    
    for idx_hyper in range(gt_hyper_num):
        gt_hyperedge = gt_im[:,idx_hyper]
        gt_hyperedge = gt_hyperedge.reshape(gt_node_num,1)
        temp =  torch.abs((est_im - gt_hyperedge))
        temp = temp.sum(0)
        temp = (temp == 0) * 1
        stemp = temp.sum()
        if((stemp >= 1)):
            TP = TP + 1
            
    return TP

def compute_FP_FN(TP,gt_im,est_im):
    
    _, gt_hyper_num = gt_im.shape
    _, est_hyper_num = est_im.shape
    
    FP = est_hyper_num - TP
    
    FN = gt_hyper_num - TP
    
    return FP, FN

def compute_F1(TP,FP,FN):
    
    if (TP==0):
        return 0,0,0
    
    precision = TP / (TP + FP)
    
    recall = TP / (TP + FN)
    
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1, precision, recall

def icm_2_set(icm):
    _, edge_num = icm.shape
    edge_set = []
    for i in range(edge_num):
        hedge = icm[:,i]
        t = (hedge == 1).nonzero(as_tuple=False)
        t = t[:,0].tolist()
        t.sort()
        if(t in edge_set):
            continue
        edge_set.append(t)
    return edge_set

def compute_metric_set(est_im, gt_im):
    TP = compute_TP_set(gt_im, est_im)
    FP, FN = compute_FP_FN_set(TP,gt_im,est_im)
    f1, precision, recall = compute_F1(TP,FP,FN)
    
    return f1, precision, recall

def compute_TP_set(gt_im, est_im):
    TP = 0
    for hyperedge in gt_im:
        if(hyperedge in est_im):
            TP += 1
    return TP

def compute_FP_FN_set(TP,gt_im,est_im):
    gt_hyper_num = len(gt_im)
    est_hyper_num = len(est_im)
    
    FP = est_hyper_num - TP
    
    FN = gt_hyper_num - TP
    
    return FP, FN
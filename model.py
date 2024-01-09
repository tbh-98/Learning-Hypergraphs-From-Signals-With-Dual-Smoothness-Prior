import torch
import leidenalg
import igraph
from optimizer import PDS
from src.utils import *

class HGSL():
    def __init__(self, device, args):
        self.threshold = args.threshold
        self.device = device
        self.num_node = args.num_node
        self.pds = PDS(device, args)
            
    def fine_hyperedge_size1(self, ic_matrix, node_a):
        node_as = node_a.sum(-1)
        nonz = (node_as == 0).nonzero(as_tuple=True)[0]
        ic_matrix_tensor_new = torch.zeros(self.num_node,len(nonz)).to(self.device)
        for i in range(len(nonz)):
            ic_matrix_tensor_new[nonz[i],i] = 1
        
        ic_matrix = torch.cat((ic_matrix,ic_matrix_tensor_new),-1)

        return ic_matrix
    
    
    def get_hyperedges(self, ic_matrix, i, enodes, edge_nodes, enode_list = None):
        nodes = []
        for enode in (enodes):
            if(enode_list is not None):
                enode = enode_list[enode][0]
            edge_node = edge_nodes[enode].tolist()
            nodes = nodes + edge_node
            nodes = list(set(nodes))
        nodes = torch.tensor(nodes).long()
        if(i == 0):
            ic_matrix[nodes,0] = 1
            i += 1
        else:
            ic_matrix_tensor_new = torch.zeros(self.num_node,1).to(self.device)
            ic_matrix_tensor_new[nodes,0] = 1
            ic_matrix = torch.cat((ic_matrix,ic_matrix_tensor_new),-1)
        
        return ic_matrix, i
    
    def form_hypergraph(self, combinations, edge_nodes, node_a, iso_edges, node_list):
        ic_matrix = torch.zeros(self.num_node, 1).to(self.device)
        i = 0
        for combination in combinations:
            ic_matrix, i = self.get_hyperedges(ic_matrix, i, list(combination), edge_nodes, node_list)
        for iso_edge in iso_edges:
            ic_matrix, i = self.get_hyperedges(ic_matrix, i, iso_edge, edge_nodes)
        
        ic_matrix = self.fine_hyperedge_size1(ic_matrix, node_a)
        
        return ic_matrix
    
    def community_detection(self,w):
        w = w.cpu().detach().numpy()
        G = igraph.Graph.Weighted_Adjacency(w)
        communities = list(leidenalg.find_partition(G, leidenalg.ModularityVertexPartition))
        
        return communities
    
    def get_iso_edges(self, line_graph): #[l, l]
        line_graph = line_graph.sum(-1) # [l]
        iso_edges_indicator = (line_graph == 0) * 1 # [l]
        
        return iso_edges_indicator
    
    def get_line_graph(self, edge_node, edge_f): #[l,2]; [n,n], [l,f]
        # get the structure of the line graph
        position_indicator = torch.zeros(edge_node.shape[0], self.num_node).to(self.device) #[l,n]
        node_indicator = edge_node.reshape(-1) #[l*2]
        l_indicator = torch.arange(edge_node.shape[0]).to(self.device).long() #[l]
        l_indicator = l_indicator.unsqueeze(-1).repeat(1, edge_node.shape[1]).reshape(-1) #[l*2]
        
        position_indicator[l_indicator, node_indicator] = 1 #[l,n]
        
        ori_line_graph = position_indicator @ position_indicator.t() #[l,l]
        ori_line_graph.fill_diagonal_(0)
        
        # get isolated edges
        iso_edges_indicator = self.get_iso_edges(ori_line_graph)
        
        # get edge features and non-zero line graph
        ori_line_graph, node_list, _ = self.get_non_zero_graph(ori_line_graph)
        edge_f = edge_f[node_list]
        ori_line_graph = torch_squareform_to_vector_v(ori_line_graph, self.device)
        ori_line_graph = (ori_line_graph > 0) * 1.0
        iso_edges_indicator = (iso_edges_indicator == 1).nonzero(as_tuple=False)
        
        return ori_line_graph, iso_edges_indicator, edge_f, node_list #[l*l], [l'], [l, f]

    def refine_line_graph(self, edge_f, ori_edge_s, p = 1): #[l,f] [l*l]
        edge_a_weight = torch_get_distance_halfvector(edge_f, self.device) #[b,l*l]
        edge_a_weight = edge_a_weight[0]
        edge_a_weight = torch.exp(-p * edge_a_weight)
        
        edge_a_weight = edge_a_weight * ori_edge_s
            
        return edge_a_weight
    
    def set_rate_threshold(self, w, rate):
        _, edge_num = w.shape
        top_k_num = int((edge_num * rate))
        topk, _ = torch.topk(w,k=top_k_num,dim=-1)
        threshold = topk[:,-1]
        threshold = threshold.unsqueeze(-1)
        w = (w >= threshold) * 1.0
        w = torch_sqaureform_to_matrix(w, self.device)
        
        return w
    
    def norm_feature(self, feature):
        std = feature.std(axis=-2, keepdims=True)
        m = feature.mean(axis=-2, keepdims=True)
        feature = ((feature - m) / (std + 0.00000001))
        return feature
    
    def get_non_zero_graph(self, w, threshold = 0):
        edge_node = torch.triu(w, diagonal=1)
        edge_node = (edge_node == 1).nonzero(as_tuple=False)
        
        w_s = w.sum(-1)
        w_s = (w_s > threshold) * 1
        node_list = w_s.nonzero(as_tuple=False)
        node_idx = node_list.reshape(-1)
        nz_edge_a_vsi = w[node_idx,:]
        nz_edge_a_vsi = nz_edge_a_vsi[:,node_idx]
        
        return nz_edge_a_vsi, node_list, edge_node
    
    def forward(self, x):
        x = self.norm_feature(x)
        x = x.to(self.device)
        z = torch_get_distance_halfvector(x, self.device)

        ws = self.pds.solve(z, 500)
        w = ws[:,-1,:]
        node_a_matrix = self.set_rate_threshold(w, self.threshold)
        
        _, _, edge_node = self.get_non_zero_graph(node_a_matrix[0])
                
        # get edge features
        aif = torch_squareform_to_vector_v(node_a_matrix[0], self.device)
        edge_f = w[0]
        edge_f = edge_f[(aif == 1)]
                
        ori_edge_s, iso_edges, edge_f, node_list = self.get_line_graph(edge_node, edge_f)
                
        if(edge_f.shape[0] == 0):
            community = []
        else:
            edge_f = edge_f.unsqueeze(0)
            edge_a_vs = self.refine_line_graph(edge_f, ori_edge_s)
            edge_a_vs = torch_sqaureform_to_matrix_v(edge_a_vs, self.device)
            if(edge_a_vs.sum() == 0):
                community = []
            else:
                community = self.community_detection(edge_a_vs)
                        
        inc_matrix = self.form_hypergraph(community, edge_node, node_a_matrix[0], iso_edges, node_list)
        
        return inc_matrix
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat


#coding eta(t) in dynamic graph (Refer to 4.1)
class ModuleTimestamping(nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        #GRU : Gated Recurrence Unit : A variant of RNN
        #nn.GRU input : the number of input features / the number of features in hidden state / num_layers / dropout
        #nn.GRU output[0] : tensor of shape (sequence length , batch size , hidden feature size) - (not bidirectional->hidden feature size * 1)

    def forward(self, t, sampling_endpoints):
        return self.rnn(t[:sampling_endpoints[-1]])[0][[p-1 for p in sampling_endpoints]]
        #input val : time from 0 to the last sampling endpoint
        #input val shape : (len(sampling_endpoints),1,1) due to broadcasting
        #self.rnn(t[:sampling_endpoints[-1]])[0] : tensor of shape same as the input shape
        #final return val : tensor of output values in the index of (values 1 before the value in the sampling_endpoints)
        #Ex. t=1200, sampling_endpoints=[300, 600, 900] => return val = h299, h599, h899
        #To conclude, the output is the output value of the GRU at the end of each interval
        
#Section 3.2
#Each layer of Graph Isomorphism Network
#AGGREAGATE & COMBINE defined in the forward function before READOUT
#epsilon parameter : whether the adjacency matrix includes self-loop - broadcasting used
class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]])) # assumes that the adjacency matrix includes self-loop
        else: self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())

    # a : adjacency matrix between the node feature, of shape (N, N)
    # v : feature matrix of that layer of shape (N, D)
    def forward(self, v, a):
        v_aggregate = torch.sparse.mm(a, v)
        v_aggregate += self.epsilon * v # assumes that the adjacency matrix includes self-loop
        v_combine = self.mlp(v_aggregate)
        return v_combine

#Mean Readout Module
class ModuleMeanReadout(nn.Module):
    #*args : enables the user to input the parameters regardless of how many parameters are there
    #**kwargs : enables the user to input the parameters with "name=value" format
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1,1,1], dtype=torch.float32)

#4.2.2. SERO module
#Squeeze-and-Excitation : refer to https://wwiiiii.tistory.com/entry/SqueezeandExcitation-Networks
class ModuleSERO(nn.Module):
    #input_dim == N in the paper
    #hidden_dim == D in the paper
    #as upscale==1.0, hidden_dim == upscale*hidden_dim so W1 matrix is of shape (D, D)
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        #W1 matrix of shape (D, D)
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale*hidden_dim)), nn.BatchNorm1d(round(upscale*hidden_dim)), nn.GELU())
        #W2 matrix of shape (N, D)
        self.attend = nn.Linear(round(upscale*hidden_dim), input_dim)
        self.dropout = nn.Dropout(dropout)

    #x is the H in the paper of shape (A, N, B, D) - A and B are arbitrary number denoting ... below
    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_readout = x.mean(node_axis) #PHImean * H
        x_shape = x_readout.shape # (A, B, D)
        x_embed = self.embed(x_readout.reshape(-1,x_shape[-1])) # x_embed's shape : (A*B, D) 
        #Asterisk before tensor name : unpacking all the values in the tensor
        #x_graphattention = Zspace calculation + reshaping => tensor shape of (A, B, N)
        #x_graphattention can be interpreted as the attention value for values in the feature matrix
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1],-1)
        permute_idx = list(range(node_axis))+[len(x_graphattention.shape)-1]+list(range(node_axis,len(x_graphattention.shape)-1)) #[0,2,1]
        x_graphattention = x_graphattention.permute(permute_idx) #shape of (A, N, B)
        #x_graphattention.unsqueeze(-1)'s shape : (A, N, B, 1)
        #Broadcasting applied when multiplying x and dropout
        #After applying mean(node_axis), the shape changes to (A, B, D))
        #returns spatially attended graph representation
        #return shape : (A, B, D), (N, A, B)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)


class ModuleGARO(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim)) #Wquery : (D, D)
        self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))   #Wkey : (D, D)
        self.dropout = nn.Dropout(dropout)

    #x is the feature matrix H in the paper, of shape (A, N, B, D)
    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        #x.mean(node_axis, keepdims=True) => shape of (A, 1, B, D)
        x_q = self.embed_query(x.mean(node_axis, keepdims=True)) #q formula in 4.2.1 - shape of (A, 1, B, D)
        x_k = self.embed_key(x)                                  #K formula in 4.2.1 - shape of (A, N, B, D)
        #result shape of rearrange : (A, 1, D, B)
        #result shape of matmul : (A, N, B, B)
        x_graphattention = torch.sigmoid(torch.matmul(x_q, rearrange(x_k, 't b n c -> t b c n'))/np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)


#Transformer module using self-attention layer
#Refer to "Attention is all you need" paper
#Self-attention -> Residual connection -> Layer normalization -> Individual MLP 
# -> Residual connection -> Layer normalization -> Output attention vector
#In STAGIN, the input vector is the sequence of graph features obtained by spatial READOUT
class ModuleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        #nn.MultiheadAttention's parameter : total dimension of the model / number of parallel attention heads
        #nn.LayerNorm : apply layer normalization over input_dim dimensions - y = (x-E)/sqrt(Var+epsilon) * gamma + beta 
        #self.mlp : MultiLayer Perceptron independently applied on each input vector after layer normalization
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))


    def forward(self, x):
        #multihead_attn's parameter : query / key / value
        #query : the criteria of comparison of key
        #key : the value supposed to be compared with query
        #value : the value used to obtain attention weights based on similarity scores
        #In self-attention layer, query/key/value are in the same set
        x_attend, attn_matrix = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x_attend) # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix

#STAGIN module
class ModelSTAGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads, num_layers, sparsity, dropout=0.5, cls_token='sum', readout='sero', garo_upscale=1.0):
        super().__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token=='sum': self.cls_token = lambda x: x.sum(0)
        elif cls_token=='mean': self.cls_token = lambda x: x.mean(0)
        elif cls_token=='param': self.cls_token = lambda x: x[-1]
        else: raise #raise error
        if readout=='garo': readout_module = ModuleGARO
        elif readout=='sero': readout_module = ModuleSERO
        elif readout=='mean': readout_module = ModuleMeanReadout
        else: raise

        self.token_parameter = nn.Parameter(torch.randn([num_layers, 1, 1, hidden_dim])) if cls_token=='param' else None

        self.num_classes = num_classes
        self.sparsity = sparsity

        # define modules
        self.timestamp_encoder = ModuleTimestamping(input_dim, hidden_dim, hidden_dim)
        self.initial_linear = nn.Linear(input_dim+hidden_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        #Add LayerGIN module defined above in ModuleList gnn_layers => add LayerGIN module to STAGIN module
        #In the same way, add ModuleTransformer and Linear to the STAGIN
        for i in range(num_layers):
            self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))
            self.readout_modules.append(readout_module(hidden_dim=hidden_dim, input_dim=input_dim, dropout=0.1))
            self.transformer_modules.append(ModuleTransformer(hidden_dim, 2*hidden_dim, num_heads=num_heads, dropout=0.1))
            self.linear_layers.append(nn.Linear(hidden_dim, num_classes))

    #Obtain dense adjacency matrix
    #sparse adjacency matrix : matrix that only contains the indices of the nonzero elements
    #dense adjacency matrix : matrix that contains 1s at the connected indices and otherwise 0s
    #Actual value input in a is a 4D tensor of shape [minibatch x time x node x node] - refer to forward() below
    def _collate_adjacency(self, a, sparse=True):
        i_list = []
        v_list = []
        #sample : index of repetition on a
        #_dyn_a : same as a[sample]
        for sample, _dyn_a in enumerate(a):
            #timepoint : index of repetition on _dyn_a
            #_a : same as _dyn_a[timepoint] == a[sample][timepoint]
            #In forward(), we assume the shape of a as [minibatch x time x node x node]
            for timepoint, _a in enumerate(_dyn_a):
                #detach() : generate the copied tensor which cannot be backpropagated(cannot be updated)
                #detach().cpu().numpy() : deliver the data from GPU to CPU and eventually to numpy
                #thresholded_a : boolean type tensor
                #if _a is over 100-sparsity percentile of initial _a, then true else false
                thresholded_a = (_a > np.percentile(_a.detach().cpu().numpy(), 100-self.sparsity))
                #_i : tensor containing the nonzero indices of thresholded_a
                #_i indicates the connected edge of the [node x node] matrix
                #_i's shape : (z, 2) where z is the number of nonzero values in thresholded_a tensor
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)
        
        #Sparse adjacency matrix to dense adjacency matrix
        #Refer to https://stackoverflow.com/questions/65263666/how-to-convert-sparse-to-dense-adjacency-matrix
        return torch.sparse.FloatTensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3]))

    # v : feature matrix
    # a : adjacency matrix
    # t : time length
    # sampling_endpoints : list of endpoints dividing time interval into pieces
    def forward(self, v, a, t, sampling_endpoints):
        # assumes shape [minibatch x time x node x feature] for v
        # assumes shape [minibatch x time x node x node] for a
        
        # logit : the value used in binary classification - ln(p/(1-p)) for probability p
        # logit range : [0,1] -> (-inf, inf)
        # In the logistic regression, we can use the logit value in the following form : logit(p)=wx+b
        # We can train the model to find the appropriate w and b value
        logit = 0.0
        reg_ortho = 0.0
        attention = {'node-attention': [], 'time-attention': []}
        latent_list = []
        minibatch_size, num_timepoints, num_nodes = a.shape[:3]

        #time_encoding : number of intervals x batch size x hidden feature size
        time_encoding = self.timestamp_encoder(t, sampling_endpoints)
        time_encoding = repeat(time_encoding, 'b t c -> t b n c', n=num_nodes)

        # h : node feature matrix H in the paper
        # cat : concatenating timestamp and feature vector
        h = torch.cat([v, time_encoding], dim=3)
        h = rearrange(h, 'b t n c -> (b t n) c')
        # h's final shape : (N, D)
        h = self.initial_linear(h)

        # a : dense adjacency matrix
        a = self._collate_adjacency(a)
        for layer, (G, R, T, L) in enumerate(zip(self.gnn_layers, self.readout_modules, self.transformer_modules, self.linear_layers)):
            #Pass the feature + timestamp matrix to GIN layer
            h = G(h, a)
            #h_bridge is the 4D tensor from h
            h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
            #Pass the GIN result to spatial readout module - node_axis=2 since node dimension is 2
            h_readout, node_attn = R(h_bridge, node_axis=2)
            #if cls_token == sum or mean -> set the shape of token_parameter[layer] according to h_readout's shape and concatenate to h_readout 
            if self.token_parameter is not None: h_readout = torch.cat([h_readout, self.token_parameter[layer].expand(-1,h_readout.shape[1],-1)])
            #pass the matrix to temporal readout module - the transformer layer
            h_attend, time_attn = T(h_readout)
            #Orthogonal regularization
            ortho_latent = rearrange(h_bridge, 't b n c -> (t b) n c')
            matrix_inner = torch.bmm(ortho_latent, ortho_latent.permute(0,2,1))
            reg_ortho += (matrix_inner/matrix_inner.max(-1)[0].unsqueeze(-1) - torch.eye(num_nodes, device=matrix_inner.device)).triu().norm(dim=(1,2)).mean()

            #cls_token : special classification token introduced in the BERT paper
            #last hidden state of model corresponding to the token is used in classification
            #In this paper, 3 options - sum / mean / param - is the possible cls_token
            #sum : sum(dim=0)
            #mean : mean(dim=0) 
            #param : the last element
            latent = self.cls_token(h_attend)
            #L(latent) : final graph representation
            logit += self.dropout(L(latent))

            attention['node-attention'].append(node_attn)
            attention['time-attention'].append(time_attn)
            latent_list.append(latent)

        attention['node-attention'] = torch.stack(attention['node-attention'], dim=1).detach().cpu()
        attention['time-attention'] = torch.stack(attention['time-attention'], dim=1).detach().cpu()
        latent = torch.stack(latent_list, dim=1)

        return logit, attention, latent, reg_ortho


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import torch
import torch_geometric
from torch.nn import Sequential, Linear, ReLU, LayerNorm, LSTM, GRU
from torch_geometric.nn import MessagePassing
import functools
import collections
from torch_geometric.data import Data

# Graph = collections.namedtuple('Graph', ['edge_index', 'node_features', 'edge_features'])

class GraphNetBlock(MessagePassing):
    """Message passing."""
    
    def __init__(self,latent_size, in_size1, in_size2): 
        super(GraphNetBlock, self).__init__(aggr='add')        
        self._latent_size = latent_size
        
        # First net (MLP): eij' = f1(xi, xj, eij)
        self.edge_net = Sequential(Linear(in_size1,self._latent_size),
                                   ReLU(),
                                   Linear(self._latent_size,self._latent_size),
                                   ReLU(),
                                   LayerNorm(self._latent_size))        
        
        # Second net (MLP): xi' = f2(xi, sum(eij'))
        self.node_net = Sequential(Linear(in_size2,self._latent_size),
                                   ReLU(),
                                   Linear(self._latent_size,self._latent_size),
                                   ReLU(),
                                   LayerNorm(self._latent_size))       
        
    def forward(self, graph): # mandatory with propagate
        
        edge_index = graph.edge_index
        # x = graph.node_features
        # edge_features = graph.edge_features  
        x = graph.x
        edge_features = graph.edge_attr
        # print(edge_index.size())
        # print(x.size())
        # print(edge_features.size())
        
        # Node update #optiona
        new_node_features = self.propagate(edge_index, x= x, edge_attr = edge_features)        
        
        # Edge update
        row, col = edge_index
        new_edge_features = self.edge_net(torch.cat([x[row], x[col], edge_features], dim=-1))
        
        # Add residuals
        new_node_features = new_node_features + graph.x
        new_edge_features = new_edge_features + graph.edge_attr       
        
        # return Graph(edge_index, new_node_features,new_edge_features)
        return Data(edge_index = edge_index, x = new_node_features, edge_attr = new_edge_features)        
    
    def message(self, x_i, x_j, edge_attr): # mandatory            
        features = torch.cat([x_i, x_j, edge_attr], dim=-1)        
        
        return self.edge_net(features)
    
    def update(self, aggr_out, x): # mandatory
        # aggr_out has shape [num_nodes, out_channels]        
        tmp = torch.cat([aggr_out, x], dim=-1)                
       
        # Step 5: Return new node embeddings.        
        return self.node_net(tmp)
    
class EncodeProcessDecode(torch.nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(self,
                 node_feat_size,                 
                 edge_feat_size,
                 output_size,
                 latent_size,                 
                 message_passing_steps,
                 window,
                 name='EncodeProcessDecode'):
        super(EncodeProcessDecode, self).__init__()      
        self._node_feat_size = node_feat_size        
        self._edge_feat_size = edge_feat_size
        self._latent_size = latent_size
        self._output_size = output_size      
        self._message_passing_steps = message_passing_steps    
        self._window = window
        
        # Encoding net (MLP) for node_features
        self.node_encode_net = Sequential(Linear(self._node_feat_size,self._latent_size),
                         ReLU(),
                         Linear(self._latent_size,self._latent_size),
                         ReLU(),
                         LayerNorm(self._latent_size))               
               
        # Encoding net (MLP) for edge_features
        self.edge_encode_net = Sequential(Linear(self._edge_feat_size,self._latent_size),
                         ReLU(),
                         Linear(self._latent_size,self._latent_size),
                         ReLU(),
                         LayerNorm(self._latent_size))      
       
        # Processor
        self.message_pass = GraphNetBlock(self._latent_size, self._latent_size*3, self._latent_size*2)
    
        
        # Recurrent net
        self.recurrent_1 = GRU(self._latent_size, self._latent_size, 1)                             
        self.recurrent_2 = GRU(self._latent_size, self._latent_size, 1)                             
       
        # Decoding net (MLP) for node_features (output)        
        self.node_decode_net = Sequential(Linear(self._latent_size, self._latent_size),
                        ReLU(),
                        Linear(self._latent_size,self._output_size))
        
        
    def forward(self, graph_sequence):
        # Initialize hidden states for recurrent layers
        h1 = torch.zeros(1, graph_sequence[0].x.size(0), self._latent_size).to(graph_sequence[0].x.device)
        h2 = torch.zeros(1, graph_sequence[0].x.size(0), self._latent_size).to(graph_sequence[0].x.device)
        
        outputs = []
        
        for t in range(len(graph_sequence)):
            graph = graph_sequence[t]
            edge_index = graph.edge_index
            x = graph.x
            edge = graph.edge_attr
            
            # Encode node and edge features
            node_latents = self.node_encode_net(x)
            edge_latents = self.edge_encode_net(edge)
            
            # Message passing (node and edge features are updated)
            latent_graph = Data(edge_index=edge_index, x=node_latents, edge_attr=edge_latents)
            for _ in range(self._message_passing_steps):
                latent_graph = self.message_pass(latent_graph)
            
            # Recurrent update: combine current features with hidden state from previous time steps
            node_latents, h1 = self.recurrent_1(latent_graph.x.unsqueeze(0), h1)  # First GRU layer
            node_latents, h2 = self.recurrent_2(node_latents, h2)  # Second GRU layer
            
            node_latents = node_latents.squeeze(0)  # Remove batch dimension after GRU
            
            # Decode node features to get the output
            decoded_nodes = self.node_decode_net(node_latents)
            outputs.append(decoded_nodes)  # Collect the output for the current time step
        
        # Return outputs for all time steps in the sequence
        return torch.stack(outputs, dim=0)

  



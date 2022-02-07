import tensorflow as tf
import numpy as np
import time
import sys
from typing import List, Optional
from typing import Callable, List
from configs import ParseParams
from DataGenerator import DataGenerator
sys.path.append("../")

from shared.embeddings import Embedding
from shared.graph_embedding.useful_files.utils import get_activation
from sklearn.preprocessing import normalize
from shared.graph_embedding.useful_files.utils import SMALL_NUMBER

def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.keras.layers.LayerNormalization(name=name,axis=-1,epsilon=1e-12,dtype=tf.float32)(input_tensor)

def get_aggregation_function(aggregation_fun: Optional[str]):
    if aggregation_fun in ['sum', 'unsorted_segment_sum']:
        return tf.math.unsorted_segment_sum
    if aggregation_fun in ['max', 'unsorted_segment_max']:
        return tf.math.unsorted_segment_max
    if aggregation_fun in ['mean', 'unsorted_segment_mean']:
        return tf.math.unsorted_segment_mean
    if aggregation_fun in ['sqrt_n', 'unsorted_segment_sqrt_n']:
        return tf.math.unsorted_segment_sqrt_n
    else:
        raise ValueError("Unknown aggregation function '%s'!" % aggregation_fun)
        
def sparse_gnn_film_layer(node_embeddings: tf.Tensor,
                          adjacency_lists: List[tf.Tensor],
                          type_to_num_incoming_edges: tf.Tensor,
                          state_dim: Optional[int],
                          num_timesteps: int = 1,
                          activation_function: Optional[str] = "ReLU",
                          message_aggregation_function: str = "sum",
                          normalize_by_num_incoming: bool = False,
                          ) -> tf.Tensor:
    """
    Compute new graph states by neural message passing modulated by the target state.
    For this, we assume existing node states h^t_v and a list of per-edge-type adjacency
    matrices A_\ell.

    We compute new states as follows:
        h^{t+1}_v := \sum_\ell
                     \sum_{(u, v) \in A_\ell}
                        \sigma(1/c_{v,\ell} * \alpha_{\ell,v} * (W_\ell * h^t_u) + \beta_{\ell,v})
        \alpha_{\ell,v} := F_{\ell,\alpha} * h^t_v
        \beta_{\ell,v} := F_{\ell,\beta} * h^t_v
        c_{\v,\ell} is usually 1 (but could also be the number of incoming edges).
    The learnable parameters of this are the W_\ell, F_{\ell,\alpha}, F_{\ell,\beta} \in R^{D, D}.

    We use the following abbreviations in shape descriptions:
    * V: number of nodes
    * D: state dimension
    * L: number of different edge types
    * E: number of edges of a given edge type

    Arguments:
        node_embeddings: float32 tensor of shape [V, D], the original representation of
            each node in the graph.
        adjacency_lists: List of L adjacency lists, represented as int32 tensors of shape
            [E, 2]. Concretely, adjacency_lists[l][k,:] == [v, u] means that the k-th edge
            of type l connects node v to node u.
        type_to_num_incoming_edges: float32 tensor of shape [L, V] representing the number
            of incoming edges of a given type. Concretely, type_to_num_incoming_edges[l, v]
            is the number of edge of type l connecting to node v.
        state_dim: Optional size of output dimension of the GNN layer. If not set, defaults
            to D, the dimensionality of the input. If different from the input dimension,
            parameter num_timesteps has to be 1.
        num_timesteps: Number of repeated applications of this message passing layer.
        activation_function: Type of activation function used.
        message_aggregation_function: Type of aggregation function used for messages.
        normalize_by_num_incoming: Flag indicating if messages should be scaled by 1/(number
            of incoming edges).

    Returns:
        float32 tensor of shape [V, state_dim]
    """
    num_nodes = tf.shape(input=node_embeddings, out_type=tf.int32)[0]
    if state_dim is None:
        state_dim = tf.shape(input=node_embeddings, out_type=tf.int32)[1]

    # === Prepare things we need across all timesteps:
    activation_fn = get_activation(activation_function)
    message_aggregation_fn = get_aggregation_function(message_aggregation_function)
    edge_type_to_message_transformation_layers = []  # Layers to compute the message from a source state
    edge_type_to_film_computation_layers = []  # Layers to compute the \beta/\gamma weights for FiLM
    edge_type_to_message_targets = []  # List of tensors of message targets
    for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
        edge_type_to_message_transformation_layers.append(
            tf.keras.layers.Dense(units=state_dim,
                                  use_bias=False,
                                  activation=None,  # Activation only after FiLM modulation
                                  name="Edge_%i_Weight" % edge_type_idx))
        edge_type_to_film_computation_layers.append(
            tf.keras.layers.Dense(units=2 * state_dim,  # Computes \gamma, \beta in one go
                                  use_bias=False,
                                  activation=None,
                                  name="Edge_%i_FiLM_Computations" % edge_type_idx))
        edge_type_to_message_targets.append(tf.cast(adjacency_list_for_edge_type[:, 1],dtype=tf.int32))

    # Let M be the number of messages (sum of all E):
    message_targets = tf.concat(edge_type_to_message_targets, axis=0)  # Shape [M]

    cur_node_states = node_embeddings
    for _ in range(num_timesteps):
        messages_per_type = []  # list of tensors of messages of shape [E, D]
        # Collect incoming messages per edge type
        for edge_type_idx, adjacency_list_for_edge_type in enumerate(adjacency_lists):
            edge_sources = tf.cast(adjacency_list_for_edge_type[:, 0],dtype=tf.int32)
            edge_targets = tf.cast(adjacency_list_for_edge_type[:, 1],dtype=tf.int32)
            edge_source_states = \
                tf.nn.embedding_lookup(params=cur_node_states,
                                       ids=edge_sources)  # Shape [E, D]
            #embedding_lookup:  if params is a tensor like [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], 
            #and ids is: [0, 3, 4],  then return value would be [[1, 2], [7, 8], [9, 10]]
            messages = edge_type_to_message_transformation_layers[edge_type_idx](edge_source_states)  # Shape [E, D]
            
            #print( type_to_num_incoming_edges[edge_type_idx, :].numpy() )
            #print( edge_targets.numpy() )
            if normalize_by_num_incoming:
                per_message_num_incoming_edges = \
                    tf.nn.embedding_lookup(params=type_to_num_incoming_edges[edge_type_idx, :],
                                           ids=edge_targets)  # Shape [E, H]

                messages = tf.expand_dims(1.0 / (per_message_num_incoming_edges + SMALL_NUMBER), axis=-1) * messages

            film_weights = edge_type_to_film_computation_layers[edge_type_idx](cur_node_states)
            per_message_film_weights = \
                tf.nn.embedding_lookup(params=film_weights, ids=edge_targets)
            per_message_film_gamma_weights = per_message_film_weights[:, :state_dim]  # Shape [E, D]
            per_message_film_beta_weights = per_message_film_weights[:, state_dim:]  # Shape [E, D]

            modulated_messages = per_message_film_gamma_weights * messages + per_message_film_beta_weights
            messages_per_type.append(modulated_messages)

        all_messages = tf.concat(messages_per_type, axis=0)  # Shape [M, D]
        all_messages = activation_fn(all_messages)  # Shape [M, D]
        aggregated_messages = \
            message_aggregation_fn(data=all_messages,
                                   segment_ids=message_targets,
                                   num_segments=num_nodes)  # Shape [V, D]
        new_node_states = aggregated_messages
        # new_node_states = activation_fn(new_node_states)

        cur_node_states = layer_norm(new_node_states)

    return cur_node_states        

class FullGraphEmbedding(Embedding):
    """
    Implements a graph embedding, not test
    """
    def __init__(self,embedding_dim,args):
        assert args['embedding_dim'] == 30, args['embedding_dim']
        super(FullGraphEmbedding,self).__init__('full_graph',embedding_dim)

        self.nb_feat = args['input_dim']
        self.n_nodes = args['n_nodes']

        self._scale = [5,12,25,50,100]
        self._scale = [i * np.sqrt(2)/100 for i in self._scale]     # rescale to the square

        self.drop_out = tf.Variable( 1.0, dtype=tf.float32)
        #self.drop_out = tf.compat.v1.placeholder(tf.float32,name='embedder_graph_dropout')
        self.params = {
            'graph_num_layers': 8,
            'graph_num_timesteps_per_layer': 3,

            'graph_layer_input_dropout_keep_prob': 0.8,
            'graph_dense_between_every_num_gnn_layers': 1,
            'graph_model_activation_function': 'tanh',
            'graph_residual_connection_every_num_layers': 1,
            'graph_inter_layer_norm': False,
            "hidden_size": 30,
            "graph_activation_function": "ReLU",
            "message_aggregation_function": "sum",
            "normalize_messages_by_num_incoming": True
            }

    def _propagate_graph_model(self,initial_node_features, incoming_edge, list_pair_adjancy):
        """
        Build the propagation model via graph
        :param initial_node_features:
        :param incoming_edge:
        :param list_pair_adjancy:
        :return:
        """
        h_dim= self.params['hidden_size']
        activation_fn = get_activation(self.params['graph_model_activation_function'])

        projected_node_features = tf.keras.layers.Dense(units=h_dim,
                                      use_bias=False,
                                      activation=activation_fn,
                                      )(initial_node_features)

        cur_node_representations = projected_node_features
        last_residual_representations = tf.zeros_like(cur_node_representations)
        for layer_idx in range(self.params['graph_num_layers']):
            # with tf.variable_scope('gnn_layer_%i' % layer_idx):
            cur_node_representations = \
                tf.nn.dropout(cur_node_representations, rate= 1- self.drop_out)
            if layer_idx % self.params['graph_residual_connection_every_num_layers'] == 0:
                t = cur_node_representations
                if layer_idx > 0:
                    cur_node_representations += last_residual_representations
                    cur_node_representations /= 2
                last_residual_representations = t
            cur_node_representations = \
                self._apply_gnn_layer(cur_node_representations,list_pair_adjancy,incoming_edge,self.params['graph_num_timesteps_per_layer'])
            if self.params['graph_inter_layer_norm']:
                cur_node_representations = tf.contrib.layers.layer_norm(cur_node_representations)
            if layer_idx % self.params['graph_dense_between_every_num_gnn_layers'] == 0:
                cur_node_representations = \
                    tf.keras.layers.Dense(units=h_dim,
                                          use_bias=False,
                                          activation=activation_fn,
                                          name="Dense",
                                          )(cur_node_representations)

        return cur_node_representations


    def _apply_gnn_layer(self,node_representations,adjacency_lists,type_to_num_incoming_edges,num_timesteps):
        """
        Apply the actual gnn layer
        """
        return sparse_gnn_film_layer(
            node_embeddings=node_representations,
            adjacency_lists=adjacency_lists,
            type_to_num_incoming_edges=type_to_num_incoming_edges,
            state_dim=self.params['hidden_size'],
            num_timesteps=num_timesteps,
            activation_function=self.params['graph_activation_function'],
            message_aggregation_function=self.params['message_aggregation_function'],
            normalize_by_num_incoming=self.params["normalize_messages_by_num_incoming"])


    def _prepare_input_data(self, input_tf):
        """
        Prepare the input data so that they are at the right size
        :param input_tf:
        :return:
        """
        #shape of input_tf is [None, 11, 3] which mean undetermined batches, 11 nodes,  andd 3 columne for each node to list the x, y coordinate and demand qty
        batch_features = tf.reshape(input_tf,[-1,self.nb_feat])
        #batch features are put the nodes infor together,  into shape [None, 3]
        input_dist = input_tf[:,:,:2]
        square_input = tf.reduce_sum(input_tensor=tf.square(input_dist), axis=2)
        row = tf.reshape(square_input, [-1,self.n_nodes,1])
        col= tf.reshape(square_input,[-1,1,self.n_nodes])
        t1 = 2 * tf.matmul(input_dist,input_dist,False,True)
        dist_matrix = tf.sqrt(tf.maximum(row - 2 * tf.matmul(input_dist,input_dist,False,True) + col,0.0))
        #shape of dist_matrix would be [?, self.n_nodes, self.n_nodes]
        # value is the distance between nodes,  coordinate of node Ni is (Xi1, Xi2), and node Nj is (Xj1, Xj2)
        #then the value in the maxtrix for position[ ?, i, j] would be sqrt( (Xi1 - Xj1 ) ^2 + (Xi2 - Xj2)^2 )
        # example dist_matrix: 
        # dist_matrix: 
        # tf.Tensor(
        #[[[ 0.         4.2426405  8.485281  12.7279215]
        #  [ 4.2426405  0.         4.2426405  8.485281 ]
        #  [ 8.485281   4.2426405  0.         4.2426405]
        #  [12.7279215  8.485281   4.2426405  0.       ]]
        # [[ 0.         4.2426405  8.485281  12.7279215]
        #  [ 4.2426405  0.         4.2426405  8.485281 ]
        #  [ 8.485281   4.2426405  0.         4.2426405]
        #  [12.7279215  8.485281   4.2426405  0.       ]]], shape=(2, 4, 4), 
        list_num_incoming_ege = []
        list_pair_edge = []
        # not_masked is a [?, self.n_nodes, self.n_nodes] shape boolean tensor, all intial values are true
        not_masked = tf.ones_like(dist_matrix,dtype=tf.bool)
        temp = tf.zeros_like(not_masked[0,:,:])
        #so set_diag will set the diagnal values to zeros, like:
        #  [ [0, 1, 1, 1],
        #    [1, 0, 1, 1],
        #    [1, 1, 0, 1],
        print( not_masked.shape)
        print( tf.zeros_like(not_masked[0,:,:]).shape )
        print( tf.zeros_like(not_masked[:,:,0]).numpy() )
        print( tf.zeros_like(not_masked[:,:,0]).shape )
        not_masked = tf.linalg.set_diag(not_masked,tf.zeros_like(not_masked[:,:,0])) #linalg_.set_diag won't change not_masked here,  shall change it to not_masked = tf.linalg.set_diag?
        
        #print( not_masked.numpy())
        for i in range(len(self._scale)):
            true_for_edge = tf.less_equal(dist_matrix,self._scale[i])
            true_for_edge = tf.logical_and(not_masked,true_for_edge)
            # continue above example , 
            # true for edge less than 5.0: 
            #tf.Tensor(
            #[[[ True,  True, False],
            #  [ False,  True,  True]],
            # [[ True,  True, False],
            # [ True,  False,  True]]], shape=(2, 2, 3), dtype=bool)
            
            # all values less than self._scale[i] are true,  and diagnal values are false
            # same shape as dist_matrix [?,n,n]
            print( true_for_edge.numpy())
            indices = tf.cast(tf.compat.v1.where(true_for_edge),dtype=tf.int32)
            print( indices.numpy() )
            #indices of  coordinates of all the edges in dist_matrix with value less than self._scale[i]
            # indices is like:  [[0, 0, 0],
            #[0, 0, 1],
            #[0, 1, 1],
            #[0, 1, 2],
            #[1, 0, 0],
            #[1, 0, 1],
            #[1, 1, 0],
            #[1, 1, 2]]
            #  notice that the batch dimension is removed,  the following steps will add offset for batch to the index
            offset = self.n_nodes * indices[:,0]    # indices' shape is [8,3], so this line will get all batch value
            #print( offset.numpy() )
             # offset of temp would be [0,0,0,0,11,11,11,11]
            offset = tf.expand_dims(offset,axis=1)
            #print( offset.numpy() )
            #after expending become: [[0],[0],[0],[0],[11],[11], [11],[11]]
            offset = tf.tile(offset,[1,2])
            #print( offset.numpy() )
            #after tiline become:  [[0,0], [0,0], [11,11],[22,22],[22,22],[22,22]]
            #indices[:,1:3] is the index of last two column,  so really the edge (from node to 'to node')
            true_indices_nodes = offset + indices[:,1:3]
            # so now actually true_indices_nodes is like embedding the batching dimension into the edge columns
            #[[ 0,  0],
           #[ 0,  1],
           #[ 1,  1],
           #[ 1,  2],
           #[11, 11],
           #[11, 12],
           #[12, 11],
           #[12, 13]]
            list_pair_edge.append(true_indices_nodes)

            num_incoming = tf.reduce_sum(input_tensor=tf.cast(true_for_edge,dtype=tf.int32), axis=1)
            # continue examples， num incoming, incoming edges to a node which is less than scale[i]
            #tf.Tensor(
            #[[1, 2, 1],
            #[2, 1, 1]], shape=(2, 3), dtype=int32)
            
            num_incoming = tf.squeeze(tf.reshape(num_incoming,[1,-1]),0)
            # reshaped: tf.Tensor([[1,2,1,2,1,1]], shape=(1, 6), dtype=int32)
            # squeezed: tf.Tensor([1,2,1,2,1,1], shape=(6,), dtype=int32)  
            list_num_incoming_ege.append(tf.cast(num_incoming,dtype=tf.float32))
            # list_num_incoming_ege is a list of 5 tensor
            # update the mask
            not_masked = tf.logical_and(not_masked,tf.logical_not(true_for_edge)) # we update the mask. The only one not masked are the one wich
                                                                                    # were not and did not belong to the edge type
            #print( not_masked.numpy())
        final_incoming_edge = tf.stack(list_num_incoming_ege)   #list_num_incoming_ege is a list of 5 tensors, tensor shape for examples are all (6,),
                                                                #  after stack, the will get a tensor with shape (5,6)

        #print( final_incoming_edge.numpy() )
        return batch_features, final_incoming_edge, list_pair_edge

    def __call__(self, input_tf):
        """
        return the node embedding
        :param input_tf: the tensor corresponding to the embedding
        :return: a tensor
        """
        time_init = time.time()
        initial_node_features, incoming_edge, list_pair_adjancy = self._prepare_input_data(input_tf)

        final_node_representations = self._propagate_graph_model(initial_node_features,incoming_edge,list_pair_adjancy)
        final_node_representations = tf.reshape(final_node_representations,[-1,self.n_nodes,self.embedding_dim])

        self.total_time += time.time() - time_init

        return final_node_representations
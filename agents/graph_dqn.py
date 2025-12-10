import torch.nn as nn
import torch

class GraphDQN(nn.Module):
    def __init__(self, config : dict, **kwargs):
        super().__init__()

        self.vision_module = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        test_input = torch.zeros((1, 3, 64, 64))
        test_output = self.vision_module(test_input)
        
        self.mem_size = config.get('mem_size', 128)
        self.mem_features = config.get('mem_features', 8)
        self.batch_size = config.get('batch_size', 32)
        self.device = kwargs['device']
        self.k_retrieve = config.get('k_retrieve', 4)
        
        self.mem_sequence_dim = self.mem_features + 6
        
        # initialize memory graph components
        self.mem_graph_nodes = torch.zeros((self.batch_size, self.mem_size, self.mem_features), device=self.device)
        """Tensor of Shape (B, mem_size, mem_features)"""
        self.mem_graph_edges = torch.full((self.batch_size, self.mem_size, self.mem_size, 6), float('inf'), device=self.device) # 6 action types
        """Tensor of shape (B, mem_size, mem_size, 6)"""
        self.mem_similarity = torch.full((self.batch_size, self.mem_size, self.mem_size), float('inf'), device=self.device) # store similarity thresholds for each memory slot
        """(B, mem_size, mem_size)"""
        
        self.vision_encoder = nn.Linear(test_output.shape[1], self.mem_features)
        
        # small model to learn how to best map target color to a memory
        self.target_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, self.mem_sequence_dim)
        )
        
        # attention memory
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.mem_sequence_dim,
            nhead=config.get('num_heads', 1),
            dim_feedforward=self.mem_sequence_dim*2,
            batch_first=True
        )
        
        self.memory_transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.get('transformer_layers', 1))
        
        self.prev_states = torch.zeros((self.batch_size, self.mem_features), device=self.device)-1
        self.prev_actions = torch.zeros((self.batch_size, 6), device=self.device)

        self.head = nn.Sequential(
            nn.Linear(self.mem_sequence_dim*2, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        if 'pretrained_weights' in kwargs:
            try:
                self.load_state_dict(kwargs['pretrained_weights'], strict=False)
            except RuntimeError as e:
                # Only load vision
                pretrained_dict = kwargs['pretrained_weights']
                model_dict = self.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('vision_module')}
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict, strict=False)

    def reset_memory(self):
        '''Reset the memory to zeros. Should be called at the start of each episode.'''
        self.mem_graph_nodes = torch.zeros_like(self.mem_graph_nodes)
        self.mem_graph_edges = torch.full((self.batch_size, self.mem_size, self.mem_size, 6), float('inf'), device=self.device)
        return
    
    def add_mem_state(self, obs_enc, act_enc, prev_index):
        """Adds a new observation to the memory graph. If there is no space, the most similar states
        are merged to make space. Memory graph is shaped (B, S, F)
        Args:
            new_obs (Torch.Tensor): Image Tensor of shape (B, C, H, W) with values in [0, 1]
            prev_index (Torch.Tensor): Tensor of shape (B,) indicating the index of the previous memory state
        """        
        # check similarity to existing states
        min_sim_new = torch.cdist(obs_enc.unsqueeze(1), self.mem_graph_nodes, p=2).squeeze(1)  # shape (B, S)
        min_sim_new = torch.min(min_sim_new, dim=1).values
        
        min_sim_existing = torch.min(self.mem_similarity, dim=-1).values
        min_sim_existing = torch.min(min_sim_existing, dim=-1).values
        
        update_mask = min_sim_new > min_sim_existing # only update if our new observation is less similar than existing
        
        if not update_mask.any():
            #no updates needed
            return
        
        # this will filter our batch size to just the elements that are changing (call it 'k')
        update_batch_idx = torch.nonzero(update_mask).squeeze(1)
        
        obs_enc = obs_enc[update_batch_idx] # now (k, F)
        act_enc = act_enc[update_batch_idx] # now (k, 6)
        prev_index = prev_index[update_batch_idx] # now (k,)
        mem_sim = self.mem_similarity[update_batch_idx] # shape (k, mem_size, mem_size)
        
        # flatten to find the absolute mins in the grid
        flat_sim = mem_sim.view(len(update_batch_idx), -1)
        min_vals, flat_idx = torch.min(flat_sim, dim=1)
        
        merge_idx = flat_idx // self.mem_size
        remove_idx = flat_idx % self.mem_size
        
        # get outgoing edges from remove and add to merge if shorter than merge's
        merge_edges = self.mem_graph_edges[update_batch_idx, merge_idx, :, :]
        remove_edges = self.mem_graph_edges[update_batch_idx, remove_idx, :, :]
        self.mem_graph_edges[update_batch_idx, merge_idx, :, :] = torch.min(merge_edges, remove_edges)
        
        # Find incoming edges and make sure they point to merge now
        merge_edges = self.mem_graph_edges[update_batch_idx, :, merge_idx, :]
        remove_edges = self.mem_graph_edges[update_batch_idx, :, remove_idx, :]
        self.mem_graph_edges[update_batch_idx, :, merge_idx, :] = torch.min(merge_edges, remove_edges)
        
        # set remove node to new node values
        self.mem_graph_nodes[update_batch_idx, remove_idx] = obs_enc
        
        #clear old edges
        self.mem_graph_edges[update_batch_idx, remove_idx, :, :] = float('inf') #outgoing
        self.mem_graph_edges[update_batch_idx, :, remove_idx, :] = float('inf') #incoming
        
        # add edge from prev_index to remove node of value act_enc, if prev_index was remove node, use merge instead
        prev_index = torch.where(prev_index == remove_idx, merge_idx, prev_index)
        self.mem_graph_edges[update_batch_idx, prev_index, remove_idx] = act_enc
        
        # update mem_similarity
        self.mem_similarity = torch.cdist(self.mem_graph_nodes, self.mem_graph_nodes, p=2)  # shape (B, S, S)
        diag_mask = torch.tril(torch.ones_like(self.mem_similarity), diagonal=0)==1
        self.mem_similarity[diag_mask] = float('inf')
            
        return
    
    def store_actions(self, actions):
        """Allows a trainer to override this model's previous actions if they were instead
        chosen by some external means for exploration

        Args:
            actions (torch.Tensor): batch of action indices (B, 1) or one-hot (B, 6)
        """
        if actions.dim() > 1 and actions.shape[1] > 1:
             # Assume logits or one-hot, take argmax
             self.prev_actions = actions.argmax(dim=1, keepdim=True)
        else:
             self.prev_actions = actions.view(-1, 1)
    
    def get_k_shortest_paths(self, cost_matrix, source_indices):
        """
        Computes Single Source Shortest Path (SSSP) for a batch of sources 
        using Bellman-Ford-like relaxation over a fixed number of steps (max_hops_sssp).
        
        Args:
            cost_matrix (B, S, S): Matrix of direct transition costs (min over all actions).
            source_indices (B,): Index of the starting node (i*) for each batch element.
        Returns:
            (B, S): Distance matrix D from source_indices to all other nodes.
        """
        B = cost_matrix.size(0)  # Get batch size from input
        S = self.mem_size
        
        # 1. Initialize distances from the source to all nodes
        # dist[b, j] = cost_matrix[b, source_indices[b], j]
        distances = cost_matrix[torch.arange(B, device=self.device), source_indices, :] # (B, S)
        
        # Set distance from source to itself to 0
        distances[torch.arange(B, device=self.device), source_indices] = 0.0

        # 2. Relax edges max_hops_sssp times (Bounded Bellman-Ford)
        # Bellman-Ford update: D[j] = min(D[j], D[i] + C[i, j]) for all edges (i, j)
        for _ in range(S - 1):
            # D_i (B, S, 1) - current best distance from source to all intermediate nodes i
            D_i = distances.unsqueeze(2) 

            # path_through_i (B, S, S) - path cost through intermediate node i to j: D[i] + C[i, j]
            path_through_i = D_i + cost_matrix 
            
            # Find the minimum distance to j over all possible intermediate nodes i
            # new_distances[b, j] = min_i (D[i] + C[i, j])
            new_distances, _ = torch.min(path_through_i, dim=1) 

            # Update D[j] = min(D[j], new_distances)
            distances = torch.min(distances, new_distances)
            
        return distances

    def forward(self, x):
        """Forward pass of the network.
        Args:
            x (Torch.Tensor): Image Tensor of shape (B, C, H, W) with values in [0, 1]

        Returns:
            _type_: output Q-values Tensor of shape (B, num_actions)
        """
        #extract the current batch size
        current_batch_size = x.shape[0]
        
        # extract the 0,0 pixel from each image in the batch as the target color
        target_color = x[:, :, 0, 0] # shape (B, C)
        target_enc = self.target_encoder(target_color).unsqueeze(1)
        
        #center the color values around 0 for the cnn
        x = x - 0.5
        x = self.vision_module(x) # shape (B, F)
        vision_enc = self.vision_encoder(x)
        
        # retrieve k_retrieve sequences from memory based on similarity
        nodes = self.mem_graph_nodes[:current_batch_size]
        distances = torch.cdist(vision_enc.unsqueeze(1), nodes, p=2).squeeze(1)
        closest = torch.argmin(distances, dim=1)
        
        # compress the actions taken to a single value for shortest path
        edges = self.mem_graph_edges[:current_batch_size]
        cost_matrix = torch.min(edges, dim=-1).values #(B, mem_size, mem_size)
        
        spd_distances = self.get_k_shortest_paths(cost_matrix, closest)
        # replace inf with a large number to allow proper sorting when reversing the distances
        spd_distances = torch.where(spd_distances == float('inf'), torch.tensor(1e9, device=self.device), spd_distances)
        
        # get the k nearest neighbors of the closest state (by action path)
        _, topk_indices = torch.topk(-spd_distances, self.k_retrieve, dim=1) #(B, k)
        
        # get min outgoing cost for sequences
        min_outgoing_cost, _ = torch.min(edges, dim=2)
        
        node_indices = topk_indices.unsqueeze(-1).expand(-1, -1, self.mem_features)
        edge_indices = topk_indices.unsqueeze(-1).expand(-1, -1, 6)
        
        retrieved_nodes = torch.gather(nodes, 1, node_indices)
        retrieved_edges = torch.gather(min_outgoing_cost, 1, edge_indices)
        
        retrieved_memory = torch.cat((retrieved_nodes, retrieved_edges), dim=-1)
        
        #add target to sequence to allow transformer to reason about it as well
        full_sequence = torch.cat((target_enc, retrieved_memory), dim=1)
        
        encoded_sequence = self.memory_transformer(full_sequence)
        
        target_context = encoded_sequence[:, 0, :]
        memory_context = torch.mean(encoded_sequence[:, 1:, :], dim=1)
        
        head_input = torch.cat((target_context, memory_context), dim=1)
        
        x = self.head(head_input)
        
        self.add_mem_state(vision_enc, self.prev_actions, self.prev_states)
        
        self.prev_actions = x.argmax(dim=1, keepdim=True)
        self.prev_states = closest
        
        return x

__all__ = ["GraphDQN"]

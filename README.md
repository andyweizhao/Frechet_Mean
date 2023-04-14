# Frechet_Mean

Implementation of the Fréchet mean (Karcher mean) to be used as a building block in a graph convolutional layer on an spd manifold.

Todos: the forward direction occasionally does not work as it should: sometimes the iteration gets stuck. As of now, the
error is simply ignored when it occurs.



in ECML/layers/gcn replace 

        symmat_feats = SPDManifold.logmap_id(mat_feats)

        symmat_feats = symmat_feats.reshape(num_nodes, -1)

        out = self.propagate(edge_index, x=symmat_feats, edge_weight=edge_weight, size=None)

        out = out.reshape(num_nodes, n, n)
        
        out = SPDManifold.expmap_id(out)
by 

        out = frechet_agg(to_coordinates(mat_feats), edge_index, edge_weight)
        out = to_matrix(out)
        

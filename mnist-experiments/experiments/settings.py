parameters = {"num_images_raw": 2559,
                      "selection_random_seed": 0,
                      "num_pca_dimensions": 30,
                      "pca_random_seed": 0,
                      'tsne_momentum': 0.8,
                      'tsne_n_iters': 3000,
                      'tsne_early_exaggeration_iters': 300,
                      'tsne_perplexity': 30,
                      'tsne_random_seed': 1,
                      'neighbor_indices_to_pick': 1000,
                      'neighbor_picking_random_seed': 10,
                      'accuracy_nn' : 10}

raw_parameter_set = {"selection_random_seed", "num_images_raw"}
pca_parameter_set = raw_parameter_set | {"num_pca_dimensions", "pca_random_seed"}
tsne_parameter_set = pca_parameter_set | {"tsne_random_seed", "tsne_perplexity", "tsne_momentum",
                                          "tsne_n_iters", "tsne_early_exaggeration_iters"}
x_neighbors_selection_parameter_set = pca_parameter_set | {"neighbor_indices_to_pick", "neighbor_picking_random_seed"}
nn_accuracy_parameter_set = tsne_parameter_set | x_neighbors_selection_parameter_set | {"accuracy_nn"}

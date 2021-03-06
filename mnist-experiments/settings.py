parameters = {
    #"num_images_raw": 2559,
    #"selection_random_seed": 0,
    "old": True,
    "num_pca_dimensions": 784,  # 30, # 784 is the number of original dimensions. In that case PCA won't be used
    "pca_random_seed": 'full',
    "tsne_momentum": 0.8,
    "tsne_n_iters": 3000,
    "tsne_early_exaggeration_iters": 300,
    "tsne_perplexity": 50, # 30, # 50 fits better for non-PCA, 30 is for 30-dim PCA, but is also OK for non-PCA
    "tsne_random_seed": 1,
    "neighbor_indices_to_pick": 1000,
    "neighbor_picking_random_seed": 10,
    "accuracy_nn" : 10,
    "precision_nn" : 50,  # As perplexity, which is essentially a number of nearest neighbors
    "outlier_indices_to_pick" : 1000,
    "outlier_random_seed" : 23412,
    "keras_random_seed" : 123, # Anything will do, just keep it consistent
    "letter_random_seed" : 0,
    "letter_indices_to_pick" : 1000,
    "letter_A_random_seed": 0,
    "letter_A_indices_to_pick": 100,
    }

raw_parameter_set = {"selection_random_seed", "num_images_raw", "old"}
pca_parameter_set = raw_parameter_set | {"num_pca_dimensions", "pca_random_seed"}
outlier_parameter_set = pca_parameter_set | {"outlier_indices_to_pick", "outlier_random_seed"}
letter_parameter_set = pca_parameter_set | {"letter_indices_to_pick", "letter_random_seed"}
letter_A_parameter_set = pca_parameter_set | {"letter_A_indices_to_pick", "letter_A_random_seed"}
tsne_parameter_set = pca_parameter_set | {"tsne_random_seed", "tsne_perplexity", "tsne_momentum",
                                          "tsne_n_iters", "tsne_early_exaggeration_iters"}
x_neighbors_selection_parameter_set = pca_parameter_set | {"neighbor_indices_to_pick", "neighbor_picking_random_seed"}
nn_accuracy_parameter_set = tsne_parameter_set | x_neighbors_selection_parameter_set | {"accuracy_nn"}

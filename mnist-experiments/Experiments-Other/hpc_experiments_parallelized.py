"""
Splits some experiments into smaller chunks that can be run in parallel.
"""

# TODO

# Baseline accuracy: sometimes the point is in the mixed cluster, and K nearest neighbors of it belong to different clusters
# themselves. Accuracy 100% would not be achieved that way.

# TODO Should we call our metric accuracy at all?
# Should we count sample itself, when calculating baseline accuracy? On one hand, no - training sample is obviously its own NN.
# On another hand, we are looking for accuracy upper bound. And it is not training sample itself we are avaluating, it is
# "in upper bound case we would have had embedding very close to ..." etc.
# If we say "keep training sample", our target accuracy will be higher.
# So let's make sure that errors are not in our favor.
# Actaully, see the paper.
# per_sample_accuracy = np.zeros((len(picked_neighbors),))
# for i in range(len(picked_neighbors)):
#    expected_label = picked_neighbors_labels[i]
#    nn_indices = get_nearest_neighbors_in_y(picked_indices_y_mnist[i, :], n=accuracy_nn)
#    obtained_labels = labels_mnist[nn_indices]
#    per_sample_accuracy[i] = sum(obtained_labels == expected_label) / len(obtained_labels)
# baseline_accuracy = np.mean(per_sample_accuracy)
# print("Baseline Accuracy: ", baseline_accuracy)
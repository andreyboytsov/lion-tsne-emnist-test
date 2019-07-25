import pickle

with open('ptsne_results.p', 'rb') as f:
    ptsne_accuracy, ptsne_precision, _, _  = pickle.load(f)

print(ptsne_accuracy, ptsne_precision)
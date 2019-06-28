# TODO just starting

regenerate_kernelized_tsne_parameters = False

if os.path.isfile('kernelized_tsne_parameters_cache.p') and not regenerate_kernelized_tsne_parameters:
    with open('kernelized_tsne_parameters_cache.p', 'rb') as f:
        kernelized_tsne_parameters_cache = pickle.load(f)
else:
    kernelized_tsne_parameters_cache = dict()
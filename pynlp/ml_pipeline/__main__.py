from ml_pipeline import experiment
import argparse

datasets = ['data/gibert/', 'data/OLID_full/levelA/' ]
experiments = ['naive_bayes_counts','svm_libsvc_counts','cnn_raw']

result = dict.fromkeys(datasets) 
for dataset in datasets:
    result[dataset] = dict.fromkeys(experiments)
    for def_ in experiments:
        parser = argparse.ArgumentParser(description='run classifier on data')
        parser.add_argument('--task', dest='task', default="vua_format")
        parser.add_argument('--data_dir', dest='data_dir', default=dataset)
        parser.add_argument('--print_predictions', dest='print_predictions', default=False)
        parser.add_argument('--pipeline', dest='pipeline', default=def_)
        args = parser.parse_args()
        result[dataset][def_] = experiment.run(args.task, args.data_dir, args.pipeline, args.print_predictions)


# these are the options (see experiments.pipeline(name)). 
#     'naive_bayes_counts'
#     'naive_bayes_tdidf'
#     'svm_libsvc_counts'
#     'svm_libsvc_tfidf'
#     'svm_libsvc_embed' (word embeddings needed)
#     'svm_sigmoid_embed' (word embeddings needed)
#     'cnn_raw'
#     'cnn_prep'
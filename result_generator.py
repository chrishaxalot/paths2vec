from sklearn import svm
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from helpers.pipeline_evaluator import PipelineEvaluator

import gensim


cpu_count = 16
num_runs = 3
sample_size = 100
window_in_nodes = 2

# ogbg-molhiv, 41127 1 binary classification
# ogbg-molpcba, 437929 128 binary classification
# ogbg-moltox21, 7831 12 binary classification
# ogbg-molbace, 1513 1 binary classification
# ogbg-molbbbp, 2039 1 binary classification
# ogbg-molclintox, 1477 2 binary classification
# ogbg-molmuv, 93087 17 binary classification
# ogbg-molsider, 1427 27 binary classification
# ogbg-moltoxcast, 8576 617 binary classification
# ogbg-molesol, 1128 1 regression
# ogbg-molfreesolv, 642 1 regression
# ogbg-mollipo, 4200 1 regression

dataset_estimator_dict = {
    # (multi-task) binary classification
    "ogbg-molhiv": MultiOutputClassifier(svm.SVC()),
    # "ogbg-molpcba": MultiOutputClassifier(svm.SVC()),
    # "ogbg-moltox21": MultiOutputClassifier(svm.SVC()),
    # "ogbg-molbace": MultiOutputClassifier(svm.SVC()),
    # "ogbg-molbbbp": MultiOutputClassifier(svm.SVC()),
    # "ogbg-molclintox": MultiOutputClassifier(svm.SVC()),
    # "ogbg-molmuv": MultiOutputClassifier(svm.SVC()),
    # "ogbg-molsider": MultiOutputClassifier(svm.SVC()),
    # "ogbg-moltoxcast": MultiOutputClassifier(svm.SVC()),
    # regression
    # "ogbg-molesol": MultiOutputRegressor(svm.SVR()),
    # "ogbg-molfreesolv": MultiOutputRegressor(svm.SVR()),
    # "ogbg-mollipo": MultiOutputRegressor(svm.SVR()),
}

if __name__ == "__main__":
    print(f"are you using fast gensim?: {gensim.models.doc2vec.FAST_VERSION > -1}")

    pipeline_evaluator = PipelineEvaluator(
        cpu_count=cpu_count,
        num_runs=num_runs,
        window_in_nodes=window_in_nodes,
        sample_size=sample_size,
        vertex_feature_idx=range(9),  # [0],
        edge_feature_idx=range(3),  # [0],
    )

    for dataset_name, estimator in dataset_estimator_dict.items():
        pipeline_evaluator.evaluate(dataset_name=dataset_name, estimator=estimator)

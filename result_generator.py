from sklearn import svm
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from helpers.pipeline_evaluator import PipelineEvaluator

import gensim


cpu_count = 40
num_runs = 3
sample_size = 150
window_in_nodes = 3

dataset_estimator_dict = {
    "ogbg-molfreesolv": MultiOutputRegressor(svm.SVR()),  # 642
    "ogbg-molesol": MultiOutputRegressor(svm.SVR()),  # 1128
    "ogbg-molsider": MultiOutputClassifier(svm.SVC()),  # 1427
    "ogbg-molbbbp": MultiOutputClassifier(svm.SVC()),  # 1477
    "ogbg-moltox21": MultiOutputClassifier(svm.SVC()),  # 1513
    "ogbg-molbace": MultiOutputClassifier(svm.SVC()),  # 2039
    "ogbg-molmuv": MultiOutputClassifier(svm.SVC()),  # 7831
    "ogbg-mollipo": MultiOutputRegressor(svm.SVR()),  # 4200
    "ogbg-moltoxcast": MultiOutputClassifier(svm.SVC()),  # 8576
    "ogbg-molhiv": MultiOutputClassifier(svm.SVC()),  # 41127
    "ogbg-molclintox": MultiOutputClassifier(svm.SVC()),  # 93087
    "ogbg-molpcba": MultiOutputClassifier(svm.SVC()),  # 437929
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
        pipeline_evaluator.evaluate(
            dataset_name=dataset_name, estimator=estimator, max_elem=None
        )

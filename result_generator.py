from sklearn import svm
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from helpers.pipeline_evaluator import PipelineEvaluator

import gensim
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


cpu_count = 16
num_runs = 1  # 10
sample_size = 10  # 150
window_in_nodes = 1  # 3


dataset_estimator_dict = {
    "ogbg-molfreesolv": MultiOutputRegressor(svm.SVR()),  # <10_000
    "ogbg-molesol": MultiOutputRegressor(svm.SVR()),  # <10_000
    # "ogbg-molsider": MultiOutputClassifier(svm.SVC()),  # <10_000
    # "ogbg-molclintox": MultiOutputClassifier(svm.SVC()),  # <10_000
    "ogbg-molbace": MultiOutputClassifier(svm.SVC()),  # <10_000
    "ogbg-molbbbp": MultiOutputClassifier(svm.SVC()),  # <10_000
    "ogbg-mollipo": MultiOutputRegressor(svm.SVR()),  # <10_000
    # "ogbg-moltox21": MultiOutputClassifier(svm.SVC()),  # <10_000
    # "ogbg-moltoxcast": MultiOutputClassifier(svm.SVC()),  # <10_000
    # "ogbg-molhiv": MultiOutputClassifier(svm.SVC()), # <100_000
    # "ogbg-molmuv": MultiOutputClassifier(svm.SVC()), # <100_000
    # "ogbg-molpcba": MultiOutputClassifier(svm.SVC()), # >=100_00
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
            dataset_name=dataset_name,
            estimator=estimator,
            subset_name="test",
            result_dir="results/",
            max_elem=None,
        )

from sklearn import svm
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from helpers.pipeline_evaluator import PipelineEvaluator

cpu_count = 16
num_runs = 1
sample_size = 30
window_in_nodes = 3

dataset_estimator_dict = {
    # (multi-task) binary classification
    "ogbg-molhiv": MultiOutputClassifier(svm.SVC()),
    "ogbg-molpcba": MultiOutputClassifier(svm.SVC()),
    "ogbg-moltox21": MultiOutputClassifier(svm.SVC()),
    "ogbg-molbace": MultiOutputClassifier(svm.SVC()),
    "ogbg-molbbbp": MultiOutputClassifier(svm.SVC()),
    "ogbg-molclintox": MultiOutputClassifier(svm.SVC()),
    "ogbg-molmuv": MultiOutputClassifier(svm.SVC()),
    "ogbg-molsider": MultiOutputClassifier(svm.SVC()),
    "ogbg-moltoxcast": MultiOutputClassifier(svm.SVC()),
    # regression
    "ogbg-molesol": MultiOutputRegressor(svm.SVR()),
    "ogbg-molfreesolv": MultiOutputRegressor(svm.SVR()),
    "ogbg-mollipo": MultiOutputRegressor(svm.SVR()),
}

if __name__ == "__main__":
    pipeline_evaluator = PipelineEvaluator(
        cpu_count=cpu_count,
        num_runs=num_runs,
        window_in_nodes=window_in_nodes,
        sample_size=sample_size,
    )

    for dataset_name, estimator in dataset_estimator_dict.items():
        pipeline_evaluator.evaluate(dataset_name=dataset_name, estimator=estimator)

from sklearn import svm

from helpers.pipeline_evaluator import PipelineEvaluator

cpu_count = 16
num_runs = 3
sample_size = 100
window_in_nodes = 3

dataset_estimator_dict = {
    # "ogbg-molfreesolv": svm.SVR(),
    "ogbg-molsider": svm.SVC(),
    # "#ogbg-molfreesolv": svm.SVR(),
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

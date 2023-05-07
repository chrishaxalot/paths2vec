import click
from sklearn import svm
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from helpers.pipeline_evaluator import PipelineEvaluator


@click.command()
@click.option("--dataset_name")
@click.option("--task_type")
@click.option("--num_runs")
@click.option("--sample_size")
@click.option("--window_in_nodes")
def main(dataset_name, task_type, num_runs, sample_size, window_in_nodes):
    if task_type == "classification":
        estimator = MultiOutputClassifier(svm.SVC())
    elif task_type == "regression":
        estimator = MultiOutputRegressor(svm.SVR())

    pipeline_evaluator = PipelineEvaluator(
        cpu_count=40,
        num_runs=num_runs,
        window_in_nodes=window_in_nodes,
        sample_size=sample_size,
        vertex_feature_idx=range(9),
        edge_feature_idx=range(3),
    )

    pipeline_evaluator.evaluate(
        dataset_name=dataset_name, estimator=estimator, max_elem=None
    )


if __name__ == "__main__":
    main()

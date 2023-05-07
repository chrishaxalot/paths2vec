import click
from sklearn import svm
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from helpers.pipeline_evaluator import PipelineEvaluator


@click.command()
@click.option("--dataset_name")
@click.option("--task_type")
@click.option("--num_runs", type=int)
@click.option("--sample_size", type=int)
@click.option("--window_in_nodes", type=int)
@click.option("--max_elem", default=None)
def main(dataset_name, task_type, num_runs, sample_size, window_in_nodes, max_elem):
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
        dataset_name=dataset_name, estimator=estimator, max_elem=max_elem
    )


if __name__ == "__main__":
    main()

# python .\result_generator_batch.py --dataset_name=ogbg-molfreesolv --task_type=regression --num_runs=1 --sample_size=1 --window_in_nodes=1

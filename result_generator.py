import json

from helpers.pipeline_evaluator import PipelineEvaluator

cpu_count = 16
num_runs = 1
sample_size = 150
window_in_nodes = 3


dataset_estimator_dict = [
    "ogbg-molfreesolv",
    "ogbg-molesol",
    # "ogbg-molsider",
    # "ogbg-molclintox",
    "ogbg-molbace",
    "ogbg-molbbbp",
    "ogbg-mollipo"
    # "ogbg-moltox21",
    # "ogbg-moltoxcast",
    # "ogbg-molhiv",
    # "ogbg-molmuv",
    # "ogbg-molpcba"
]

if __name__ == "__main__":
    pipeline_evaluator = PipelineEvaluator(
        cpu_count=cpu_count,
        num_runs=num_runs,
        window_in_nodes=window_in_nodes,
        sample_size=sample_size,
        vertex_feature_idx=range(9),
        edge_feature_idx=range(3),
    )

    for dataset_name in dataset_estimator_dict:
        result_dict = pipeline_evaluator.evaluate(dataset_name=dataset_name)

        with open(f"results/{dataset_name}.json", "w") as fp:
            json.dump(result_dict, fp, indent=4)

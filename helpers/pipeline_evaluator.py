from ogb.graphproppred import Evaluator, GraphPropPredDataset
from sklearn import svm
import numpy as np

from paths2vec import Paths2Vec
from helpers import GraphGenerator, ResultPrinter


def get_paths2vec_X(dataset_name, dataset, cpu_count, sample_size, window_in_nodes):
    # convert ogb dicts to networkx graphs
    dict_calculator = GraphGenerator()
    graphs = dict_calculator.ogb_dataset_to_graphs(dataset=dataset)

    # generate vectors for graphs
    corpus_file = f"{dataset_name}_paths.cor"
    paths2vec = Paths2Vec(cpu_count=cpu_count)
    X = paths2vec.fit(
        graphs=graphs,
        corpus_file=corpus_file,
        sample_size=sample_size,
        window_in_nodes=window_in_nodes,
    )

    return X


def get_random_X(dataset_name, dataset, cpu_count, sample_size, window_in_nodes):
    X = [np.random.normal(size=100) for _ in range(len(dataset))]
    return X


class PipelineEvaluator:
    def __init__(self, cpu_count, num_runs, window_in_nodes, sample_size):
        self.cpu_count = cpu_count
        self.num_runs = num_runs
        self.window_in_nodes = window_in_nodes
        self.sample_size = sample_size
        pass

    def get_result_dicts(
        self,
        X_func,
        dataset_name,
        estimator,
    ):
        result_dicts = []

        for i in range(self.num_runs):
            print(f"starting run {i + 1} of {self.num_runs}")

            dataset = GraphPropPredDataset(name=dataset_name)

            X = X_func(
                dataset_name,
                dataset,
                self.cpu_count,
                self.sample_size,
                self.window_in_nodes,
            )

            # split data
            data = dict()
            for name, idx_list in dataset.get_idx_split().items():
                data[name] = dict()
                data[name]["X"] = np.array([X[idx] for idx in idx_list])
                data[name]["y"] = np.array(
                    [dataset[idx][1] for idx in idx_list]
                ).ravel()

            # fit
            estimator.fit(data["train"]["X"], data["train"]["y"])

            # predict
            data["valid"]["y_predicted"] = estimator.predict(data["valid"]["X"])

            # evaluate
            evaluator = Evaluator(name=dataset_name)
            input_dict = {
                "y_true": data["valid"]["y"].reshape((-1, 1)),
                "y_pred": data["valid"]["y_predicted"].reshape((-1, 1)),
            }

            result_dicts.append(evaluator.eval(input_dict))

            # newline for space in log file
            print()

        return result_dicts

    def evaluate(self, dataset_name, estimator):
        methods = {"path2vec": get_paths2vec_X, "random": get_random_X}

        for methodname, method in methods.items():
            result_dicts = self.get_result_dicts(
                method,
                dataset_name,
                estimator,
            )

            # print results
            print(f"dataset: {dataset_name}")
            print(f"method: {methodname}")
            print(f"runs: {self.num_runs}")
            result_printer = ResultPrinter()
            result_printer.print(result_dicts=result_dicts)

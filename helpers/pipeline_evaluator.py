import numpy as np
from ogb.graphproppred import Evaluator, GraphPropPredDataset
from sklearn.impute import SimpleImputer

from helpers import GraphGenerator, ResultPrinter
from paths2vec import Paths2Vec
import random
import time


def get_paths2vec_X(
    dataset_name,
    graphs,
    cpu_count,
    sample_size,
    window_in_nodes,
    vertex_feature_idx,
    edge_feature_idx,
    split_idx,
):
    # generate vectors for graphs
    corpus_file = f"{dataset_name}_paths.cor"
    paths2vec = Paths2Vec(cpu_count=cpu_count)
    X = paths2vec.fit(
        graphs=graphs,
        corpus_file=corpus_file,
        sample_size=sample_size,
        window_in_nodes=window_in_nodes,
        vertex_feature_idx=vertex_feature_idx,
        edge_feature_idx=edge_feature_idx,
    )

    return X


def get_random_X(
    dataset_name,
    graphs,
    cpu_count,
    sample_size,
    window_in_nodes,
    vertex_feature_idx,
    edge_feature_idx,
    split_idx,
):
    X = [np.random.normal(size=100) for _ in range(len(graphs))]
    return X


class PipelineEvaluator:
    def __init__(
        self,
        cpu_count,
        num_runs,
        window_in_nodes,
        sample_size,
        vertex_feature_idx,
        edge_feature_idx,
    ):
        self.cpu_count = cpu_count
        self.num_runs = num_runs
        self.window_in_nodes = window_in_nodes
        self.sample_size = sample_size
        self.vertex_feature_idx = vertex_feature_idx
        self.edge_feature_idx = edge_feature_idx
        pass

    def get_result_dicts(self, X_func, dataset_name, estimator, max_elem):
        result_dicts = []

        for i in range(self.num_runs):
            print(f"starting run {i + 1} of {self.num_runs}")

            dataset = GraphPropPredDataset(name=dataset_name)

            if max_elem == None:
                frac = 1
            elif len(dataset) > max_elem:
                frac = 1 / len(dataset) * max_elem
            else:
                frac = 1

            # get subset
            used_new_idx = []
            split_idx = dataset.get_idx_split()
            for name, idx in split_idx.items():
                random_sublist = random.sample(list(idx), int(len(idx) * frac))
                split_idx[name] = random_sublist
                used_new_idx.extend(random_sublist)
            used_new_idx.sort()

            # convert ogb dicts to networkx graphs
            dict_calculator = GraphGenerator()
            sub_dataset = [dataset[i] for i in used_new_idx]
            graphs = dict_calculator.ogb_dataset_to_graphs(dataset=sub_dataset)

            X = X_func(
                dataset_name,
                graphs,
                self.cpu_count,
                self.sample_size,
                self.window_in_nodes,
                self.vertex_feature_idx,
                self.edge_feature_idx,
                split_idx,
            )

            # split data
            data = dict()
            for name, idx_list in split_idx.items():
                data[name] = dict()
                data[name]["X"] = np.array(
                    [X[used_new_idx.index(idx)] for idx in idx_list]
                )
                data[name]["y"] = np.array([dataset[idx][1] for idx in idx_list])

            # fit
            imp = SimpleImputer(strategy="most_frequent")
            y = imp.fit_transform(data["train"]["y"])
            estimator.fit(data["train"]["X"], y)
            # predict
            prediction = estimator.predict(data["valid"]["X"])
            data["valid"]["y_predicted"] = prediction
            # evaluate
            evaluator = Evaluator(name=dataset_name)
            input_dict = {
                "y_true": data["valid"]["y"],
                "y_pred": data["valid"]["y_predicted"],
            }
            result_dicts.append(evaluator.eval(input_dict))

            # newline for space in log file
            print()

        return result_dicts

    def evaluate(self, dataset_name, estimator, max_elem=None):
        methods = {"path2vec": get_paths2vec_X, "random": get_random_X}

        result_str = ""
        for methodname, method in methods.items():
            start_time = time.time()
            result_dicts = self.get_result_dicts(
                method, dataset_name, estimator, max_elem=max_elem
            )
            end_time = time.time()

            # print results
            result_str += f"dataset: {dataset_name}\n"
            result_str += f"method: {methodname}\n"
            result_str += f"runs: {self.num_runs}\n"
            result_str += f"s/run: {(end_time-start_time)/self.num_runs}\n"
            result_str += f"max_elem: {max_elem}\n"
            result_printer = ResultPrinter()
            result_str += result_printer.print(result_dicts=result_dicts)
            result_str += "\n"

        print(result_str)
        with open("result.result", "a") as result_file:
            result_file.write(result_str)

        return result_str

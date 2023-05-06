from ogb.graphproppred import Evaluator, GraphPropPredDataset
from sklearn import svm
import numpy as np

from paths2vec import Paths2Vec
from helpers import GraphGenerator, ResultPrinter

# dataset dependent variables
dataset_name = "ogbg-molfreesolv"
estimator = svm.SVR()


cpu_count = 16  # dataset independent variables
num_runs = 10  # number of runs to calculate mean
sample_size = (
    None  # number of selected subset of walks. If None, then full list is used
)
window_in_nodes = 3  # distance of nodes for Doc2Vec window

if __name__ == "__main__":
    result_dicts = []

    for i in range(num_runs):
        print(f"starting run {i + 1} of {num_runs}")

        dataset = GraphPropPredDataset(name=dataset_name)

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

        # split data
        data = dict()
        for name, idx_list in dataset.get_idx_split().items():
            data[name] = dict()
            data[name]["X"] = np.array([X[idx] for idx in idx_list])
            data[name]["y"] = np.array(
                [dataset[idx][1] for idx in idx_list]
            )  # .ravel()

        # fit
        estimator.fit(data["train"]["X"], data["train"]["y"])

        # predict
        data["valid"]["y_predicted"] = estimator.predict(data["valid"]["X"])

        # evaluate
        evaluator = Evaluator(name=dataset_name)
        input_dict = {
            "y_true": data["valid"]["y"],  # .reshape((-1, 1)),
            "y_pred": data["valid"]["y_predicted"],  # .reshape((-1, 1))
        }

        result_dicts.append(evaluator.eval(input_dict))

        # newline for space in log file
        print()

    # print results
    print(f"dataset: {dataset_name}")
    print(f"runs: {num_runs}")
    result_printer = ResultPrinter()
    result_printer.print(result_dicts=result_dicts)

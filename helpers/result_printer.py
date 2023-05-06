import numpy as np


class ResultPrinter:
    def __init__(self) -> None:
        pass

    def print(self, result_dicts):
        result = dict()

        for key in result_dicts[0].keys():
            result[key] = []
            for result_dict in result_dicts:
                result[key].append(result_dict[key])

        for key, list in result.items():
            print(f"metric: {key}")
            print(f"mean: {np.mean(list)}")
            print(f"std: {np.std(list)}")
            print(f"Full results: {list}")

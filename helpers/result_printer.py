import numpy as np


class ResultPrinter:
    def __init__(self) -> None:
        pass

    def print(self, result_dicts):
        result_str = ""
        result = dict()

        for key in result_dicts[0].keys():
            result[key] = []
            for result_dict in result_dicts:
                result[key].append(result_dict[key])

        for key, list in result.items():
            result_str += f"metric: {key}\n"
            result_str += f"mean: {np.mean(list)}\n"
            result_str += f"std: {np.std(list)}\n"
            result_str += f"Full results: {list}\n"

        return result_str

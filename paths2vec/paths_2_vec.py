from gensim.models.doc2vec import Doc2Vec

from .epoch_logger import EpochLogger
from .path_generator import PathGenerator


class Paths2Vec:
    def __init__(self, cpu_count):
        self.cpu_count = cpu_count

    def fit(self, graphs, corpus_file, sample_size, window_in_nodes):
        print("write paths to file")
        len_vertex_feature = len_vertex_feature = len(graphs[0].nodes[0]["feature"])
        first_edge = list(graphs[0].edges)[0]
        len_edge_feature = len_edge_feature = len(graphs[0].edges[first_edge])

        graph_to_path = PathGenerator(
            corpus_file=corpus_file, cpu_count=self.cpu_count, sample_size=sample_size
        )
        graph_to_path.paths_to_file(
            graphs=graphs,
            len_vertex_feature=len_vertex_feature,
            len_edge_feature=len_edge_feature,
        )

        print("get vectors")
        epoch_logger = EpochLogger(epochs=10)
        model = Doc2Vec(
            corpus_file=corpus_file,
            window=window_in_nodes * (len_vertex_feature + len_edge_feature),
            workers=self.cpu_count,
            callbacks=[epoch_logger],
        )

        return [model.dv[i] for i in range(len(graphs))]

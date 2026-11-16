import random
from typing import List, Tuple

from loaders.gisp import ALPHA, EDGE_COST, EDGE_PROBABILITY, NODE_WEIGHT, LoaderGISP, erdos_renyi, partition_edges


def test_generate_follows_the_dmiplib_settings(tmp_path) -> None:
    loader: LoaderGISP = LoaderGISP(data_dir=str(tmp_path))
    instances = loader.generate(5, seed=3, min_nodes=60, max_nodes=80)
    assert loader.generate(5, seed=3, min_nodes=60, max_nodes=80) == instances  # reproducible
    assert len({inst["name"] for inst in instances}) == 5
    for inst in instances:
        args = inst["args"]
        n: int = len(args["nodes"])
        assert 60 <= n <= 80 and args["nodes"] == list(range(n))
        assert inst["name"] == f"gisp_s3_{instances.index(inst)}_n{n}"
        assert (args["weight"], args["cost"]) == (NODE_WEIGHT, EDGE_COST) == (100, 1)
        edges: List[Tuple[int, int]] = args["permanent"] + args["removable"]
        assert len(set(edges)) == len(edges) and all(0 <= i < j < n for (i, j) in edges)


def test_the_graphs_have_the_generators_density_and_share_of_removable_edges(tmp_path) -> None:
    instances = LoaderGISP(data_dir=str(tmp_path)).generate(20, seed=0, min_nodes=150, max_nodes=150)
    pairs: int = 20 * 150 * 149 // 2
    edges: int = sum(len(i["args"]["permanent"]) + len(i["args"]["removable"]) for i in instances)
    removable: int = sum(len(i["args"]["removable"]) for i in instances)
    assert abs(edges / pairs - EDGE_PROBABILITY) < 0.01  # D-MIPLIB GISP medium: density 0.297
    assert abs(removable / edges - ALPHA) < 0.01  # D-MIPLIB GISP medium: 0.251 removable


def test_erdos_renyi_and_partition() -> None:
    assert erdos_renyi(5, 1.0, random.Random(0)) == [(i, j) for i in range(5) for j in range(i + 1, 5)]
    assert erdos_renyi(5, 0.0, random.Random(0)) == []
    edges = erdos_renyi(30, 0.5, random.Random(1))
    permanent, removable = partition_edges(edges, 0.25, random.Random(2))
    assert sorted(permanent + removable) == sorted(edges) and not set(permanent) & set(removable)
    assert partition_edges(edges, 1.0, random.Random(0)) == ([], edges)

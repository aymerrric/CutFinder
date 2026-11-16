from loaders.corridor import BUDGET_FRACTIONS, COST_MAX, UTILITY_MAX, LoaderCorridor, connectable, lattice


def test_generate_follows_the_corridor_generator(tmp_path) -> None:
    loader: LoaderCorridor = LoaderCorridor(data_dir=str(tmp_path))
    instances = loader.generate(6, seed=2, min_order=5, max_order=9)
    assert loader.generate(6, seed=2, min_order=5, max_order=9) == instances  # reproducible
    assert len({inst["name"] for inst in instances}) == 6
    for inst in instances:
        args = inst["args"]
        n: int = len(args["nodes"])
        order: int = round(n ** 0.5)
        assert 5 <= order <= 9 and order * order == n and args["edges"] == lattice(order)
        reserves = args["reserves"]
        assert len(reserves) == 3 and 0 in reserves and n - 1 in reserves and reserves == sorted(reserves)
        assert all(args["cost"][t] == 0 for t in reserves)
        assert all(1 <= args["cost"][i] <= COST_MAX for i in args["nodes"] if i not in reserves)
        assert all(1 <= args["utility"][i] <= UTILITY_MAX for i in args["nodes"])
        fraction: float = args["budget"] / sum(args["cost"].values())
        assert any(abs(fraction - f) < 1e-9 for f in BUDGET_FRACTIONS)
        assert inst["name"].startswith(f"corridor_s2_{instances.index(inst)}_o{order}_b")


def test_lattice_edges() -> None:
    assert lattice(2) == [(0, 1), (0, 2), (1, 3), (2, 3)]
    edges = lattice(5)
    assert len(edges) == 2 * 5 * 4 and all(j - i in (1, 5) for (i, j) in edges)


def test_connectable() -> None:
    path = dict(nodes=[0, 1, 2, 3], edges=[(0, 1), (1, 2), (2, 3)], reserves=[0, 3], cost={0: 0, 1: 4, 2: 5, 3: 0})
    assert connectable(**path, budget=9) and not connectable(**path, budget=8.9)
    split = dict(path, edges=[(0, 1), (2, 3)])
    assert not connectable(**split, budget=100)  # no path at all

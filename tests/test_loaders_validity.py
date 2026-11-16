import os
import shutil
from typing import List

from loaders import validity
from loaders.base import Instance
from problems import PROBLEMS

DATA: str = os.path.join(os.path.dirname(__file__), "data", "jsplib")


def test_every_generated_instance_has_a_name_and_a_note() -> None:
    for make in (validity.tsp_instances, validity.cwlp_instances, validity.jssp_generated, validity.gisp_instances,
                 validity.corridor_instances):
        instances: List[Instance] = make()
        assert len({i["name"] for i in instances}) == len(instances) and all(i["note"] for i in instances)


def test_generated_instances_are_reproducible() -> None:
    assert validity.cwlp_instances()[0]["args"] == validity.cwlp_instances()[0]["args"]
    assert validity.jssp_generated()[3]["args"] == validity.jssp_generated()[3]["args"]


def test_tight_instances_are_tight() -> None:
    cwlp = {i["name"]: i["args"] for i in validity.cwlp_instances()}
    tight = cwlp["tight_cap_1_20x6"]
    assert sum(tight["u"].values()) <= 1.06 * sum(tight["d"].values())  # buffer 1.05 +- the spread
    big = cwlp["big_customers_0_12x6"]
    assert all(cap / 3 < d <= cap / 2 for d in big["d"].values() for cap in big["u"].values())
    jssp = {i["name"]: i["args"] for i in validity.jssp_generated()}
    dominant = jssp["dominant_job_6x4"]
    assert sum(dominant["times"][0]) > max(sum(row) for row in dominant["times"][1:])
    flow = jssp["flow_shop_5x4"]
    assert all(order == list(range(4)) for order in flow["machines"])


def test_polygon_tour_is_the_perimeter() -> None:
    polygon: Instance = validity.tsp_instances()[-1]
    c = polygon["args"]["c"]
    n: int = polygon["args"]["n"]
    perimeter: float = sum(c[i, i % n + 1] for i in range(1, n + 1))
    assert perimeter < sum(c[i, (i + 1) % n + 1] for i in range(1, n + 1))  # skipping a vertex is longer


def test_jsplib_instances_are_parsed_without_the_comments(tmp_path) -> None:
    for name in ("ft06", "la01"):
        shutil.copy(os.path.join(DATA, name), tmp_path / name)  # no download
    ft06, la01 = validity.jsplib_instances(str(tmp_path), ["ft06", "la01"])
    assert ft06["args"]["n_jobs"] == 6 and ft06["args"]["n_machines"] == 6 and ft06["optimum"] == 55
    assert la01["args"]["n_jobs"] == 10 and la01["args"]["n_machines"] == 5
    assert la01["args"]["machines"][0][:2] == [1, 0] and la01["args"]["times"][0][:2] == [21, 53]
    assert "published optimum 666" in la01["note"]


def test_every_problem_has_validity_instances() -> None:
    assert len(PROBLEMS["tsp"].validity_instances("unused")) == 9
    assert len(PROBLEMS["cwlp"].validity_instances("unused")) == 10


def test_mcnd_validity_instances(tmp_path, monkeypatch) -> None:
    import tarfile
    with tarfile.open(tmp_path / "R.tgz", "w:gz") as tar:  # the public part, without a download
        tar.add(os.path.join(os.path.dirname(__file__), "data", "commalab", "r01.1.dow"), arcname="r01.1.dow")
    from loaders.benchmarks import COMMALAB_R_NAMES
    public: List[str] = list(validity.MCND_PUBLIC)
    assert len(public) == 27 and not set(public) & set(COMMALAB_R_NAMES)  # never the paper's test instances
    monkeypatch.setattr(validity, "MCND_PUBLIC", ["r01.1"])
    instances: List[Instance] = validity.mcnd_instances(str(tmp_path))
    assert len({i["name"] for i in instances}) == len(instances) and all(i["note"] for i in instances)
    by_name = {i["name"]: i["args"] for i in instances}
    assert "r01.1" in by_name and "random_tight_10n30a12k" in by_name
    split = by_name["split_demand"]
    assert split["d"][1] > max(split["u"].values())  # no single arc carries the demand: it must split
    cycle = by_name["cycle_only"]
    assert len(cycle["A"]) == len(cycle["N"]) == 6
    assert validity.mcnd_instances(str(tmp_path))[0]["args"] == instances[0]["args"]  # reproducible


def test_imo6_validity_notes_do_not_give_the_answer_away() -> None:
    instances: List[Instance] = validity.imo6_instances()  # notes are quoted to the LLM
    assert [inst["args"]["N"] for inst in instances] == validity.IMO6_SIZES
    assert [inst["note"] for inst in instances][:2] == ["the 2 x 2 grid", "the 3 x 3 grid"]


def test_gisp_validity_graphs() -> None:
    gisp = {i["name"]: i["args"] for i in validity.gisp_instances()}
    assert validity.gisp_instances()[2]["args"] == validity.gisp_instances()[2]["args"]  # reproducible
    for args in gisp.values():
        n: int = len(args["nodes"])
        edges = args["permanent"] + args["removable"]
        assert len(set(edges)) == len(edges) and all(0 <= i < j < n for (i, j) in edges)
    assert len(gisp["gisp_permanent_clique_6"]["permanent"]) == 15 and not gisp["gisp_permanent_clique_6"]["removable"]
    assert len(gisp["gisp_removable_clique_6"]["removable"]) == 15 and not gisp["gisp_removable_clique_6"]["permanent"]
    assert gisp["gisp_break_even_6"]["weight"] == gisp["gisp_break_even_6"]["cost"]

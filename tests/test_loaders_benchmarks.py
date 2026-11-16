import gzip
import os
import shutil
import tarfile
import zipfile
from typing import Dict, List

import pytest

from loaders import benchmarks
from loaders.base import Instance
from loaders.benchmarks import (
    commalab_r_instances, parse_dow,
    BENCHMARKS, beasley_instances, parse_beasley, parse_taillard, parse_tsplib, parse_tsplib_tour,
    taillard_instances, tour_length, tsplib_instances,
)
from problems import PROBLEMS

DATA: str = os.path.join(os.path.dirname(__file__), "data")
# smallest TSPLIB instance of each distance format that ships an optimal tour, with the optimal
# length TSPLIB publishes: the length of that tour under our distances must equal it
TSPLIB_OPTIMA: Dict[str, float] = {
    "ulysses16": 6859,  # GEO
    "gr24": 1272,  # EXPLICIT LOWER_DIAG_ROW
    "bays29": 2020,  # EXPLICIT FULL_MATRIX
    "bayg29": 1610,  # EXPLICIT UPPER_ROW
    "att48": 10628,  # ATT
    "eil51": 426,  # EUC_2D
}


def read_gz(name: str) -> str:
    with gzip.open(os.path.join(DATA, "tsplib", name), "rt") as f:
        return f.read()


@pytest.mark.parametrize("name", sorted(TSPLIB_OPTIMA))
def test_tsplib_distances_give_the_published_optimum(name: str) -> None:
    instance: Instance = parse_tsplib(read_gz(f"{name}.tsp.gz"))
    tour: List[int] = parse_tsplib_tour(read_gz(f"{name}.opt.tour.gz"))
    n: int = instance["args"]["n"]
    assert instance["name"] == name  # also for "NAME: ulysses16.tsp"
    assert sorted(tour) == list(range(1, n + 1))
    assert len(instance["args"]["c"]) == n * (n - 1)  # every ordered pair of distinct cities
    assert tour_length(instance["args"]["c"], tour) == TSPLIB_OPTIMA[name]


def test_tsplib_explicit_matrices_are_symmetric() -> None:
    for name in ("gr24", "bays29", "bayg29"):
        c = parse_tsplib(read_gz(f"{name}.tsp.gz"))["args"]["c"]
        assert all(c[i, j] == c[j, i] for (i, j) in c)


def test_tsplib_upper_diag_row_and_rounding() -> None:
    # UPPER_DIAG_ROW (si175's format) on 3 cities: rows (1,1) (1,2) (1,3) / (2,2) (2,3) / (3,3)
    text: str = ("NAME: t3\nTYPE: TSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: EXPLICIT\n"
                 "EDGE_WEIGHT_FORMAT: UPPER_DIAG_ROW\nEDGE_WEIGHT_SECTION\n0 4 5\n0 6\n0\nEOF\n")
    assert parse_tsplib(text)["args"]["c"] == {(1, 2): 4, (2, 1): 4, (1, 3): 5, (3, 1): 5, (2, 3): 6, (3, 2): 6}
    # EUC_2D rounds to the nearest integer: 2.5 -> 3
    euc: str = ("NAME: e2\nTYPE: TSP\nDIMENSION: 2\nEDGE_WEIGHT_TYPE: EUC_2D\n"
                "NODE_COORD_SECTION\n1 0 0\n2 2.5 0\nEOF\n")
    assert parse_tsplib(euc)["args"]["c"][1, 2] == 3


def test_tsplib_refuses_what_it_cannot_read() -> None:
    with pytest.raises(ValueError, match="symmetric TSP"):
        parse_tsplib("NAME: a\nTYPE: ATSP\nDIMENSION: 2\nEDGE_WEIGHT_TYPE: EXPLICIT\nEOF\n")
    with pytest.raises(ValueError, match="not supported"):
        parse_tsplib("NAME: a\nTYPE: TSP\nDIMENSION: 2\nEDGE_WEIGHT_TYPE: EUC_3D\nNODE_COORD_SECTION\n1 0 0 0\nEOF\n")
    with pytest.raises(ValueError, match="weights for"):
        parse_tsplib("NAME: a\nTYPE: TSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: EXPLICIT\n"
                     "EDGE_WEIGHT_FORMAT: UPPER_ROW\nEDGE_WEIGHT_SECTION\n1 2\nEOF\n")


def test_tsplib_instances_keeps_small_instances_and_their_optimum(tmp_path) -> None:
    archive: str = str(tmp_path / "ALL_tsp.tar.gz")  # what download() would have saved
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("ulysses16", "eil51", "att48"):
            for suffix in (".tsp.gz", ".opt.tour.gz"):
                tar.add(os.path.join(DATA, "tsplib", name + suffix), arcname=name + suffix)
    instances: List[Instance] = tsplib_instances(str(tmp_path), max_cities=50)  # eil51 left out
    assert [inst["name"] for inst in instances] == ["ulysses16", "att48"]  # smallest first
    assert [inst["optimum"] for inst in instances] == [6859, 10628]


def test_parse_beasley_cap61() -> None:
    with open(os.path.join(DATA, "beasley", "cap61")) as f:
        instance: Instance = parse_beasley(f.read(), "cap61")
    a = instance["args"]
    assert a["I"] == list(range(1, 51)) and a["J"] == list(range(1, 17))
    assert a["u"][1] == 15000.0 and a["f"][1] == 7500.0 and a["f"][11] == 0.0  # line 12 of the file
    assert a["d"][1] == 146.0 and a["d"][50] == 222.0
    assert a["c"][1, 1] == 6739.725 and len(a["c"]) == 50 * 16
    assert sum(a["u"].values()) >= sum(a["d"].values())


def test_parse_beasley_layout_and_size_check() -> None:
    # 2 warehouses, 3 customers: then the cost rows are per WAREHOUSE (c[i, j] = row j, column i)
    text: str = "2 3\n10 100\n20 200\n1 2 3\n11 12 13\n21 22 23\n"
    a = parse_beasley(text, "toy")["args"]
    assert a["c"] == {(1, 1): 11, (2, 1): 12, (3, 1): 13, (1, 2): 21, (2, 2): 22, (3, 2): 23}
    assert a["u"] == {1: 10, 2: 20} and a["f"] == {1: 100, 2: 200} and a["d"] == {1: 1, 2: 2, 3: 3}
    with pytest.raises(ValueError, match="expected"):
        parse_beasley("2 3\n10 100\n20 200\n1 2 3\n", "short")


def test_beasley_instances_reads_the_zip(tmp_path) -> None:
    with zipfile.ZipFile(tmp_path / "BeasleyData.zip", "w") as z:
        z.write(os.path.join(DATA, "beasley", "cap61"), "Istanze/cap61")
        z.writestr("Istanze/readme.txt", "not an instance")
    assert [inst["name"] for inst in beasley_instances(str(tmp_path))] == ["cap61"]


def test_parse_taillard_ta01() -> None:
    with open(os.path.join(DATA, "taillard", "ta01")) as f:
        a = parse_taillard(f.read(), "ta01")["args"]
    assert a["n_jobs"] == 15 and a["n_machines"] == 15
    assert a["machines"][0][:3] == [6, 12, 4] and a["times"][0][:3] == [94, 66, 10]
    assert all(sorted(row) == list(range(15)) for row in a["machines"])


def test_parse_taillard_checks() -> None:
    with pytest.raises(ValueError, match="numbers for"):
        parse_taillard("2 2\n0 1 1 2\n", "short")
    with pytest.raises(ValueError, match="every machine once"):
        parse_taillard("1 2\n0 1 0 2\n", "twice")


def test_taillard_instances_uses_downloaded_files(tmp_path, monkeypatch) -> None:
    shutil.copy(os.path.join(DATA, "taillard", "ta01"), tmp_path / "ta01")
    monkeypatch.setattr(benchmarks, "TAILLARD_NAMES", ["ta01"])
    assert [inst["name"] for inst in taillard_instances(str(tmp_path))] == ["ta01"]


def test_download_keeps_an_existing_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "f"
    path.write_text("already here")
    monkeypatch.setattr(benchmarks.urllib.request, "urlopen", lambda *a, **k: pytest.fail("no download"))
    assert benchmarks.download("http://example.invalid/f", str(path)) == str(path)


def test_every_benchmark_has_a_problem() -> None:
    assert set(BENCHMARKS) <= set(PROBLEMS)


def test_parse_dow_r01_1() -> None:
    with open(os.path.join(DATA, "commalab", "r01.1.dow")) as f:
        a = parse_dow(f.read(), "r01.1")["args"]
    assert a["N"] == list(range(1, 11)) and len(a["A"]) == 35 and a["K"] == list(range(1, 11))
    # the first arc line: "1 2 100 613 595 1 1" = i, j, unit cost, capacity, fixed cost
    assert (a["c"][1, 2], a["u"][1, 2], a["f"][1, 2]) == (100, 613, 595)
    assert (a["c"][1, 10], a["u"][1, 10], a["f"][1, 10]) == (54, 81, 244)
    # the last commodity line: "9 8 73"
    assert (a["O"][10], a["D"][10], a["d"][10]) == (9, 8, 73) and (a["O"][9], a["D"][9]) == (6, 1)
    assert a["A"] == sorted(a["A"]) and set(a["c"]) == set(a["f"]) == set(a["u"]) == set(a["A"])


def test_parse_dow_checks() -> None:
    with pytest.raises(ValueError, match="lines for"):
        parse_dow("T\n2 2 1\n1 2 1 1 1\n", "short")
    with pytest.raises(ValueError, match="twice"):
        parse_dow("T\n2 2 1\n1 2 1 1 1\n1 2 1 1 1\n1 2 5\n", "double")


def test_commalab_r_instances_reads_the_archive(tmp_path) -> None:
    with tarfile.open(tmp_path / "R.tgz", "w:gz") as tar:
        tar.add(os.path.join(DATA, "commalab", "r01.1.dow"), arcname="r01.1.dow")
    assert [inst["name"] for inst in commalab_r_instances(str(tmp_path), names=["r01.1"])] == ["r01.1"]
    with pytest.raises(ValueError, match="lacks r10.1"):
        commalab_r_instances(str(tmp_path), names=["r10.1"])
    assert len(benchmarks.COMMALAB_R_NAMES) == 81 and benchmarks.COMMALAB_R_NAMES[:2] == ["r10.1", "r10.2"]


def test_download_creates_its_folder(tmp_path, monkeypatch) -> None:
    import io
    monkeypatch.setattr(benchmarks.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(b"data"))
    path: str = str(tmp_path / "raw" / "mcnd" / "R.tgz")
    assert benchmarks.download("https://example.org/R.tgz", path) == path
    assert open(path, "rb").read() == b"data" and not os.path.exists(path + ".part")


def test_imo6_benchmark_is_the_papers_test_set() -> None:
    import statistics
    from loaders.benchmarks import imo6_benchmark
    instances = imo6_benchmark()
    sizes = [inst["args"]["N"] for inst in instances]
    assert sizes == list(range(3, 37)) and len(sizes) == BENCHMARKS["imo6"].paper_count
    # model sizes (tests/test_models_imo6.py checks these counts on a built model): Table 20
    variables = [N * N + 3 * N * N * (N + 1) // 2 for N in sizes]
    constraints = [2 * N + N * N + (N + 1) * N * (N + 1) // 2 for N in sizes]
    assert statistics.mean(variables) == BENCHMARKS["imo6"].paper_variables
    assert statistics.mean(constraints) == BENCHMARKS["imo6"].paper_constraints


# a 4-node GISP model as Distributional MIPLIB stores it (the SCIP LP layout of its "easy" files):
# permanent edges (0, 1), (1, 2); removable edges (0, 2), (2, 3)
GISP_LP: str = """\\ SCIP STATISTICS
\\   Problem name     : GISP
Minimize
 Obj: -100 node_0 -100 node_1 -100 node_2 -100 node_3 +1 edge_0_2 +1 edge_2_3
Subject to
 _C1: +1 node_0 +1 node_1 <= +1
 _C2: +1 node_1 +1 node_2 <= +1
 _C3: -1 edge_0_2 +1 node_0 +1 node_2 <= +1
 _C4: -1 edge_2_3 +1 node_2 +1 node_3 <= +1
Bounds
 0 <= node_0 <= 1
 0 <= node_1 <= 1
 0 <= node_2 <= 1
 0 <= node_3 <= 1
 0 <= edge_0_2 <= 1
 0 <= edge_2_3 <= 1
Binaries
 node_0 node_1 node_2 node_3 edge_0_2 edge_2_3
End
"""


def gisp_csv(path: str, rows: List[Dict[str, str]]) -> None:
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["", "MILP", "format"])
        writer.writeheader()
        writer.writerows(rows)


def test_gisp_benchmark_reads_dmiplib_lp_and_mps_rows(tmp_path) -> None:
    import gurobipy as gp
    from loaders.benchmarks import gisp_benchmark
    lp_path: str = str(tmp_path / "small.lp")
    with open(lp_path, "w") as f:
        f.write(GISP_LP)
    mps_path: str = str(tmp_path / "small.mps")
    model = gp.read(lp_path)
    model.write(mps_path)  # the MPS layout of the "medium" and "hard" files
    model.dispose()
    gisp_csv(str(tmp_path / "gisp_hard_test.csv"), [  # already there: nothing is downloaded
        {"": "0", "MILP": repr(GISP_LP.encode()), "format": ""},
        {"": "1", "MILP": repr(open(mps_path).read().encode()), "format": "mps"},
    ])
    instances = gisp_benchmark(str(tmp_path))
    assert [inst["name"] for inst in instances] == ["gisp_dmiplib_hard_00", "gisp_dmiplib_hard_01"]
    expected = {"nodes": [0, 1, 2, 3], "permanent": [(0, 1), (1, 2)], "removable": [(0, 2), (2, 3)],
                "weight": 100, "cost": 1}
    assert all(inst["args"] == expected for inst in instances)
    m = PROBLEMS["gisp"].base(**expected)  # the same model as theirs
    assert (m.NumVars, m.NumConstrs) == (6, 4)
    m.dispose()


def test_gisp_from_model_refuses_another_model(tmp_path) -> None:
    import gurobipy as gp
    from loaders.benchmarks import gisp_from_model
    path: str = str(tmp_path / "other.lp")
    with open(path, "w") as f:
        f.write(GISP_LP.replace("-100 node_3", "-90 node_3"))
    m = gp.read(path)
    with pytest.raises(ValueError, match="not one weight"):
        gisp_from_model(m, "other")
    m.dispose()


def test_gisp_benchmark_entry() -> None:
    assert BENCHMARKS["gisp"].paper_count == 100 and BENCHMARKS["gisp"].fetch is benchmarks.gisp_benchmark


def test_corridor_benchmark_is_fixed_and_feasible() -> None:
    import statistics
    from loaders.benchmarks import corridor_benchmark
    from loaders.corridor import connectable
    instances = corridor_benchmark()
    assert len(instances) == BENCHMARKS["corridor"].paper_count == 50
    assert [i["name"] for i in corridor_benchmark()] == [i["name"] for i in instances]  # the same every time
    assert all(connectable(**i["args"]) for i in instances)
    assert all(20 <= round(len(i["args"]["nodes"]) ** 0.5) <= 30 for i in instances)
    # model sizes: x and y per cell (n + 1), a flow per arc; rows: budget, 3 reserves, source, arcs, balances
    variables = [len(i["args"]["nodes"]) + 2 * len(i["args"]["edges"]) + 1 for i in instances]
    constraints = [5 + 2 * len(i["args"]["edges"]) + len(i["args"]["nodes"]) for i in instances]
    assert statistics.mean(variables) == BENCHMARKS["corridor"].paper_variables
    assert statistics.mean(constraints) == BENCHMARKS["corridor"].paper_constraints

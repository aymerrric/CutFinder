import pytest

from loaders.imo6 import LoaderIMO6, instance


def test_generate_draws_distinct_sizes_in_order(tmp_path) -> None:
    loader: LoaderIMO6 = LoaderIMO6(data_dir=str(tmp_path))
    instances = loader.generate(5, seed=3, min_n=10, max_n=20)
    sizes = [inst["args"]["N"] for inst in instances]
    assert sizes == sorted(set(sizes)) and len(sizes) == 5 and all(10 <= N <= 20 for N in sizes)
    assert [inst["name"] for inst in instances] == [f"imo6_{N}" for N in sizes]
    assert loader.generate(5, seed=3, min_n=10, max_n=20) == instances  # reproducible
    assert [inst["args"]["N"] for inst in loader.generate(3, seed=0, min_n=4, max_n=6)] == [4, 5, 6]


def test_generate_refuses_more_instances_than_sizes(tmp_path) -> None:
    with pytest.raises(ValueError, match="only 3 sizes"):
        LoaderIMO6(data_dir=str(tmp_path)).generate(4, seed=0, min_n=4, max_n=6)


def test_instance_is_only_its_size() -> None:
    assert instance(7) == {"name": "imo6_7", "args": {"N": 7}}

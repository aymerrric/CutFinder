import pytest

from evolution.individual import COPY, ELITE, INIT, Individual


def test_defaults() -> None:
    ind: Individual = Individual(idea="2-cycle", code="m.addConstr(...)")
    assert ind.fitness is None and not ind.evaluated
    assert ind.generator == INIT and ind.parents == []


def test_ids_are_unique() -> None:
    a: Individual = Individual(idea="a", code="")
    b: Individual = Individual(idea="a", code="")
    assert a.id != b.id


def test_parents_lists_are_not_shared() -> None:
    a: Individual = Individual(idea="a", code="")
    b: Individual = Individual(idea="b", code="")
    a.parents.append("x")
    assert b.parents == []


def test_dict_round_trip_keeps_everything() -> None:
    ind: Individual = Individual(idea="a", code="c", fitness=0.5, generator=COPY, parents=["p1"])
    assert Individual.from_dict(ind.to_dict()) == ind


@pytest.mark.parametrize("generator", [ELITE, COPY])
def test_copy_is_a_new_individual_with_the_same_cut(generator: str) -> None:
    parent: Individual = Individual(idea="a", code="c", fitness=0.5, generator="Lifted", parents=["x"])
    child: Individual = parent.copy(generator)
    assert child is not parent and child.id != parent.id
    assert (child.idea, child.code, child.fitness) == ("a", "c", 0.5)  # fitness not re-evaluated
    assert child.generator == generator and child.parents == [parent.id]
    assert child.is_copy and not parent.is_copy


def test_copy_with_another_generator_is_refused() -> None:
    with pytest.raises(ValueError):
        Individual(idea="a", code="c").copy("Lifted")

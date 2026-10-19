import os
from pathlib import Path
from typing import List

import pytest

from evolution.individual import ELITE, Individual
from evolution.population import Population


def make_generation(fitnesses: List[float]) -> List[Individual]:
    return [Individual(idea=f"idea {f}", code="", fitness=f) for f in fitnesses]


def test_empty_population() -> None:
    pop: Population = Population(3)
    assert pop.current == [] and pop.history == [] and pop.best is None
    assert pop.generation == 0


def test_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Population(0)


@pytest.mark.parametrize("n", [2, 4])
def test_wrong_generation_size_is_refused(n: int) -> None:
    pop: Population = Population(3)
    with pytest.raises(ValueError):
        pop.set_generation(make_generation([1.0] * n))
    assert pop.current == []  # nothing changed


def test_unevaluated_individual_is_refused() -> None:
    pop: Population = Population(2)
    with pytest.raises(ValueError):
        pop.set_generation([Individual(idea="a", code="", fitness=1.0), Individual(idea="b", code="")])


def test_previous_generation_goes_to_history() -> None:
    pop: Population = Population(2)
    first: List[Individual] = make_generation([1.0, 2.0])
    second: List[Individual] = make_generation([3.0, 4.0])
    pop.set_generation(first)
    pop.set_generation(second)
    assert pop.history == [first] and pop.current == second
    assert pop.generation == 1


def test_current_is_a_copy_of_the_given_list() -> None:
    pop: Population = Population(2)
    gen: List[Individual] = make_generation([1.0, 2.0])
    pop.set_generation(gen)
    gen.append(Individual(idea="extra", code="", fitness=9.0))
    assert len(pop.current) == 2


def test_best_is_kept_across_worse_generations() -> None:
    pop: Population = Population(2)
    pop.set_generation(make_generation([1.0, 5.0]))
    best: Individual = pop.best
    pop.set_generation(make_generation([2.0, 3.0]))
    assert pop.best is best and pop.best.fitness == 5.0
    pop.set_generation(make_generation([6.0, 0.0]))
    assert pop.best.fitness == 6.0


def test_tie_keeps_the_older_best() -> None:
    pop: Population = Population(2)
    pop.set_generation(make_generation([5.0, 5.0]))
    first_best: Individual = pop.best
    pop.set_generation(make_generation([5.0, 1.0]))
    assert pop.best is first_best


def test_top_is_sorted_best_first() -> None:
    pop: Population = Population(4)
    pop.set_generation(make_generation([2.0, 4.0, 1.0, 3.0]))
    assert [ind.fitness for ind in pop.top(2)] == [4.0, 3.0]
    assert len(pop.top(10)) == 4


def test_same_individual_twice_is_refused() -> None:
    pop: Population = Population(2)
    ind: Individual = Individual(idea="a", code="", fitness=1.0)
    with pytest.raises(ValueError):
        pop.set_generation([ind, ind])


def test_reusing_an_elite_object_is_refused_but_its_copy_is_accepted() -> None:
    pop: Population = Population(2)
    first: List[Individual] = make_generation([1.0, 5.0])
    pop.set_generation(first)
    with pytest.raises(ValueError):
        pop.set_generation([first[1], first[1]])
    pop.set_generation([first[1].copy(ELITE)] + make_generation([2.0]))
    assert pop.generation == 1


def test_all_individuals_skips_copies() -> None:
    pop: Population = Population(2)
    first: List[Individual] = make_generation([1.0, 5.0])
    pop.set_generation(first)
    pop.set_generation([first[1].copy(ELITE)] + make_generation([2.0]))
    assert [ind.fitness for ind in pop.all_individuals()] == [1.0, 5.0, 2.0]


def test_elite_copy_with_equal_fitness_keeps_the_original_best() -> None:
    pop: Population = Population(2)
    first: List[Individual] = make_generation([1.0, 5.0])
    pop.set_generation(first)
    pop.set_generation([first[1].copy(ELITE)] + make_generation([2.0]))
    assert pop.best is first[1]


def test_save_then_load(tmp_path: Path) -> None:
    pop: Population = Population(2)
    pop.set_generation(make_generation([1.0, 5.0]))
    pop.set_generation(make_generation([2.0, 3.0]))
    path: str = os.path.join(tmp_path, "pop.json")
    pop.save(path)
    loaded: Population = Population.load(path)
    assert loaded.size == 2
    assert loaded.current == pop.current and loaded.history == pop.history
    assert loaded.best == pop.best and loaded.generation == 1


def test_save_then_load_empty(tmp_path: Path) -> None:
    path: str = os.path.join(tmp_path, "pop.json")
    Population(3).save(path)
    loaded: Population = Population.load(path)
    assert loaded.size == 3 and loaded.best is None and loaded.current == []

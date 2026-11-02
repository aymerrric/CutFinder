from utils.run_files import run_name_of


def test_run_name_of_a_population_file() -> None:
    assert run_name_of("../runs/tsp_m_saumon_1/population.json") == "tsp_m_saumon_1"
    assert run_name_of("../runs/tsp_m_saumon_1.json") == "tsp_m_saumon_1"  # old layout

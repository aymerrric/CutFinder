import os
import tarfile

from utils.log_archive import archive_path, pack_logs


def test_pack_logs_replaces_the_folder_by_one_archive(tmp_path) -> None:
    folder = tmp_path / "logs" / "run_a"
    (folder / "sub").mkdir(parents=True)
    (folder / "base_i1_seed42.log").write_text("Gurobi log 1")
    (folder / "sub" / "x.log").write_text("Gurobi log 2")
    archive: str = pack_logs(str(folder))
    assert archive == str(tmp_path / "logs" / "run_a.tar.gz")
    assert not folder.exists() and os.listdir(tmp_path / "logs") == ["run_a.tar.gz"]
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == ["run_a", "run_a/base_i1_seed42.log", "run_a/sub", "run_a/sub/x.log"]
        assert tar.extractfile("run_a/base_i1_seed42.log").read() == b"Gurobi log 1"


def test_a_resumed_run_gets_a_second_archive(tmp_path) -> None:
    folder = tmp_path / "run_a"
    for k in range(2):
        folder.mkdir()
        (folder / "f.log").write_text(str(k))
        pack_logs(str(folder))
    assert sorted(os.listdir(tmp_path)) == ["run_a.tar.gz", "run_a_2.tar.gz"]
    assert archive_path(str(folder)) == str(tmp_path / "run_a_3.tar.gz")


def test_nothing_to_pack(tmp_path) -> None:
    assert pack_logs(str(tmp_path / "missing")) is None
    (tmp_path / "empty").mkdir()
    assert pack_logs(str(tmp_path / "empty")) is None and (tmp_path / "empty").exists()

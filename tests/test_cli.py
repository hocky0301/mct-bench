import pytest

from mct_bench.cli import main, parser


@pytest.mark.parametrize("arguments", [
    ["run", "--repeats", "0"], ["run", "--warmup", "-1"],
    ["run", "--bits", "3"], ["run", "--bits", "4", "--config", "anything.json"],
    ["run", "--eval-batch-size", "0"],
])
def test_invalid_arguments(arguments):
    with pytest.raises(SystemExit) as result:
        parser().parse_args(arguments)
    assert result.value.code == 2


def test_no_overwrite_existing_results(tmp_path, capsys):
    assert main(["run", "--output", str(tmp_path)]) == 1
    assert "FileExistsError" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_dataset_count_checked_before_download(tmp_path):
    output = tmp_path / "no-output"
    assert main(["run", "--output", str(output), "--eval-samples", "10001"]) == 1
    assert not output.exists()

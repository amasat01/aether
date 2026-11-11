"""The one validator (Rust, rcore::manifest): verdict and text equal the old Python validators' on recorded cases
(tests/data/schema_validator_cases.jsonl: one or two documents per distinct message, recorded from the Python code it replaced)."""
import json
from pathlib import Path
import pytest
from raptor.schema import validate_manifest, manifest

CASES = [json.loads(l) for l in (Path(__file__).parent / "data" / "schema_validator_cases.jsonl").read_text().splitlines()]


def test_recorded_cases():
    assert len(CASES) > 100
    bad = []
    for c in CASES:
        try:
            validate_manifest(json.loads(c["json"])); err = None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        if err != c["err"]:
            bad.append((c["json"][:120], err, c["err"]))
    assert not bad, bad[:3]


def test_not_a_dict():
    with pytest.raises(TypeError, match="expected a dict, got list"):
        validate_manifest([])


def test_name_is_in_the_text():
    with pytest.raises(ValueError, match=r"'my.json' was built for plugin schema v9"):
        manifest.check_schema_version({"schema_version": 9}, name="my.json")
    assert manifest.check_schema_version({"version": 2}, allow_legacy_version_key=True) == 2
    assert manifest.check_schema_version({"version": 2}) == 1
    with pytest.raises(ValueError, match=r"'x': index_planes must be a non-empty list"):
        manifest.check_index_planes({"index_planes": []}, name="x")


def test_block_descriptor_cases():
    """validate_block_descriptor (Rust) against what the Python validator said, recorded over a mutation grid of one valid document."""
    from raptor.schema.blocks import validate_block_descriptor
    cases = json.loads((Path(__file__).parent / "data" / "block_descriptor_cases.json").read_text())
    assert len(cases) > 500
    bad = []
    for c in cases:
        try:
            validate_block_descriptor(json.loads(c["json"]), name="nm.json"); err = None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        if err != c["err"]:
            bad.append((c["json"][:150], err, c["err"]))
    assert not bad, (len(bad), bad[:3])

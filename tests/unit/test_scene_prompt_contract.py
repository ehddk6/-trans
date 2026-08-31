from pathlib import Path

import pytest

from translation_forensics.prompt_contract import validate_prompt_contract


ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "prompts"


@pytest.mark.parametrize(
    "name",
    [
        "scene-semantic-reconstruction-v1",
        "scene-dialogue-realization-v1",
        "scene-subtitle-segmentation-v1",
        "scene-source-faithful-v1",
        "scene-semantic-drift-critic-v1",
        "korean-dialogue-critic-v1",
        "scene-dialogue-repair-v1",
        "scene-visual-semantic-observation-v1",
    ],
)
def test_scene_v2_prompt_contracts_validate(name: str) -> None:
    report = validate_prompt_contract(PROMPTS / f"{name}.manifest.json")
    assert report["status"] == "pass", report["errors"]


def test_scene_v2_contract_rejects_wrong_role(tmp_path: Path) -> None:
    source = PROMPTS / "scene-dialogue-realization-v1.manifest.json"
    destination = tmp_path / source.name
    text = source.read_text(encoding="utf-8").replace('"role":"translation-terra"', '"role":"meaning-frame-terra"')
    destination.write_text(text, encoding="utf-8")
    report = validate_prompt_contract(destination, root=PROMPTS)
    assert report["status"] == "fail"
    assert any("role" in error for error in report["errors"])

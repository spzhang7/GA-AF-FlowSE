from pathlib import Path

import yaml


CONFIG_ROOT = Path("configs/af")


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_published_5000_step_templates_use_the_paper_budget() -> None:
    names = (
        "af_libritts_dns10s_sft20k_5000step.yaml",
        "gaaf_libritts_dns10s_sft20k_ogaf_0_to_5000.yaml",
        "af_libritts_dns10s_sft20k_ovrl_only_0_to_5000.yaml",
    )
    for name in names:
        config = _load(name)
        assert config["run"]["optimizer_steps"] == 5000
        assert config["run"]["conditions_per_step"] == 16
        assert config["rollout"]["candidates_per_condition"] == 8
        assert config["rollout"]["training_nfe"] == 10
        assert config["rollout"]["evaluation_nfe"] == 32
        assert not any(
            any(token in str(value).lower() for token in ("disk1", "disk3"))
            for value in _walk_strings(config)
        )


def test_published_ogaf_template_declares_the_gradient_gate() -> None:
    config = _load("gaaf_libritts_dns10s_sft20k_ogaf_0_to_5000.yaml")
    assert config["ogaf"]["primary"] == "dnsmos_ovrl"
    assert config["ogaf"]["auxiliaries"] == [
        "eres2net_speaker_similarity",
        "speechbertscore",
    ]


def test_smoke_template_is_small_but_keeps_the_same_candidate_semantics() -> None:
    config = _load("af_smoke.yaml")
    assert config["run"]["mode"] == "smoke"
    assert config["rollout"]["candidates_per_condition"] == 4
    assert config["advantage"]["normalization"] == "global_complete_LxK"


def _walk_strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


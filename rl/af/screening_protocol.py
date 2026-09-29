"""Build the immutable Gate-A protocol fingerprint and endpoint key space."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from rl.common.conditioning import ConditioningProtocol
from rl.rewards.evaluators import ResolvedHFModel, resolve_hf_model


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "missing"


def strict_json_mapping(path: str | Path) -> dict[str, str]:
    """Load a JSON object while rejecting duplicate utterance keys."""
    path = Path(path)

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(text, str) for key, text in value.items()
    ):
        raise ValueError("manifest must map utterance IDs to transcript strings")
    return value


def endpoint_key(protocol_hash: str, utterance: str, nfe: int, seed: int) -> str:
    return sha256_json(
        {
            "protocol_hash": protocol_hash,
            "utterance": utterance,
            "nfe": int(nfe),
            "latent_seed": int(seed),
        }
    )


def expected_endpoint_keys(
    protocol_hash: str,
    utterances: list[str],
    nfes: list[int],
    seeds: list[int],
) -> dict[str, tuple[str, int, int]]:
    result = {}
    for utterance in utterances:
        for nfe in nfes:
            for seed in seeds:
                key = endpoint_key(protocol_hash, utterance, nfe, seed)
                if key in result:
                    raise AssertionError("endpoint key collision")
                result[key] = (utterance, int(nfe), int(seed))
    return result


@dataclass(frozen=True)
class ProtocolBundle:
    protocol_hash: str
    components: dict
    manifest: dict[str, str]
    noisy_hashes: dict[str, str]
    clean_hashes: dict[str, str]
    checkpoint_path: Path
    checkpoint_sha256: str
    vocoder_path: Path
    vocoder_sha256: str
    speaker_model: ResolvedHFModel | None
    asr_model: ResolvedHFModel | None

    def output_dir(self, root: str | Path) -> Path:
        return Path(root) / self.protocol_hash


def _source_fingerprints(project_root: Path) -> dict[str, str]:
    relative_paths = [
        "infer.py",
        "model/cfm.py",
        "model/model_utils.py",
        "model/modules.py",
        "model/backbones/dit.py",
        "tools/evaluate_dnsmos.py",
        "rl/common/conditioning.py",
        "rl/common/flow_matching.py",
        "rl/common/normalization.py",
        "rl/common/flowse_interface.py",
        "rl/rewards/metrics.py",
        "rl/rewards/evaluators.py",
        "rl/af/advantage_estimation.py",
        "rl/af/screening_protocol.py",
    ]
    result = {}
    for relative in relative_paths:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = sha256_file(path)
    return result


def _audio_fingerprints(
    manifest: dict[str, str], noisy_dir: Path, clean_dir: Path
) -> tuple[dict[str, str], dict[str, str]]:
    noisy_hashes = {}
    clean_hashes = {}
    for utterance in sorted(manifest):
        noisy_path = noisy_dir / f"{utterance}.wav"
        clean_path = clean_dir / f"{utterance}.wav"
        for path in (noisy_path, clean_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        noisy_hashes[utterance] = sha256_file(noisy_path)
        clean_hashes[utterance] = sha256_file(clean_path)
    return noisy_hashes, clean_hashes


def _dnsmos_fingerprint(official_dir: Path) -> dict[str, str]:
    files = {
        "script": official_dir / "dnsmos_local.py",
        "p808_model": official_dir / "DNSMOS/model_v8.onnx",
        "primary_model": official_dir / "DNSMOS/sig_bak_ovr.onnx",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return {name: sha256_file(path) for name, path in files.items()}


def build_protocol(
    config: dict,
    *,
    project_root: str | Path = ".",
    resolve_models: bool = True,
) -> ProtocolBundle:
    """Validate all frozen inputs and hash the complete Gate-A protocol."""
    project_root = Path(project_root).resolve()
    conditioning = ConditioningProtocol.from_config(config["conditioning"])
    rollout = config["rollout"]
    if float(rollout["cfg_strength"]) != 0.0:
        raise ValueError("Gate A requires cfg_strength=0.0")
    if str(rollout["solver"]) != "euler":
        raise ValueError("Gate A requires the Euler solver")
    seeds = [int(value) for value in rollout["latent_seeds"]]
    if len(seeds) != int(rollout["group_size"]) or len(set(seeds)) != len(seeds):
        raise ValueError("group_size and unique latent_seeds are inconsistent")

    manifest_path = Path(config["manifest"])
    manifest = strict_json_mapping(manifest_path)
    expected_utterances = int(config["run"]["expected_utterances"])
    if len(manifest) != expected_utterances:
        raise ValueError(
            f"manifest has {len(manifest)} utterances, expected {expected_utterances}"
        )
    noisy_dir = Path(config["noisy_dir"])
    clean_dir = Path(config["clean_dir"])
    noisy_hashes, clean_hashes = _audio_fingerprints(
        manifest, noisy_dir, clean_dir
    )

    flowse_config_path = Path(config["flowse_config"])
    flowse = yaml.safe_load(flowse_config_path.read_text(encoding="utf-8"))
    infer_conf = flowse["infer"]
    test_conf = infer_conf["test"]
    expected_cond_type = "noisy" if conditioning.mode == "text" else "wotext"
    if str(test_conf["cond_type"]) != expected_cond_type:
        raise ValueError(
            f"FlowSE cond_type={test_conf['cond_type']!r} conflicts with "
            f"conditioning.mode={conditioning.mode!r}"
        )
    if float(test_conf["cfg_strength"]) != 0.0 or int(test_conf["steps"]) != int(
        rollout["deployment_nfe"]
    ):
        raise ValueError("FlowSE config must freeze CFG=0 and deployment NFE")
    checkpoint_path = Path(test_conf["checkpoint"]) / test_conf["pt_name"]
    vocoder_dir = Path(infer_conf["nnet_conf"]["vocoder"]["local_path"])
    vocoder_path = vocoder_dir / "pytorch_model.bin"
    vocoder_config_path = vocoder_dir / "config.yaml"
    vocabulary_path = Path(flowse["model"]["tokenizer_path"])
    for path in (
        checkpoint_path,
        vocoder_path,
        vocoder_config_path,
        vocabulary_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_hash = sha256_file(checkpoint_path)
    vocoder_hash = sha256_file(vocoder_path)
    expected_checkpoint = config["protocol"].get("expected_checkpoint_sha256")
    if not expected_checkpoint or len(str(expected_checkpoint)) != 64:
        raise ValueError("checkpoint hash is not frozen; run freeze_config.py first")
    if checkpoint_hash != expected_checkpoint:
        raise ValueError("checkpoint SHA-256 differs from the frozen expectation")

    parity_path = Path(config["protocol"]["required_parity_report"])
    if not parity_path.is_file():
        raise FileNotFoundError(
            f"required upstream parity report is missing: {parity_path}"
        )
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity_checks = {
        "passed": parity.get("passed") is True,
        "conditioning": parity.get("conditioning") == conditioning.fingerprint(),
        "checkpoint": parity.get("checkpoint_sha256") == checkpoint_hash,
        "vocoder": parity.get("vocoder_sha256") == vocoder_hash,
        "flowse_config": parity.get("flowse_config_sha256")
        == sha256_file(flowse_config_path),
        "nfe": int(parity.get("nfe", -1)) == int(rollout["deployment_nfe"]),
        "source": parity.get("source_sha256")
        == {
            path: sha256_file(project_root / path)
            for path in (
                "model/cfm.py",
                "rl/common/conditioning.py",
                "rl/common/flow_matching.py",
                "rl/common/normalization.py",
                "rl/common/flowse_interface.py",
                "scripts/fetch_flowse.py",
            )
        },
    }
    failed_parity = [name for name, passed in parity_checks.items() if not passed]
    if failed_parity:
        raise ValueError(
            "upstream parity report conflicts with protocol: "
            + ", ".join(failed_parity)
        )

    dnsmos = _dnsmos_fingerprint(Path(config["dnsmos_official_dir"]))
    evaluator_conf = config["evaluators"]
    for evaluator_name in ("speaker", "asr"):
        evaluator = evaluator_conf[evaluator_name]
        if evaluator["enabled"] and len(str(evaluator["revision"])) != 40:
            raise ValueError(
                f"{evaluator_name} revision is mutable; run freeze_config.py first"
            )
    speaker_model = (
        resolve_hf_model(evaluator_conf["speaker"])
        if resolve_models and evaluator_conf["speaker"]["enabled"]
        else None
    )
    asr_model = (
        resolve_hf_model(evaluator_conf["asr"])
        if resolve_models and evaluator_conf["asr"]["enabled"]
        else None
    )
    package_versions = {
        name: installed_version(name)
        for name in (
            "numpy",
            "scipy",
            "librosa",
            "soundfile",
            "pesq",
            "pystoi",
            "onnxruntime",
            "torch",
            "torchaudio",
            "torchdiffeq",
            "vocos",
            "transformers",
            "huggingface-hub",
        )
    }
    required_packages = ["pesq", "pystoi", "onnxruntime", "torch", "vocos"]
    if evaluator_conf["speaker"]["enabled"] or evaluator_conf["asr"]["enabled"]:
        required_packages.append("transformers")
    missing = [name for name in required_packages if package_versions[name] == "missing"]
    if missing:
        raise RuntimeError(f"missing Gate-A packages: {', '.join(missing)}")

    components = {
        "schema_version": 2,
        "run": config["run"],
        "conditioning": conditioning.fingerprint(),
        "rollout": rollout,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "utterances": sorted(manifest),
            "transcript_sha256": {
                utterance: hashlib.sha256(text.encode("utf-8")).hexdigest()
                for utterance, text in sorted(manifest.items())
            },
        },
        "audio": {
            "noisy_dir": str(noisy_dir),
            "clean_dir": str(clean_dir),
            "noisy_sha256": noisy_hashes,
            "clean_sha256": clean_hashes,
        },
        "flowse": {
            "config_path": str(flowse_config_path),
            "config_sha256": sha256_file(flowse_config_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "mel_frontend": infer_conf["nnet_conf"]["mel_spec"],
            "input_sample_rate": int(infer_conf["datareader"]["mix_fs"]),
            "model_sample_rate": int(
                infer_conf["nnet_conf"]["mel_spec"]["target_sample_rate"]
            ),
            "output_sample_rate": int(infer_conf["save"]["fs"]),
            "tokenizer": flowse["model"]["tokenizer"],
            "vocabulary_sha256": sha256_file(vocabulary_path),
        },
        "vocoder": {
            "weights_sha256": vocoder_hash,
            "config_sha256": sha256_file(vocoder_config_path),
        },
        "normalization": config["protocol"]["normalization"],
        "upstream_parity": {
            "report_sha256": sha256_file(parity_path),
            "checks": parity_checks,
        },
        "reward_and_analysis": {
            "rewards": config["rewards"],
            "advantage": config["advantage"],
            "bootstrap": config["bootstrap"],
            "thresholds": config["thresholds"],
            "baseline_diagnostics": config["baseline_diagnostics"],
        },
        "evaluator_device": evaluator_conf["device"],
        "dnsmos": dnsmos,
        "speaker_evaluator": (
            speaker_model.fingerprint()
            if speaker_model is not None
            else {"enabled": False}
        ),
        "asr_evaluator": (
            asr_model.fingerprint() if asr_model is not None else {"enabled": False}
        ),
        "package_versions": package_versions,
        "source_sha256": _source_fingerprints(project_root),
    }
    protocol_hash = sha256_json(components)
    return ProtocolBundle(
        protocol_hash=protocol_hash,
        components=components,
        manifest=manifest,
        noisy_hashes=noisy_hashes,
        clean_hashes=clean_hashes,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
        vocoder_path=vocoder_path,
        vocoder_sha256=vocoder_hash,
        speaker_model=speaker_model,
        asr_model=asr_model,
    )


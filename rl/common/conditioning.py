"""Method-neutral conditioning protocol shared by controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditioningProtocol:
    mode: str
    use_text: bool
    drop_text: bool

    @classmethod
    def from_config(cls, config: dict) -> "ConditioningProtocol":
        protocol = cls(
            mode=str(config["mode"]),
            use_text=bool(config["use_text"]),
            drop_text=bool(config["drop_text"]),
        )
        protocol.validate()
        return protocol

    def validate(self) -> None:
        allowed = {"text": (True, False), "wotext": (False, True)}
        if self.mode not in allowed:
            raise ValueError(f"conditioning.mode must be one of {sorted(allowed)}")
        expected = allowed[self.mode]
        if (self.use_text, self.drop_text) != expected:
            raise ValueError(
                f"conditioning mode {self.mode!r} requires "
                f"use_text={expected[0]} and drop_text={expected[1]}"
            )

    def policy_text(self, transcript: str) -> str:
        """Return the only text that may be passed to the FlowSE policy."""
        if self.mode == "text":
            if not transcript.strip():
                raise ValueError("text-conditioned policy requires a transcript")
            return transcript
        return " "

    def fingerprint(self) -> dict[str, str | bool]:
        return {
            "mode": self.mode,
            "use_text": self.use_text,
            "drop_text": self.drop_text,
        }

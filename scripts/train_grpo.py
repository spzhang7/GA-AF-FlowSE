"""Public launcher for the controlled FlowSE-GRPO baseline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.grpo.trainer import main  # noqa: E402


if __name__ == "__main__":
    main()

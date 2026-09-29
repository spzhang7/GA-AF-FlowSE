"""Public launcher for AF and GA-AF training.

The implementation lives in ``rl.af.trainer``; this thin entry point keeps the
command shown in the README stable for users of the release.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.af.trainer import main  # noqa: E402


if __name__ == "__main__":
    main()

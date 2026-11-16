import subprocess
import sys

import fnd_api
import fnd_ml


def test_packages_expose_version() -> None:
    assert fnd_ml.__version__ == "0.1.0"
    assert fnd_api.__version__ == "0.1.0"


def test_api_imports_without_training_packages() -> None:
    """The API must start on a machine that only has the base dependencies installed."""
    code = (
        "import sys, fnd_api.main\n"
        "heavy = ('torch', 'transformers', 'xgboost', 'matplotlib')\n"
        "print(','.join(name for name in heavy if name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""

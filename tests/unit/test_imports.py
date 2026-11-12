import fnd_api
import fnd_ml


def test_packages_expose_version() -> None:
    assert fnd_ml.__version__ == "0.1.0"
    assert fnd_api.__version__ == "0.1.0"

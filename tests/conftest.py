import json

import pytest

from nbs_runtime.contract import load_package


@pytest.fixture(scope="session")
def pkg():
    return load_package()


@pytest.fixture(scope="session")
def conformance(pkg):
    return json.loads((pkg.root / "conformance_examples.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def policy_cases(pkg):
    return json.loads((pkg.root / "policy_conformance.json").read_text(encoding="utf-8"))

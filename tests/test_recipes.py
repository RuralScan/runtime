import json

import pytest

from nbs_runtime.recipes import (
    RecipeError,
    discover,
    load_family,
    load_recipe,
    load_roles,
    validate_all,
)
from recipe_helpers import toy_family, write_recipe, write_roles

# ------------------------------------------------------------------ the recipes in the repo

def test_repo_recipes_are_valid():
    assert validate_all() == []


def test_agroforestry_cross_family_roles_and_weights():
    fam = load_family("agroforestry", "cross_family")
    assert set(fam.variables) == {"annual_precipitation", "elevation", "slope", "ndvi"}
    assert {r.variable for r in fam.limiting} == {"annual_precipitation", "elevation"}
    w = fam.modulating_weights()
    assert set(w) == {"slope", "ndvi"}
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["slope"] == pytest.approx(0.25 / 0.40)
    m = fam.manifest()
    assert len(m["recipes"]["slope"]["sha256"]) == 64


# ------------------------------------------------------------------ one recipe

def test_toy_family_loads(tmp_path):
    root, roles = toy_family(tmp_path)
    fam = load_family("toy", "fam", root, roles)
    assert fam.modulating_weights() == pytest.approx({"slope": 0.6, "ndvi": 0.4})
    assert fam.modulating_weights("equal") == pytest.approx({"slope": 0.5, "ndvi": 0.5})
    assert discover(root) == {("toy", "fam"): sorted((root / "toy" / "fam").glob("*.json"))}


@pytest.mark.parametrize("x, score, msg", [
    ([1, 1, 2], [0.1, 0.2, 0.3], "strictly increasing"),
    ([3, 2, 1], [0.1, 0.2, 0.3], "strictly increasing"),
    ([1, 2, 3], [0.1, 0.2], "nodes"),
    ([1, 2], [0.1, 1.2], "between 0 and 1"),
    ([1], [0.5], "at least two nodes"),
])
def test_bad_functions(tmp_path, x, score, msg):
    p = write_recipe(tmp_path, "nbs", "fam", "v", x, score, weight=1)
    with pytest.raises(RecipeError, match=msg):
        load_recipe(p)


def test_weight_as_list_is_rejected(tmp_path):
    p = write_recipe(tmp_path, "nbs", "fam", "v", [1, 2], [0.1, 0.2], weight=[0.4])
    with pytest.raises(RecipeError, match="not a list"):
        load_recipe(p)


def test_band_must_contain_score(tmp_path):
    p = write_recipe(tmp_path, "nbs", "fam", "v", [1, 2], [0.5, 0.6], low=[0.6, 0.5], high=[0.7, 0.7])
    with pytest.raises(RecipeError, match="low"):
        load_recipe(p)


def test_folder_must_match(tmp_path):
    p = write_recipe(tmp_path, "nbs", "fam", "v", [1, 2], [0.1, 0.2])
    data = json.loads(p.read_text())
    data["family"] = "other"
    p.write_text(json.dumps(data))
    with pytest.raises(RecipeError, match="folder"):
        load_recipe(p)


# ------------------------------------------------------------------ roles and families

def test_roles_default_and_override(tmp_path):
    path = write_roles(tmp_path / "roles.toml", {"a": "limiting", "b": "modulating"},
                       {"mangrove": {"a": "modulating"}})
    assert load_roles("agro", path) == {"a": "limiting", "b": "modulating"}
    assert load_roles("mangrove", path)["a"] == "modulating"


def test_unknown_role_is_rejected(tmp_path):
    path = write_roles(tmp_path / "roles.toml", {"a": "filter"})
    with pytest.raises(RecipeError, match="unknown role"):
        load_roles("x", path)


def test_variable_without_role(tmp_path):
    root, roles = toy_family(tmp_path)
    write_recipe(root, "toy", "fam", "extra", [0, 1], [0, 1], weight=0.1)
    with pytest.raises(RecipeError, match="no role"):
        load_family("toy", "fam", root, roles)


def test_modulating_without_weight(tmp_path):
    root, roles = toy_family(tmp_path)
    write_recipe(root, "toy", "fam", "ndvi", [0.0, 1.0], [0.0, 1.0])          # no weight
    with pytest.raises(RecipeError, match="without weight"):
        load_family("toy", "fam", root, roles)


def test_limiting_needs_no_weight(tmp_path):
    root, roles = toy_family(tmp_path)
    write_recipe(root, "toy", "fam", "rain", [500, 2000], [0.2, 1.0])         # no weight
    load_family("toy", "fam", root, roles)


from nbs_runtime import variables as V
from nbs_runtime.recipes import discover, load_recipe


def test_every_variable_has_one_id_one_group_and_one_subgroup():
    ids = [v.id for v in V.CATALOG]
    assert len(ids) == len(set(ids)) == 15
    assert {v.group for v in V.CATALOG} == set(V.GROUPS)
    tree = V.by_group()
    assert list(tree) == list(V.GROUPS)
    assert sum(len(vs) for subs in tree.values() for vs in subs.values()) == len(ids)


def test_the_groups_follow_the_ones_you_asked_for():
    tree = V.by_group()
    assert set(tree["Soil"]) == {"Fertility", "Water"}
    assert [v.id for v in tree["Soil"]["Fertility"]] == ["soil_ph", "soil_organic_carbon"]
    assert {v.id for subs in tree["Climate variables"].values() for v in subs} == {
        "annual_precipitation", "mean_annual_temperature", "aridity_index"}
    assert {v.id for subs in tree["Project risk"].values() for v in subs} == {
        "accessibility_travel_time", "distance_to_road", "electrification_index", "protected_area_status"}


def test_hugging_face_paths_match_the_layout_of_the_dataset():
    rels = {v.id: v.path.removeprefix(V.HF) for v in V.CATALOG if v.source == V.FILE}
    assert rels == {
        "annual_precipitation": "raw/worldclim_v2_bioclim_precip/wc2.1_30s_bio_12.tif",
        "mean_annual_temperature": "raw/worldclim_v2_bioclim_temp/wc2.1_30s_bio_1.tif",
        "aridity_index": "raw/global_aridity_index_v3/ai_v31_yr.tif",
        "soil_drainage": "raw/hwsd_v2/HWSD2.bil",
        "distance_to_road": "raw/osm_road_network/gis_osm_roads_free_1.shp",
    }
    assert V.BY_ID["soil_drainage"].params["sqlite"] == V.HF + "raw/hwsd_v2/HWSD2.sqlite"


def test_scale_factors_and_labels():
    assert V.BY_ID["aridity_index"].scale == 0.0001
    for v in V.CATALOG:
        if v.kind == "categorical":
            assert v.class_labels
        if v.plausible:
            assert v.plausible[0] < v.plausible[1]
    assert V.BY_ID["land_cover"].class_labels[40] == "Cropland"


def test_select_by_group_skip_and_cost():
    assert [v.id for v in V.select(["Soil"])] == ["soil_ph", "soil_organic_carbon", "soil_drainage"]
    assert "slope" not in [v.id for v in V.select(None, include_slow=False)]
    assert [v.id for v in V.select(["Soil"], skip=["soil_ph"])] == ["soil_organic_carbon", "soil_drainage"]


def test_terrain_and_vegetation_variables_of_the_recipes():
    assert V.BY_ID["elevation"].path == V.BY_ID["slope"].path           # one asset, two variables
    ndvi = V.BY_ID["ndvi"]
    assert ndvi.scale == 0.0001 and ndvi.derive == "annual_median" and ndvi.params["year"]


def test_every_recipe_variable_has_a_catalog_entry_with_the_same_unit():
    for files in discover().values():
        for path in files:
            r = load_recipe(path)
            assert r.variable in V.BY_ID, f"{path}: no catalog entry"
            assert V.same_unit(r.unit, V.BY_ID[r.variable].unit), f"{path}: unit {r.unit!r}"


from nbs_runtime import variables as V

POLICY = "planted_terrain_soil_v1"


def test_every_variable_has_one_id_one_group_and_one_subgroup():
    ids = [v.id for v in V.CATALOG]
    assert len(ids) == len(set(ids)) == 13
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
    rels = {v.id: v.detail.get("rel") for v in V.CATALOG if v.source in (V.HF_RASTER, V.HWSD, V.OSM_ROADS)}
    assert rels == {
        "annual_precipitation": "raw/worldclim_v2_bioclim_precip/wc2.1_30s_bio_12.tif",
        "mean_annual_temperature": "raw/worldclim_v2_bioclim_temp/wc2.1_30s_bio_1.tif",
        "aridity_index": "raw/global_aridity_index_v3/ai_v31_yr.tif",
        "soil_drainage": "raw/hwsd_v2/HWSD2.bil",
        "distance_to_road": "raw/osm_road_network/gis_osm_roads_free_1.shp",
    }
    assert V.BY_ID["soil_drainage"].detail["sqlite"] == "raw/hwsd_v2/HWSD2.sqlite"


def test_scale_factors_and_labels():
    assert V.BY_ID["aridity_index"].detail["scale"] == 0.0001
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


def test_roles_come_from_the_pinned_package_not_from_the_catalog(pkg):
    roles = V.contract_roles(pkg)
    assert roles["slope"] == ["suitability_score"]
    assert roles["land_cover"] == ["eligibility_gate"]
    assert roles["protected_area_status"] == ["hard_exclusion"]
    assert V.role_text("slope", roles) == "scores the index"
    assert V.role_text("aridity_index", roles) == "not in the package"
    assert "operational" in V.role_text("accessibility_travel_time", roles)


def test_a_score_the_policy_does_not_use_is_not_called_a_score_of_the_index(pkg):
    roles = V.contract_roles(pkg)
    used = V.policy_variables(pkg, POLICY)
    assert used == {"slope", "soil_organic_carbon", "soil_ph", "land_cover", "protected_area_status"}
    assert V.role_text("slope", roles, used) == "scores the index (first-pass policy)"
    assert V.role_text("annual_precipitation", roles, used) == "a score in the package, not in the first-pass policy"
    assert V.role_text("land_cover", roles, used) == "host gate"

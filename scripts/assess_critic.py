"""Reproduce controlled CRITIC sensitivity experiments; these are not pilot results."""
import json

from nbs_runtime.priority import critic_weights


def assess():
    raw = {"a": [0, 1, 2, 3, 4, 5], "b": [0, 1, 2, 3, 4, 50],
           "c": [1, 0, 1, 0, 1, 0]}
    classified = {
        k: [0 if x == 0 else .33 if x <= 2 else .66 if x <= 4 else 1 for x in vals]
        for k, vals in raw.items()
    }
    cases = {
        "raw_minmax": (raw, "minmax"),
        "restricted_area": ({k: v[:4] for k, v in raw.items()}, "minmax"),
        "classified_fixed_scale": (classified, "fixed_0_1"),
        "classified_minmax": (classified, "minmax"),
        "duplicate_added": ({**raw, "a_copy": raw["a"]}, "minmax"),
        "constant_added": ({**raw, "constant": [2] * 6}, "minmax"),
        "opposing": ({"a": [0, .5, 1], "b": [1, .5, 0]}, "minmax"),
        "identical": ({"a": [0, .5, 1], "b": [0, .5, 1]}, "minmax"),
    }
    return {name: critic_weights(values, scale=scale) for name, (values, scale) in cases.items()}


if __name__ == "__main__":
    print(json.dumps(assess(), indent=2, allow_nan=False))

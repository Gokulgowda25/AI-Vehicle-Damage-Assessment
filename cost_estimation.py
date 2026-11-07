"""
cost_estimation.py - repair cost estimate (Indian rupees, Bengaluru workshop rates, excluding GST)
for the 7 classes in data.yaml.

The estimate depends on
    * the detected damage type (each class has its own repair / replacement rules)
    * the size of each damage in the photo
    * repair versus replacement of the part
    * labour hours (LABOUR_RATE per hour), paint and materials
    * the number of damages (shared labour and paint when repaired together)
    * the detection confidence (low confidence widens the range)

Public functions
    estimate_repair_cost(damage_class)     -> (min, max)   one class (kept for old code)
    estimate_from_detections(detections)   -> dict         all detections of one photo
"""
from severity import (get_profile, dedupe_detections, area_fraction, effective_level,
                      assess_overall_severity)

LABOUR_RATE = 500        # Rs per hour
MIN_CHARGE = 800         # smallest job a workshop will quote

# repair: {hours, paint (lo, hi), materials (lo, hi)}
# replace: {hours, paint (lo, hi), parts (lo, hi)}  - used when the damage covers at least `replace_area` of the photo
COST_TABLE = {
    "slight_scratch": {"repair": dict(hours=1.5, paint=(900, 1600), materials=(0, 0)), "replace": None, "replace_area": None},
    "severe_scratch": {"repair": dict(hours=3.5, paint=(2200, 4200), materials=(200, 600)), "replace": None, "replace_area": None},
    "slight_deformation": {"repair": dict(hours=3.0, paint=(300, 900), materials=(0, 0)), "replace": None, "replace_area": None},
    "medium_deformation": {"repair": dict(hours=5.0, paint=(1800, 3200), materials=(200, 600)),
                           "replace": None, "replace_area": None},
    "severe_deformation": {"repair": dict(hours=8.0, paint=(3500, 5500), materials=(500, 1200)),
                           "replace": dict(hours=6.0, paint=(2800, 4800), parts=(8000, 18000)), "replace_area": 0.05},
    "crack_and_hole": {"repair": dict(hours=2.5, paint=(1200, 2500), materials=(400, 1200)),
                       "replace": dict(hours=3.0, paint=(1500, 3000), parts=(5000, 11000)), "replace_area": 0.12},
    "windshield_damage": {"repair": dict(hours=0.7, paint=(0, 0), materials=(600, 1500)),
                          "replace": dict(hours=2.5, paint=(0, 0), parts=(6000, 14000)), "replace_area": 0.04},
}
DEFAULT_AREA = 0.10      # used when a detection has no box (a typical medium-sized damage)

GENERIC = {"repair": dict(hours=3.0, paint=(1500, 3000), materials=(300, 900)),
           "replace": dict(hours=4.0, paint=(2000, 3500), parts=(6000, 14000)), "replace_area": 0.15}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _estimate_item(d):
    prof = get_profile(d["name"])
    rules = COST_TABLE.get(prof["key"], GENERIC)
    area = area_fraction(d)
    a = DEFAULT_AREA if area is None else area
    size = _clamp(0.7 + 2.0 * a, 0.7, 1.5)        # bigger damage = more work and paint (capped: close-up photos look large)

    replace = bool(rules["replace"]) and rules["replace_area"] is not None and a >= rules["replace_area"]
    plan = rules["replace"] if replace else rules["repair"]

    labour = plan["hours"] * size * LABOUR_RATE
    paint_lo, paint_hi = plan["paint"][0] * size, plan["paint"][1] * size
    if replace:
        plo, phi = plan["parts"]
        pos = _clamp(a / 0.35, 0.0, 1.0)
        mat_lo, mat_hi = plo + (phi - plo) * 0.5 * pos, plo + (phi - plo) * (0.5 + 0.5 * pos)
    else:
        mat_lo, mat_hi = plan["materials"]
    return {"class": d["name"], "label": prof["label"], "part": prof["part"],
            "action": "Replace" if replace else "Repair", "s": effective_level(d) / 5.0,
            "conf": d.get("confidence", 1.0), "area": area, "labour": labour,
            "paint": (paint_lo, paint_hi), "parts": (mat_lo, mat_hi), "hours": plan["hours"] * size}


def estimate_from_detections(detections):
    """Total estimate for one photo, using every detection."""
    dets = dedupe_detections(detections)
    if not dets:
        return {"severity": "Unknown", "description": "No damage detected.", "min": 0, "max": 0,
                "items": [], "n": 0, "primary": "", "safety": ""}
    items = [_estimate_item(d) for d in dets]
    n = len(items)
    share = max(0.80, 1 - 0.07 * (n - 1))        # prep, masking and paint are shared when damages are fixed together

    total_lo = total_hi = 0.0
    for it in items:
        lo = it["parts"][0] + (it["labour"] * 0.90 + it["paint"][0]) * share
        hi = it["parts"][1] + (it["labour"] * 1.20 + it["paint"][1]) * share
        if it["conf"] < 0.5:                      # unsure detection -> wider range
            lo, hi = lo * 0.92, hi * 1.12
        it["lo"], it["hi"] = lo, hi
        total_lo += lo
        total_hi += hi

    lo_r = max(MIN_CHARGE, int(total_lo // 100 * 100))
    hi_r = max(int(-(-total_hi // 100) * 100), lo_r + 500)

    sev = assess_overall_severity(dets, total_max=hi_r)
    top = max(items, key=lambda it: it["hi"])
    return {"severity": sev["severity"], "description": sev["description"], "safety": sev["safety"],
            "min": lo_r, "max": hi_r, "items": items, "n": n, "primary": top["class"]}


def estimate_repair_cost(damage_class):
    """One class only -> (min, max). Kept so older code keeps working."""
    est = estimate_from_detections([{"name": damage_class, "confidence": 1.0}])
    return est["min"], est["max"]

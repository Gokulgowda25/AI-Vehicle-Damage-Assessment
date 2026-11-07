"""
severity.py - severity rules for the 7 classes in data.yaml:
crack_and_hole, medium_deformation, severe_deformation, severe_scratch,
slight_deformation, slight_scratch, windshield_damage

Public functions
    assess_severity(damage_class)                 -> (label, description)       (single class, kept for old code)
    assess_overall_severity(detections, ...)      -> dict                       (all detections of one photo)
"""
import re

# level: 1 = cosmetic ... 5 = major structural/panel damage
CLASS_PROFILE = {
    "slight_scratch": {"label": "Slight scratch", "level": 1.0, "part": "Paint / surface",
                       "desc": "Surface scratch. Polishing or touch-up paint is usually enough."},
    "slight_deformation": {"label": "Slight dent", "level": 2.0, "part": "Body panel",
                           "desc": "Small dent. Often fixed by paintless dent repair or light panel work."},
    "medium_deformation": {"label": "Medium dent", "level": 3.0, "part": "Body panel",
                           "desc": "Moderate dent. Panel beating and repainting are likely."},
    "severe_scratch": {"label": "Deep scratch", "level": 3.2, "part": "Paint / panel",
                       "desc": "Deep scratch through the paint. Sanding, filler and repainting are needed."},
    "crack_and_hole": {"label": "Crack / hole", "level": 3.5, "part": "Body panel / bumper",
                       "desc": "Crack or hole. Plastic welding, patching or part replacement is needed."},
    "windshield_damage": {"label": "Windshield damage", "level": 3.5, "part": "Windshield",
                          "desc": "Windshield damage. A chip can be resin-repaired, a crack needs replacement."},
    "severe_deformation": {"label": "Severe dent", "level": 5.0, "part": "Body panel",
                           "desc": "Major deformation. Panel replacement or heavy straightening is likely."},
}


def normalize_class(name):
    return re.sub(r"[^a-z]+", "_", str(name).lower()).strip("_")


def get_profile(name):
    """Profile for a class name. Unknown names are inferred from words such as slight / severe / scratch."""
    key = normalize_class(name)
    if key in CLASS_PROFILE:
        return {**CLASS_PROFILE[key], "key": key, "known": True}
    toks = set(key.split("_"))
    level = 2.5
    if toks & {"slight", "minor", "small", "light"}:
        level = 1.8
    elif toks & {"severe", "major", "heavy", "large"}:
        level = 4.5
    elif toks & {"medium", "moderate"}:
        level = 3.0
    if toks & {"windshield", "windscreen"}:
        level = max(level, 3.5)
    return {"label": key.replace("_", " ").title() or "Damage", "level": level, "part": "Vehicle",
            "desc": "Damage needs a manual inspection to confirm the repair.", "key": key, "known": False}


def level_to_label(level):
    return "Severe" if level >= 4.3 else "Medium" if level >= 2.4 else "Slight"


def area_fraction(d):
    """Share of the photo covered by the detection box (0..1), or None if no box is available."""
    b = d.get("box")
    if not b or len(b) != 4:
        return None
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedupe_detections(detections):
    """Drop boxes that cover the same damage twice (keep the most confident one)."""
    kept = []
    for d in sorted(detections, key=lambda x: -x.get("confidence", 0)):
        if d.get("box") and any(k.get("box") and _iou(d["box"], k["box"]) > 0.6 for k in kept):
            continue
        kept.append(d)
    return kept


def effective_level(d):
    """Class level adjusted by how large the damage is in the photo."""
    level = get_profile(d["name"])["level"]
    a = area_fraction(d)
    if a is not None:
        level += 0.7 if a > 0.30 else 0.4 if a > 0.15 else -0.3 if a < 0.03 else 0.0
    return min(5.0, max(1.0, level))


def assess_severity(damage_class):
    """Single class -> (label, description). Kept so older code keeps working."""
    p = get_profile(damage_class)
    return level_to_label(p["level"]), p["desc"]


def assess_overall_severity(detections, total_max=None):
    """Overall severity for one photo from all its detections.
    Uses the worst damage, its size, the number of damages and (optionally) the total repair cost."""
    dets = dedupe_detections(detections)
    if not dets:
        return {"severity": "Unknown", "score": 0, "description": "No damage detected.", "safety": ""}
    levels = [effective_level(d) for d in dets]
    worst = max(levels)
    score = min(5.0, worst + min(1.0, 0.3 * (len(dets) - 1)))
    if total_max and total_max >= 25000:
        score = max(score, 4.4)
    label = level_to_label(score)

    top = dets[levels.index(worst)]
    prof = get_profile(top["name"])
    names = {get_profile(d["name"])["key"] for d in dets}
    safety = ""
    if "windshield_damage" in names:
        safety = "Windshield damage can block your view. Get it checked before long drives."
    elif label == "Severe":
        safety = "Avoid driving until the car has been inspected."
    elif "crack_and_hole" in names:
        safety = "Check for sharp edges and hidden damage behind the panel."
    n = len(dets)
    desc = (f"{n} damaged area{'s' if n != 1 else ''} found. Main issue: {prof['label'].lower()}. "
            f"{prof['desc']}")
    if n > 2:
        desc += " Several damages usually mean a longer repair."
    return {"severity": label, "score": round(score, 2), "description": desc, "safety": safety}

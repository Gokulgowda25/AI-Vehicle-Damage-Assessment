def assess_severity(damage_class):

    damage = damage_class.lower().replace("_", " ")

    if "slight" in damage:
        severity = "Slight"
        description = "Minor damage. Basic repair may be sufficient."

    elif "medium" in damage:
        severity = "Medium"
        description = "Moderate damage. Professional repair is recommended."

    elif "severe" in damage:
        severity = "Severe"
        description = "Major damage. Detailed inspection is required."

    elif "crack" in damage or "hole" in damage:
        severity = "Medium"
        description = "Crack or hole detected. Further inspection is needed."

    elif "windshield" in damage:
        severity = "Medium"
        description = "Windshield damage detected. Inspection is required."

    else:
        severity = "Unknown"
        description = "Damage severity needs manual inspection."

    return severity, description
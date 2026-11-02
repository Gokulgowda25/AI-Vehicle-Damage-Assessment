def estimate_repair_cost(damage_class):

    damage = damage_class.lower().replace("_", " ")

    if "slight scratch" in damage:
        cost = (1000, 2500)

    elif "slight deformation" in damage:
        cost = (2000, 4000)

    elif "medium deformation" in damage:
        cost = (4000, 8000)

    elif "severe deformation" in damage:
        cost = (8000, 18000)

    elif "severe scratch" in damage:
        cost = (3000, 7000)

    elif "crack" in damage or "hole" in damage:
        cost = (2500, 8000)

    elif "windshield" in damage:
        cost = (4000, 12000)

    else:
        cost = (0, 0)

    return cost
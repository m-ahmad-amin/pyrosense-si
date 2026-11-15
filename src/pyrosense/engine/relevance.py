import math


def score_investigation_for_mission(inv: dict, params: dict) -> dict:
    target_o2 = float(params.get("o2_pct", 34.0))
    target_p = float(params.get("pressure_psi", 10.2))
    target_g = float(params.get("gravity_g", 0.16))
    target_flow = float(params.get("airflow_cms", 12.0))
    focus_pillar = params.get("pillar", "All")

    o2_min, o2_max = inv["o2_range_pct"]
    if o2_min <= target_o2 <= o2_max:
        o2_score = 100.0
    else:
        dist = min(abs(target_o2 - o2_min), abs(target_o2 - o2_max))
        o2_score = max(25.0, 100.0 - dist * 2.4)

    p_min, p_max = inv["pressure_range_psi"]
    if p_min <= target_p <= p_max:
        p_score = 100.0
    else:
        dist_p = min(abs(target_p - p_min), abs(target_p - p_max))
        p_score = max(35.0, 100.0 - dist_p * 8.5)

    f_min, f_max = inv["airflow_range_cms"]
    if f_min <= target_flow <= f_max:
        flow_score = 100.0
    else:
        dist_f = min(abs(target_flow - f_min), abs(target_flow - f_max))
        flow_score = max(30.0, 100.0 - dist_f * 2.5)

    if 0.03 <= target_g <= 0.60:
        if inv["id"] == "E-FIELD":
            g_score = 98.0
        elif inv["id"] in ("BASS-II", "BASS", "SAFFIRE-II", "DAFT-2", "BRE"):
            g_score = 88.0
        else:
            g_score = 72.0
    else:
        g_score = 92.0

    pillar_bonus = 0.0
    if focus_pillar != "All":
        if inv["pillar"].lower() == focus_pillar.lower():
            pillar_bonus = 12.0
        elif inv.get("SecondaryPillar", "").lower() == focus_pillar.lower():
            pillar_bonus = 5.0
        else:
            pillar_bonus = -14.0

    composite = (
        0.34 * o2_score
        + 0.20 * p_score
        + 0.22 * flow_score
        + 0.24 * g_score
        + pillar_bonus
    )
    final_score = round(min(99.4, max(18.0, composite)), 1)

    rationale = []
    if o2_score >= 88.0:
        rationale.append(f"Covers target {target_o2:.0f}% O2 regime ({o2_min:.0f}-{o2_max:.0f}% O2)")
    if inv["id"] == "E-FIELD" and 0.03 <= target_g <= 0.60:
        rationale.append(f"Simulates {target_g:.2f}g partial-gravity buoyancy via ion-driven electric field")
    if inv["id"] == "CFI-G" and target_o2 >= 35.0:
        rationale.append("Highest oxygen envelope in PSI (up to 85% O2)")
    if inv["id"] in ("DAFT-2", "SAME-R") and target_g > 0.0:
        rationale.append("Discriminates surface regolith dust from pyrolysis smoke")
    if not rationale:
        rationale.append(f"Provides {inv['fuel_state'].lower()}-fuel baseline in {inv['facility']} ({inv['hardware_insert']})")

    return {
        "id": inv["id"],
        "psi_id": inv["psi_id"],
        "title": inv["title"],
        "pillar": inv["pillar"],
        "secondary_pillar": inv.get("SecondaryPillar", ""),
        "facility": inv["facility"],
        "hardware_insert": inv["hardware_insert"],
        "fuel_state": inv["fuel_state"],
        "relevance_score": final_score,
        "o2_match": round(o2_score, 1),
        "pressure_match": round(p_score, 1),
        "flow_match": round(flow_score, 1),
        "gravity_match": round(g_score, 1),
        "rationale": "; ".join(rationale),
        "key_findings": inv["key_findings"],
    }


def compare_investigations(inv_a: dict, inv_b: dict) -> dict:
    sensors_a = set(inv_a["sensors"])
    sensors_b = set(inv_b["sensors"])
    o2_overlap = [
        max(inv_a["o2_range_pct"][0], inv_b["o2_range_pct"][0]),
        min(inv_a["o2_range_pct"][1], inv_b["o2_range_pct"][1]),
    ]
    has_o2_overlap = o2_overlap[0] <= o2_overlap[1]

    return {
        "left": inv_a,
        "right": inv_b,
        "shared_facility": inv_a["facility"] == inv_b["facility"],
        "shared_hardware_insert": inv_a["hardware_insert"] == inv_b["hardware_insert"],
        "same_fuel_state": inv_a["fuel_state"] == inv_b["fuel_state"],
        "o2_overlap_pct": o2_overlap if has_o2_overlap else None,
        "shared_sensors": sorted(list(sensors_a.intersection(sensors_b))),
        "unique_sensors_left": sorted(list(sensors_a - sensors_b)),
        "unique_sensors_right": sorted(list(sensors_b - sensors_a)),
    }


def compute_aerosol_discrimination_curve(
    pyrolysis_material: str = "kapton_wire",
    dust_load_mg_m3: float = 1.2,
    gravity_g: float = 0.16,
) -> dict:
    material_modes = {
        "kapton_wire": {"cmd_0g": 0.42, "cmd_1g": 0.19, "sigma": 1.62},
        "teflon_insulation": {"cmd_0g": 0.34, "cmd_1g": 0.14, "sigma": 1.55},
        "silicone_elastomer": {"cmd_0g": 0.68, "cmd_1g": 0.31, "sigma": 1.70},
        "cotton_cellulose": {"cmd_0g": 0.55, "cmd_1g": 0.25, "sigma": 1.65},
    }
    cfg = material_modes.get(pyrolysis_material, material_modes["kapton_wire"])
    cmd_smoke = cfg["cmd_0g"] - (cfg["cmd_0g"] - cfg["cmd_1g"]) * (gravity_g**0.45)
    sigma_smoke = cfg["sigma"]

    cmd_dust = 2.85 * (1.0 - 0.22 * gravity_g)
    sigma_dust = 1.85

    diameters = [round(0.05 * (1.28**i), 3) for i in range(25)]
    smoke_dist = []
    dust_dist = []
    scattering_ratio = []

    for d in diameters:
        z_s = math.log(d / cmd_smoke) / math.log(sigma_smoke)
        val_s = math.exp(-0.5 * z_s * z_s) / (d * math.log(sigma_smoke) * math.sqrt(2 * math.pi))

        z_d = math.log(d / cmd_dust) / math.log(sigma_dust)
        val_d = (
            dust_load_mg_m3
            * math.exp(-0.5 * z_d * z_d)
            / (d * math.log(sigma_dust) * math.sqrt(2 * math.pi))
        )

        smoke_dist.append(round(val_s, 4))
        dust_dist.append(round(val_d, 4))
        blue_to_ir = round(2.4 / (1.0 + (d / 0.85) ** 1.7), 3)
        scattering_ratio.append(blue_to_ir)

    return {
        "pyrolysis_material": pyrolysis_material,
        "gravity_g": gravity_g,
        "count_median_diameter_smoke_um": round(cmd_smoke, 3),
        "count_median_diameter_dust_um": round(cmd_dust, 3),
        "optimal_cutoff_diameter_um": 1.15,
        "diameters_um": diameters,
        "smoke_density": smoke_dist,
        "regolith_dust_density": dust_dist,
        "dual_wavelength_ratio": scattering_ratio,
    }


def query_si_copilot(question: str, role: str, catalog: dict) -> dict:
    q_lower = question.lower()
    matched_invs = []
    for inv in catalog["investigations"]:
        if (
            inv["id"].lower() in q_lower
            or inv["fuel_state"].lower() in q_lower
            or inv["pillar"].lower() in q_lower
            or any(f.lower().split()[0] in q_lower for f in inv["fuels"])
        ):
            matched_invs.append(inv)

    if not matched_invs:
        if "moon" in q_lower or "lunar" in q_lower or "partial" in q_lower or "gravity" in q_lower:
            matched_invs = [
                i for i in catalog["investigations"] if i["id"] in ("E-FIELD", "BASS-II", "CFI-G", "DAFT-2")
            ]
        elif "smoke" in q_lower or "dust" in q_lower or "alarm" in q_lower or "artemis" in q_lower:
            matched_invs = [
                i for i in catalog["investigations"] if i["id"] in ("SAME", "SAME-R", "DAFT-2")
            ]
        elif "cool" in q_lower or "droplet" in q_lower or "suppress" in q_lower:
            matched_invs = [
                i for i in catalog["investigations"] if i["id"] in ("FLEX", "FLEX-2", "CFI", "CFI-G")
            ]
        else:
            matched_invs = catalog["investigations"][:4]

    role_prefix = {
        "engineer": "Engineering Design & Safety Margin Assessment",
        "professor": "Theoretical Scaling & Experimental Methodology Review",
        "student": "Foundational Microgravity Combustion Synthesis",
    }.get(role.lower(), "Engineering Design & Safety Margin Assessment")

    bullet_points = []
    for inv in matched_invs[:4]:
        bullet_points.append(
            f"[{inv['id']} / {inv['psi_id']}] ({inv['facility']} - {inv['hardware_insert']}, "
            f"{inv['o2_range_pct'][0]:.0f}-{inv['o2_range_pct'][1]:.0f}% O2): {inv['key_findings']}"
        )

    physics_note = (
        "In partial gravity (0.16g Lunar / 0.38g Martian habitats at 30-34% O2 and <=10.2 psi), "
        "buoyant convective inflow u_b = sqrt(g * beta * dT * L) prevents the diffusive oxygen starvation "
        "seen in 0g still air while operating below the 1g convective blowoff limit. "
        "Consequently, Limiting Oxygen Index (LOI) values drop below terrestrial NASA-STD-6001 baselines."
    )

    return {
        "role_mode": role_prefix,
        "query": question,
        "cited_investigations": [i["id"] for i in matched_invs[:4]],
        "evidence_summary": bullet_points,
        "physics_interpretation": physics_note,
    }

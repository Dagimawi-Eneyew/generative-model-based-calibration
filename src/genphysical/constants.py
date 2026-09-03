"""Fixed properties of the case-study building and of the calibration problem.

Everything here is a consequence of the building energy model in
``assets/idf/baseline_model_atlanta.idf`` (the DOE small-office reference
building for Atlanta, GA) and of the variable selection described in Sections
5.1-5.3.2 of the paper.  Nothing in this module is tunable; genuine
hyperparameters live in ``configs/``.

The calibration problem is, in the notation of Section 4.1,

    M : x -> y      with   x in R^9   (unobserved model inputs)
                           y in R^14  (physically observable model outputs)

so every array handled downstream has the layout

    columns  0 .. 8    unobserved model inputs  x
    columns  9 .. 22   observed model outputs   y_o
    columns 23 .. 36   clean copy of y_o        (DECI-Net reconstruction target only)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Building geometry
# ---------------------------------------------------------------------------
# Conditioned floor areas in m^2, read from the IDF.  The attic is unconditioned
# and is not part of the total, but its air temperature and humidity *are*
# observed, so it appears in the observation vector.
ZONE_AREAS_M2 = {
    "Attic": 567.9774,
    "Core_ZN": 149.6574,
    "Perimeter_ZN_1": 113.4500,
    "Perimeter_ZN_2": 67.3000,
    "Perimeter_ZN_3": 113.4500,
    "Perimeter_ZN_4": 67.3000,
}

#: Total conditioned floor area (Section 5.1: "a total floor area of 511 m2").
TOTAL_CONDITIONED_AREA_M2 = (
    ZONE_AREAS_M2["Core_ZN"]
    + ZONE_AREAS_M2["Perimeter_ZN_1"]
    + ZONE_AREAS_M2["Perimeter_ZN_2"]
    + ZONE_AREAS_M2["Perimeter_ZN_3"]
    + ZONE_AREAS_M2["Perimeter_ZN_4"]
)  # 511.1574 m^2

#: Section 5.2: perimeter zones 1 and 3 are geometrically identical, as are
#: zones 2 and 4.  The paper therefore estimates one set of inputs per group,
#: giving 9 unobserved inputs rather than 15.  Zone 3 mirrors zone 1 and zone 4
#: mirrors zone 2 when the estimated inputs are written back into the model.
ZONE_GROUPS = {
    "Core": ["Core_ZN"],
    "Zone-1,3": ["Perimeter_ZN_1", "Perimeter_ZN_3"],
    "Zone-2,4": ["Perimeter_ZN_2", "Perimeter_ZN_4"],
}

#: Representative zone of each group - the one whose EnergyPlus output columns
#: are read, and whose area is used to convert a density into an absolute load.
ZONE_GROUP_REPRESENTATIVE = {
    "Core": "Core_ZN",
    "Zone-1,3": "Perimeter_ZN_1",
    "Zone-2,4": "Perimeter_ZN_2",
}

# ---------------------------------------------------------------------------
# The nine unobserved model inputs, x  (Section 5.2)
# ---------------------------------------------------------------------------
# EnergyPlus output-variable names, in the canonical column order used
# throughout the package.  Occupant counts first, then lighting loads, then
# plug loads; within each triple the order is Core, Zone-1/3, Zone-2/4.
UNOBSERVED_INPUT_COLUMNS = [
    "CORE_ZN:Zone People Occupant Count [](Hourly)",
    "PERIMETER_ZN_1:Zone People Occupant Count [](Hourly)",
    "PERIMETER_ZN_2:Zone People Occupant Count [](Hourly)",
    "CORE_ZN_LIGHTS:Lights Electricity Rate [W](Hourly)",
    "PERIMETER_ZN_1_LIGHTS:Lights Electricity Rate [W](Hourly)",
    "PERIMETER_ZN_2_LIGHTS:Lights Electricity Rate [W](Hourly)",
    "CORE_ZN_MISCPLUG_EQUIP:Electric Equipment Electricity Rate [W](Hourly)",
    "PERIMETER_ZN_1_MISCPLUG_EQUIP:Electric Equipment Electricity Rate [W](Hourly)",
    "PERIMETER_ZN_2_MISCPLUG_EQUIP:Electric Equipment Electricity Rate [W](Hourly)",
]

#: Short, publication-ready labels for the same nine variables, matching the
#: row labels of Tables 4 and 5 and the axis labels of Figs. 7-15.
UNOBSERVED_INPUT_LABELS = [
    "Occupant Count: Core-Zone",
    "Occupant Count: Zone-1,3",
    "Occupant Count: Zone-2,4",
    "Lighting Load: Core-Zone",
    "Lighting Load: Zone-1,3",
    "Lighting Load: Zone-2,4",
    "Plug Load: Core-Zone",
    "Plug Load: Zone-1,3",
    "Plug Load: Zone-2,4",
]

#: Units of the nine inputs after post-processing (loads converted W -> kW).
UNOBSERVED_INPUT_UNITS = ["people"] * 3 + ["kW"] * 6

#: Indices of the six load columns, i.e. the ones converted from W to kW.
LOAD_COLUMN_INDICES = list(range(3, 9))

#: Watts per kilowatt - the divisor applied to the load columns.
W_PER_KW = 1000.0

# ---------------------------------------------------------------------------
# The fourteen observed model outputs, y_o  (Section 5.3.2)
# ---------------------------------------------------------------------------
# "fourteen output variables were identified as physically-observed model
#  outputs.  These include temperature and humidity measurements from five
#  zones, an additional attic zone, and readings of building-level electricity
#  and gas consumption."
OBSERVED_OUTPUT_COLUMNS = [
    "ATTIC:Zone Air Temperature [C](Hourly)",
    "ATTIC:Zone Air Relative Humidity [%](Hourly)",
    "CORE_ZN:Zone Air Temperature [C](Hourly)",
    "CORE_ZN:Zone Air Relative Humidity [%](Hourly)",
    "PERIMETER_ZN_1:Zone Air Temperature [C](Hourly)",
    "PERIMETER_ZN_1:Zone Air Relative Humidity [%](Hourly)",
    "PERIMETER_ZN_2:Zone Air Temperature [C](Hourly)",
    "PERIMETER_ZN_2:Zone Air Relative Humidity [%](Hourly)",
    "PERIMETER_ZN_3:Zone Air Temperature [C](Hourly)",
    "PERIMETER_ZN_3:Zone Air Relative Humidity [%](Hourly)",
    "PERIMETER_ZN_4:Zone Air Temperature [C](Hourly)",
    "PERIMETER_ZN_4:Zone Air Relative Humidity [%](Hourly)",
    "Electricity:Facility [J](Hourly)",
    "NaturalGas:Facility [J](Hourly)",
]

OBSERVED_OUTPUT_LABELS = [
    "Attic air temperature",
    "Attic relative humidity",
    "Core-Zone air temperature",
    "Core-Zone relative humidity",
    "Zone-1 air temperature",
    "Zone-1 relative humidity",
    "Zone-2 air temperature",
    "Zone-2 relative humidity",
    "Zone-3 air temperature",
    "Zone-3 relative humidity",
    "Zone-4 air temperature",
    "Zone-4 relative humidity",
    "Facility electricity",
    "Facility natural gas",
]

#: Index of ``Electricity:Facility`` *within the 14-element observation vector*.
#: Exposed so that individual observations can be shielded from random masking
#: through ``augmentation.protected_observations``.
ELECTRICITY_FACILITY_OBS_INDEX = 12
NATURAL_GAS_FACILITY_OBS_INDEX = 13

# ---------------------------------------------------------------------------
# Schedules driving the prototype building
# ---------------------------------------------------------------------------
#: Fractional occupancy / lighting / equipment schedules of the prototype model.
#: The test-set generator reads these once and multiplies them by the sampled
#: densities to build the hourly Schedule:File that drives the simulation
#: (Section 5.3.2: "dynamic scheduled inputs to alter the model inputs during
#: every hour of the simulation").
SCHEDULE_VALUE_COLUMNS = {
    "occupancy": "BLDG_OCC_SCH:Schedule Value [](Hourly)",
    "lighting": "BLDG_LIGHT_SCH:Schedule Value [](Hourly)",
    "equipment": "BLDG_EQUIP_SCH:Schedule Value [](Hourly)",
}

#: Names of the fractional schedules as they appear in the baseline IDF.
IDF_SCHEDULE_NAMES = {
    "occupancy": "BLDG_OCC_SCH",
    "lighting": "BLDG_LIGHT_SCH",
    "equipment": "BLDG_EQUIP_SCH",
}

# ---------------------------------------------------------------------------
# Derived dimensions - the single source of truth for every array layout
# ---------------------------------------------------------------------------
N_UNOBSERVED_INPUTS = len(UNOBSERVED_INPUT_COLUMNS)   # 9  = S in Section 4.1
N_OBSERVED_OUTPUTS = len(OBSERVED_OUTPUT_COLUMNS)     # 14 = D in Section 4.1

#: Column slice of the unobserved model inputs x.
INPUT_SLICE = slice(0, N_UNOBSERVED_INPUTS)
#: Column slice of the observed model outputs y_o.
OBSERVATION_SLICE = slice(N_UNOBSERVED_INPUTS, N_UNOBSERVED_INPUTS + N_OBSERVED_OUTPUTS)
#: Column slice of the clean y_o copy used as the DECI-Net reconstruction target.
CLEAN_TARGET_SLICE = slice(
    N_UNOBSERVED_INPUTS + N_OBSERVED_OUTPUTS,
    N_UNOBSERVED_INPUTS + 2 * N_OBSERVED_OUTPUTS,
)

#: Width of a cINN row (x and y_o).
N_COLUMNS_CINN = N_UNOBSERVED_INPUTS + N_OBSERVED_OUTPUTS            # 23
#: Width of a DECI-Net row (x, y_o and the clean y_o copy).
N_COLUMNS_DECINET = N_UNOBSERVED_INPUTS + 2 * N_OBSERVED_OUTPUTS     # 37

#: All columns read from an EnergyPlus run, in canonical order.
ALL_MODEL_COLUMNS = UNOBSERVED_INPUT_COLUMNS + OBSERVED_OUTPUT_COLUMNS

#: Hours in a non-leap year - the length of one annual hourly simulation.
HOURS_PER_YEAR = 8760

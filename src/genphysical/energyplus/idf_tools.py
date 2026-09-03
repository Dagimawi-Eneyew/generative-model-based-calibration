"""Reading and editing the building energy model.

Two distinct ways of driving the unobserved model inputs are needed, matching
the two datasets of Section 5.3.2:

1. **Constant densities per run** (training set).  Each of the 400 LHS samples
   becomes one IDF in which occupant density, lighting power density and
   equipment power density are held constant for the whole year.  See
   :func:`apply_densities` and :func:`generate_modified_idfs`.

2. **Hourly-varying absolute loads** (test set).  A single IDF whose People,
   Lights and ElectricEquipment objects are driven by ``Schedule:File`` objects
   that read one value per hour from a CSV, so the unobserved inputs change
   every hour of the simulation - "The 8760 LHS samples were used with a single
   IDF file, actual weather data for 2022, and dynamic scheduled inputs to alter
   the model inputs during every hour of the simulation."  See
   :func:`build_schedule_file_idf`.

   The same mechanism is reused in Section 5.10 to re-simulate the building with
   the calibrator's *estimated* inputs and compute CVRMSE.

``eppy`` is imported lazily so the rest of the package works without EnergyPlus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..constants import (
    HOURS_PER_YEAR,
    IDF_SCHEDULE_NAMES,
    SCHEDULE_VALUE_COLUMNS,
    ZONE_AREAS_M2,
    ZONE_GROUPS,
)
from ..utils.logging_utils import ProgressLogger, get_logger

logger = get_logger(__name__)

#: Conditioned zones, in the order their columns appear in the schedule CSV.
CONDITIONED_ZONES: List[str] = [
    "Core_ZN",
    "Perimeter_ZN_1",
    "Perimeter_ZN_2",
    "Perimeter_ZN_3",
    "Perimeter_ZN_4",
]

#: Which zone group supplies the values for each conditioned zone.  Zones 3 and
#: 4 mirror zones 1 and 2 (Section 5.2: the paper estimates one set of inputs
#: for each geometrically identical pair).
ZONE_TO_GROUP: Dict[str, str] = {
    "Core_ZN": "Core",
    "Perimeter_ZN_1": "Zone-1,3",
    "Perimeter_ZN_2": "Zone-2,4",
    "Perimeter_ZN_3": "Zone-1,3",
    "Perimeter_ZN_4": "Zone-2,4",
}

#: ScheduleTypeLimits used by the generated Schedule:File objects.  "Any Number"
#: is already defined in the prototype IDF and imposes no bounds, which is what
#: absolute occupant counts and wattages need.
ANY_NUMBER_LIMITS = "Any Number"

_LOAD_KINDS = ("occupancy", "lighting", "equipment")


# ---------------------------------------------------------------------------
# eppy plumbing
# ---------------------------------------------------------------------------
def _import_eppy():
    """Import eppy, with an actionable message if it is missing."""
    try:
        from eppy.modeleditor import IDF  # noqa: WPS433 (deliberate lazy import)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "eppy is required for the EnergyPlus stages.\n"
            "  Install it with:  pip install 'genphysical[energyplus]'\n"
            "  (or: pip install eppy)"
        ) from exc
    return IDF


_IDD_SET = False


def set_idd(idd_file: str | Path) -> None:
    """Register the Energy+.idd data dictionary with eppy.

    eppy allows this exactly once per process, so repeated calls with the same
    file are ignored and a call with a *different* file raises.
    """
    global _IDD_SET
    IDF = _import_eppy()
    idd_file = str(Path(idd_file))
    if _IDD_SET:
        current = getattr(IDF, "iddname", None)
        if current and Path(current) != Path(idd_file):
            raise RuntimeError(
                "eppy's IDD can only be set once per process. Already using "
                f"{current}, refusing to switch to {idd_file}."
            )
        return
    IDF.setiddname(idd_file)
    _IDD_SET = True
    logger.info("Using EnergyPlus IDD: %s", idd_file)


def load_idf(
    idf_file: str | Path, epw_file: str | Path, keep_daylight_saving: bool = False
):
    """Open an IDF/EPW pair as an eppy ``IDF`` object.

    Parameters
    ----------
    idf_file, epw_file:
        The model and its weather file.
    keep_daylight_saving:
        Leave the prototype's daylight-saving period in place.  Off by default -
        see :func:`disable_daylight_saving` for why every model in this pipeline
        runs on a single, unshifted time convention.
    """
    IDF = _import_eppy()
    idf = IDF(str(idf_file), str(epw_file))
    if not keep_daylight_saving:
        disable_daylight_saving(idf)
    return idf


def disable_daylight_saving(idf) -> bool:
    """Remove the daylight-saving period from a model.

    Returns
    -------
    bool
        Whether a daylight-saving object was present and removed.
    """
    objects = idf.idfobjects.get("RunPeriodControl:DaylightSavingTime".upper(), [])
    if not objects:
        return False
    for obj in list(objects):
        idf.removeidfobject(obj)
    logger.debug("Removed %d RunPeriodControl:DaylightSavingTime object(s)", len(objects))
    return True


def _zone_of(obj) -> str:
    """Return an internal-gain object's zone name.

    EnergyPlus 23.1 renamed the field from ``Zone_Name`` to
    ``Zone_or_ZoneList_or_Space_or_SpaceList_Name``; accept either so the code
    survives a version bump in both directions.
    """
    for attribute in (
        "Zone_or_ZoneList_or_Space_or_SpaceList_Name",
        "Zone_or_ZoneList_Name",
        "Zone_Name",
    ):
        value = getattr(obj, attribute, None)
        if value:
            return str(value)
    raise KeyError(f"Could not determine the zone of IDF object {obj.Name!r}.")


# ---------------------------------------------------------------------------
# 1. Constant-density variants (training set)
# ---------------------------------------------------------------------------
def apply_densities(
    idf,
    occupant_density: float,
    lighting_power_density: float,
    equipment_power_density: float,
) -> None:
    """Set the three area-dependent model inputs on every internal-gain object.

    Modifies ``idf`` in place, mirroring the sampled row onto all five
    conditioned zones.

    Parameters
    ----------
    occupant_density:
        Floor area per person, m2/person.  Larger means *fewer* occupants.
    lighting_power_density:
        Lighting power per floor area, W/m2.
    equipment_power_density:
        Plug-load power per floor area, W/m2.
    """
    for people in idf.idfobjects["People"]:
        people.Number_of_People_Calculation_Method = "Area/Person"
        people.Floor_Area_per_Person = occupant_density

    for lights in idf.idfobjects["Lights"]:
        lights.Design_Level_Calculation_Method = "Watts/Area"
        lights.Watts_per_Zone_Floor_Area = lighting_power_density

    for equipment in idf.idfobjects["ElectricEquipment"]:
        equipment.Design_Level_Calculation_Method = "Watts/Area"
        equipment.Watts_per_Zone_Floor_Area = equipment_power_density


def generate_modified_idfs(
    samples: pd.DataFrame,
    baseline_idf: str | Path,
    epw_file: str | Path,
    output_dir: str | Path,
    prefix: str = "modified",
) -> List[Path]:
    """Write one IDF per row of the LHS design (Section 5.3.2, training set).

    Parameters
    ----------
    samples:
        Design matrix from :mod:`genphysical.energyplus.sampling`, with the
        columns named by :attr:`SamplingConfig.parameter_names`.
    baseline_idf, epw_file:
        The prototype model and the weather file it will be run against.
    output_dir:
        Destination directory; created if absent.
    prefix:
        Filename stem, producing ``<prefix>_1.idf`` ... ``<prefix>_N.idf``.
        The trailing integer is how :mod:`.runner` names each run directory.

    Returns
    -------
    list of pathlib.Path
        The written IDF paths, in sample order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    progress = ProgressLogger(logger, len(samples), "Writing IDF variants")

    for position, (_, row) in enumerate(samples.iterrows()):
        idf = load_idf(baseline_idf, epw_file)
        apply_densities(
            idf,
            occupant_density=float(row["Occupant Density [m2/person]"]),
            lighting_power_density=float(row["Lighting Power Density [W/m2]"]),
            equipment_power_density=float(row["Equipment Power Density [W/m2]"]),
        )
        target = output_dir / f"{prefix}_{position + 1}.idf"
        idf.saveas(str(target))
        written.append(target)
        progress.update(position)

    logger.info("Wrote %d modified IDF files to %s", len(written), output_dir)
    return written


# ---------------------------------------------------------------------------
# 2. Hourly Schedule:File variant (test set and re-simulation)
# ---------------------------------------------------------------------------
def schedule_csv_columns() -> List[str]:
    """Column names of the hourly schedule CSV, in Schedule:File column order.

    Fifteen columns: occupant count, lighting load and equipment load for each
    of the five conditioned zones.  Column *numbers* in the generated
    ``Schedule:File`` objects are 1-based positions in this list.
    """
    columns: List[str] = []
    for kind, suffix in (
        ("occupancy", "Occupant_Count"),
        ("lighting", "Lighting_Load_W"),
        ("equipment", "Equipment_Load_W"),
    ):
        columns.extend(f"{zone}_{suffix}" for zone in CONDITIONED_ZONES)
    return columns


def extract_fractional_schedules(simulation_csv: str | Path) -> pd.DataFrame:
    """Read the prototype building's hourly fractional schedules from a run.

    The prototype model drives occupancy, lighting and plug loads with the
    fractional schedules ``BLDG_OCC_SCH``, ``BLDG_LIGHT_SCH`` and
    ``BLDG_EQUIP_SCH``.  The IDF reports them through ``Output:Variable, *,
    Schedule Value, Hourly``, so a single reference simulation of the untouched
    baseline yields all 8760 hourly fractions.

    Multiplying these fractions by a sampled density reproduces exactly what
    EnergyPlus would have computed internally, which is what lets the
    ``Schedule:File`` variant drive absolute values hour by hour.

    Returns
    -------
    pandas.DataFrame
        8760 rows with columns ``occupancy``, ``lighting`` and ``equipment``.
    """
    frame = pd.read_csv(simulation_csv)
    missing = [
        column
        for column in SCHEDULE_VALUE_COLUMNS.values()
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(
            "Reference simulation output is missing schedule columns "
            f"{missing}. The IDF needs 'Output:Variable, *, Schedule Value, "
            "Hourly;' for the fractional schedules to be reported."
        )

    schedules = frame[list(SCHEDULE_VALUE_COLUMNS.values())].iloc[:HOURS_PER_YEAR]
    schedules.columns = list(SCHEDULE_VALUE_COLUMNS.keys())
    if len(schedules) != HOURS_PER_YEAR:
        raise ValueError(
            f"Expected {HOURS_PER_YEAR} hourly schedule rows, got {len(schedules)}. "
            "Was the reference run a full annual simulation?"
        )
    return schedules.reset_index(drop=True)


def densities_to_hourly_loads(
    samples: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    """Turn an 8760-row LHS design into hourly absolute per-zone loads.

    For hour ``t`` and zone ``z`` with floor area ``A_z``:

        occupant count  = (A_z / occupant_density[t])       * occupancy_fraction[t]
        lighting load W =  A_z * lighting_power_density[t]  * lighting_fraction[t]
        equipment load W = A_z * equipment_power_density[t] * equipment_fraction[t]

    These are exactly the quantities EnergyPlus reports as ``Zone People
    Occupant Count``, ``Lights Electricity Rate`` and ``Electric Equipment
    Electricity Rate``, so feeding them back through a ``Schedule:File`` with a
    unit design level reproduces them to floating-point precision.  That
    identity is asserted by :func:`verify_schedule_roundtrip`.

    Returns
    -------
    pandas.DataFrame
        8760 rows, 15 columns, ordered as :func:`schedule_csv_columns`.
    """
    if len(samples) != len(schedules):
        raise ValueError(
            f"samples has {len(samples)} rows but schedules has {len(schedules)}; "
            "one LHS sample is required per simulated hour."
        )

    occupant_density = samples["Occupant Density [m2/person]"].to_numpy()
    lighting_density = samples["Lighting Power Density [W/m2]"].to_numpy()
    equipment_density = samples["Equipment Power Density [W/m2]"].to_numpy()

    occupancy_fraction = schedules["occupancy"].to_numpy()
    lighting_fraction = schedules["lighting"].to_numpy()
    equipment_fraction = schedules["equipment"].to_numpy()

    data: Dict[str, np.ndarray] = {}
    for zone in CONDITIONED_ZONES:
        area = ZONE_AREAS_M2[zone]
        data[f"{zone}_Occupant_Count"] = (area / occupant_density) * occupancy_fraction
    for zone in CONDITIONED_ZONES:
        area = ZONE_AREAS_M2[zone]
        data[f"{zone}_Lighting_Load_W"] = (area * lighting_density) * lighting_fraction
    for zone in CONDITIONED_ZONES:
        area = ZONE_AREAS_M2[zone]
        data[f"{zone}_Equipment_Load_W"] = (
            area * equipment_density
        ) * equipment_fraction

    return pd.DataFrame(data, columns=schedule_csv_columns())


def group_predictions_to_hourly_loads(
    predictions: np.ndarray,
    clip_at_zero: bool = True,
) -> pd.DataFrame:
    """Expand the calibrator's 9 estimated inputs into the 15 schedule columns.

    The calibrator estimates one occupant count, lighting load and plug load per
    *zone group* (Core, Zone-1/3, Zone-2/4).  Perimeter zones 3 and 4 receive
    the values estimated for zones 1 and 2 respectively, which is exactly how
    they were generated (identical geometry, identical densities).

    Parameters
    ----------
    predictions:
        ``(n_hours, 9)`` array in the canonical input order: occupant counts for
        (Core, Zone-1/3, Zone-2/4), then lighting loads in **kW**, then plug
        loads in **kW**.
    clip_at_zero:
        Clip negatives to zero.  The posterior mean of an occupant count or a
        wattage can be slightly negative, which EnergyPlus rejects; the original
        study applied the same clipping before re-simulating.

    Returns
    -------
    pandas.DataFrame
        8760 x 15, ready for :func:`write_schedule_csv`.  Loads are converted
        back from kW to W, the unit the model expects.
    """
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] != 9:
        raise ValueError(
            f"Expected predictions with shape (n_hours, 9), got {predictions.shape}."
        )
    if clip_at_zero:
        predictions = np.clip(predictions, 0.0, None)

    group_names = list(ZONE_GROUPS)  # ["Core", "Zone-1,3", "Zone-2,4"]
    group_index = {name: position for position, name in enumerate(group_names)}

    data: Dict[str, np.ndarray] = {}
    # Occupant counts: columns 0-2, already in people.
    for zone in CONDITIONED_ZONES:
        offset = group_index[ZONE_TO_GROUP[zone]]
        data[f"{zone}_Occupant_Count"] = predictions[:, offset]
    # Lighting loads: columns 3-5, kW -> W.
    for zone in CONDITIONED_ZONES:
        offset = 3 + group_index[ZONE_TO_GROUP[zone]]
        data[f"{zone}_Lighting_Load_W"] = predictions[:, offset] * 1000.0
    # Plug loads: columns 6-8, kW -> W.
    for zone in CONDITIONED_ZONES:
        offset = 6 + group_index[ZONE_TO_GROUP[zone]]
        data[f"{zone}_Equipment_Load_W"] = predictions[:, offset] * 1000.0

    return pd.DataFrame(data, columns=schedule_csv_columns())


def write_schedule_csv(loads: pd.DataFrame, path: str | Path) -> Path:
    """Write the hourly schedule CSV that ``Schedule:File`` objects read.

    The file has a single header row (matched by ``Rows to Skip at Top = 1``)
    and one row per simulated hour.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    expected = schedule_csv_columns()
    if list(loads.columns) != expected:
        raise ValueError(
            "Schedule CSV columns must be exactly, and in the order of, "
            f"{expected}; got {list(loads.columns)}."
        )
    if len(loads) != HOURS_PER_YEAR:
        raise ValueError(
            f"Schedule CSV must hold {HOURS_PER_YEAR} hourly rows, got {len(loads)}."
        )
    if (loads.to_numpy() < 0).any():
        raise ValueError(
            "Schedule CSV contains negative occupant counts or loads, which "
            "EnergyPlus rejects. Clip the predictions before writing."
        )

    loads.to_csv(path, index=False)
    logger.info("Wrote hourly schedule file: %s", path)
    return path


def build_schedule_file_idf(
    baseline_idf: str | Path,
    epw_file: str | Path,
    schedule_csv: str | Path,
    output_idf: str | Path,
    n_hours: int = HOURS_PER_YEAR,
):
    """Convert the prototype model to one driven by an hourly ``Schedule:File``.

    Each People / Lights / ElectricEquipment object is switched from a
    density-based design level with a fractional schedule to a **unit** design
    level with an **absolute** schedule:

        People            "People" method,          Number of People = 1
        Lights            "LightingLevel" method,   Lighting Level   = 1 W
        ElectricEquipment "EquipmentLevel" method,  Design Level     = 1 W

    Multiplying a unit design level by the schedule makes the reported
    ``Zone People Occupant Count`` / ``Lights Electricity Rate`` / ``Electric
    Equipment Electricity Rate`` equal to the CSV value for that hour, so the
    unobserved model inputs can be set directly rather than through a density.

    This is what makes two things possible: the 8760-sample test set of Section
    5.3.2, in which the inputs change every hour, and the re-simulation of
    Section 5.10, in which the *estimated* inputs are pushed back into the model
    to measure CVRMSE.

    Parameters
    ----------
    baseline_idf, epw_file:
        The prototype model and its weather file.
    schedule_csv:
        Hourly values from :func:`write_schedule_csv`.  Referenced by absolute
        path, since EnergyPlus resolves it relative to the run directory.
    output_idf:
        Where to write the converted model.
    n_hours:
        Value of the ``Number of Hours of Data`` field.

    Returns
    -------
    eppy.modeleditor.IDF
        The converted model (also saved to ``output_idf``).
    """
    schedule_csv = Path(schedule_csv).resolve()
    output_idf = Path(output_idf)
    output_idf.parent.mkdir(parents=True, exist_ok=True)

    idf = load_idf(baseline_idf, epw_file)
    _ensure_any_number_limits(idf)

    columns = schedule_csv_columns()
    column_number = {name: position + 1 for position, name in enumerate(columns)}

    def add_schedule(zone: str, suffix: str) -> str:
        """Create one Schedule:File for a zone/quantity and return its name."""
        csv_column = f"{zone}_{suffix}"
        schedule_name = f"GENPHYSICAL_{csv_column}".upper()
        idf.newidfobject(
            "Schedule:File".upper(),
            Name=schedule_name,
            Schedule_Type_Limits_Name=ANY_NUMBER_LIMITS,
            File_Name=str(schedule_csv),
            Column_Number=column_number[csv_column],
            Rows_to_Skip_at_Top=1,          # the CSV header
            Number_of_Hours_of_Data=n_hours,
            Column_Separator="Comma",
            Interpolate_to_Timestep="No",   # hold each hourly value constant
            Minutes_per_Item=60,
        )
        return schedule_name

    # --- People -------------------------------------------------------------
    for people in idf.idfobjects["People"]:
        zone = _zone_of(people)
        if zone not in CONDITIONED_ZONES:
            continue
        people.Number_of_People_Schedule_Name = add_schedule(zone, "Occupant_Count")
        people.Number_of_People_Calculation_Method = "People"
        people.Number_of_People = 1.0
        people.People_per_Floor_Area = ""
        people.Floor_Area_per_Person = ""

    # --- Lights -------------------------------------------------------------
    for lights in idf.idfobjects["Lights"]:
        zone = _zone_of(lights)
        if zone not in CONDITIONED_ZONES:
            continue
        lights.Schedule_Name = add_schedule(zone, "Lighting_Load_W")
        lights.Design_Level_Calculation_Method = "LightingLevel"
        lights.Lighting_Level = 1.0
        lights.Watts_per_Zone_Floor_Area = ""
        lights.Watts_per_Person = ""

    # --- ElectricEquipment --------------------------------------------------
    for equipment in idf.idfobjects["ElectricEquipment"]:
        zone = _zone_of(equipment)
        if zone not in CONDITIONED_ZONES:
            continue
        equipment.Schedule_Name = add_schedule(zone, "Equipment_Load_W")
        equipment.Design_Level_Calculation_Method = "EquipmentLevel"
        equipment.Design_Level = 1.0
        equipment.Watts_per_Zone_Floor_Area = ""
        equipment.Watts_per_Person = ""

    idf.saveas(str(output_idf))
    logger.info(
        "Wrote Schedule:File-driven model %s (reading %s)", output_idf, schedule_csv
    )
    return idf


def _ensure_any_number_limits(idf) -> None:
    """Make sure an unbounded ScheduleTypeLimits exists for the new schedules.

    The prototype IDF already defines "Any Number"; this only adds it if a
    different starting model does not.
    """
    existing = {
        str(limits.Name).strip().lower()
        for limits in idf.idfobjects["ScheduleTypeLimits"]
    }
    if ANY_NUMBER_LIMITS.lower() not in existing:
        idf.newidfobject(
            "ScheduleTypeLimits".upper(),
            Name=ANY_NUMBER_LIMITS,
            Numeric_Type="CONTINUOUS",
        )


def verify_schedule_roundtrip(
    schedule_csv: str | Path,
    simulation_csv: str | Path,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Assert that the simulation reproduced the schedule it was given.

    Confirms that a unit design level times the ``Schedule:File`` value really
    does come back out as the reported occupant count and load, i.e. that the
    unobserved model inputs were set exactly as intended.

    Raises
    ------
    AssertionError
        If any zone's reported values differ from the schedule it was fed.
    """
    scheduled = pd.read_csv(schedule_csv)
    simulated = pd.read_csv(simulation_csv)

    checks = [
        ("{zone}_Occupant_Count", "{ZONE}:Zone People Occupant Count [](Hourly)"),
        (
            "{zone}_Lighting_Load_W",
            "{ZONE}_LIGHTS:Lights Electricity Rate [W](Hourly)",
        ),
        (
            "{zone}_Equipment_Load_W",
            "{ZONE}_MISCPLUG_EQUIP:Electric Equipment Electricity Rate [W](Hourly)",
        ),
    ]

    mismatched: List[str] = []
    for zone in CONDITIONED_ZONES:
        for csv_template, output_template in checks:
            csv_column = csv_template.format(zone=zone)
            output_column = output_template.format(ZONE=zone.upper())
            if output_column not in simulated.columns:
                mismatched.append(f"{output_column} (absent from simulation output)")
                continue
            if not np.allclose(
                scheduled[csv_column].to_numpy(),
                simulated[output_column].to_numpy()[: len(scheduled)],
                rtol=rtol,
                atol=atol,
            ):
                mismatched.append(f"{csv_column} != {output_column}")

    if mismatched:
        raise AssertionError(
            "The Schedule:File-driven simulation did not reproduce its input "
            "schedule for:\n  " + "\n  ".join(mismatched)
        )
    logger.info(
        "Schedule round-trip verified: all %d zones reproduced their scheduled "
        "occupant counts and loads.",
        len(CONDITIONED_ZONES),
    )

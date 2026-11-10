"""Deterministic, disclosed demographic cohort selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

from .contracts import CohortUse, Demographics

AGE_BANDS = ("18_19", "20_29", "30_39", "40_49", "50_59", "60_69", "70_plus")
BAND_BOUNDS = {
    "18_19": (18, 19),
    "20_29": (20, 29),
    "30_39": (30, 39),
    "40_49": (40, 49),
    "50_59": (50, 59),
    "60_69": (60, 69),
    "70_plus": (70, 130),
}
BINARY_SEX_PREFIX = {"female": "f", "male": "m"}
GENERIC_ETHNICITIES = frozenset(
    {"another", "another_or_not_listed", "mixed", "multiethnic", "prefer_not_to_say"}
)
DISCLOSED_BROADER_ETHNICITY = {
    "east_asian": "broader_asian",
    "southeast_asian": "broader_asian",
}


class NoCompatibleCohort(LookupError):
    """No powered cohort survives the allowed fallback chain."""


def age_band(age: int) -> str:
    if isinstance(age, bool) or not isinstance(age, int):
        raise ValueError("age must be a whole number")
    if age < 18:
        raise ValueError("analysis is available only to adults 18 and older")
    if age <= 19:
        return "18_19"
    if age >= 70:
        return "70_plus"
    lower = (age // 10) * 10
    return f"{lower}_{lower + 9}"


def _normalize_band(raw: str) -> str:
    value = raw.strip().lower().replace("–", "-").replace("—", "-")
    value = value.replace(" years", "").replace("aged ", "")
    value = value.replace("70+", "70_plus").replace("70 plus", "70_plus")
    value = re.sub(r"\s+to\s+", "_to_", value)
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def _normalize_sex(raw: str) -> str:
    value = str(raw).strip().lower()
    if value in {"f", "female", "woman", "women"}:
        return "female"
    if value in {"m", "male", "man", "men"}:
        return "male"
    if value in {"all", "all_sexes", "pooled"}:
        return "all"
    raise ValueError(f"unsupported cohort sex label: {raw}")


def _band_components(band: str) -> tuple[str, ...]:
    normalized = _normalize_band(band)
    if normalized in AGE_BANDS:
        return (normalized,)
    if "_to_" in normalized:
        left, right = normalized.split("_to_", 1)
        if left in AGE_BANDS and right in AGE_BANDS:
            lo = AGE_BANDS.index(left)
            hi = AGE_BANDS.index(right)
            if lo <= hi:
                return AGE_BANDS[lo : hi + 1]
    return ()


def _band_distance(age: int, band: str) -> int:
    components = _band_components(band)
    if not components:
        return 10_000
    distances = []
    for component in components:
        low, high = BAND_BOUNDS[component]
        if low <= age <= high:
            distances.append(0)
        elif age < low:
            distances.append(low - age)
        else:
            distances.append(age - high)
    return min(distances)


def _display_band(band: str) -> str:
    if band == "70_plus":
        return "70+"
    return band.replace("_to_", " to ").replace("_", "-")


@dataclass(frozen=True)
class _Entry:
    cohort_id: str
    n: int
    sex: str
    ethnicity: str
    age_band: str
    label: str
    merged: bool
    meets_floor: bool
    members: tuple[str, ...]
    virtual: bool = False


class CohortResolver:
    """Apply the fixed exact → broader ethnicity → nearest age chain."""

    def __init__(self, inventory: Mapping[str, Mapping[str, Any]], *, floor: int,
                 allow_virtual_pools: bool = True) -> None:
        if floor <= 0:
            raise ValueError("cohort floor must be positive")
        self.floor = int(floor)
        self._entries = self._normalize_inventory(inventory)
        if allow_virtual_pools:
            self._add_virtual_pools()

    def resolve(self, demographics: Demographics) -> CohortUse:
        band = age_band(demographics.age)
        ethnicity = self._single_exact_ethnicity(demographics)
        requested_id = self._requested_id(demographics, ethnicity, band)
        requested_label = self._requested_label(demographics, band)

        if demographics.sex in BINARY_SEX_PREFIX:
            prefix = BINARY_SEX_PREFIX[demographics.sex]
            if ethnicity:
                exact = self._entries.get(f"{prefix}_{ethnicity}_{band}")
                if self._powered(exact):
                    return self._result(
                        requested_id,
                        requested_label,
                        exact,
                        "exact",
                    )

                broader_label = DISCLOSED_BROADER_ETHNICITY.get(ethnicity)
                if broader_label:
                    regional = self._entries.get(f"{prefix}_{broader_label}_{band}")
                    if self._powered(regional):
                        return self._result(
                            requested_id,
                            requested_label,
                            regional,
                            f"same_age_sex_{broader_label}",
                        )

            broader = self._entries.get(f"{prefix}_all_ethnicities_{band}")
            if self._powered(broader):
                return self._result(
                    requested_id,
                    requested_label,
                    broader,
                    "same_age_sex_broader_ethnicity",
                )

            if ethnicity:
                nearest = self._nearest(
                    age=demographics.age,
                    sex=demographics.sex,
                    ethnicity=ethnicity,
                    exclude_band=band,
                )
                if nearest:
                    return self._result(
                        requested_id,
                        requested_label,
                        nearest,
                        "nearest_age_same_sex_ethnicity",
                    )

            nearest_broad = self._nearest(
                age=demographics.age,
                sex=demographics.sex,
                ethnicity="all_ethnicities",
                exclude_band=band,
            )
            if nearest_broad:
                return self._result(
                    requested_id,
                    requested_label,
                    nearest_broad,
                    "nearest_age_same_sex_broader_ethnicity",
                )
        else:
            if ethnicity:
                all_sex = self._entries.get(f"all_{ethnicity}_{band}")
                if self._powered(all_sex):
                    return self._result(
                        requested_id,
                        requested_label,
                        all_sex,
                        "all_sex_same_age_ethnicity",
                    )

                broader_label = DISCLOSED_BROADER_ETHNICITY.get(ethnicity)
                if broader_label:
                    all_sex_regional = self._entries.get(
                        f"all_{broader_label}_{band}"
                    )
                    if self._powered(all_sex_regional):
                        return self._result(
                            requested_id,
                            requested_label,
                            all_sex_regional,
                            f"all_sex_same_age_{broader_label}",
                        )

            all_sex_broad = self._entries.get(f"all_all_ethnicities_{band}")
            if self._powered(all_sex_broad):
                return self._result(
                    requested_id,
                    requested_label,
                    all_sex_broad,
                    "all_sex_same_age_broader_ethnicity",
                )

            if ethnicity:
                nearest_all = self._nearest(
                    age=demographics.age,
                    sex="all",
                    ethnicity=ethnicity,
                    exclude_band=band,
                )
                if nearest_all:
                    return self._result(
                        requested_id,
                        requested_label,
                        nearest_all,
                        "all_sex_nearest_age_ethnicity",
                    )

            nearest_all_broad = self._nearest(
                age=demographics.age,
                sex="all",
                ethnicity="all_ethnicities",
                exclude_band=band,
            )
            if nearest_all_broad:
                return self._result(
                    requested_id,
                    requested_label,
                    nearest_all_broad,
                    "all_sex_nearest_age_broader_ethnicity",
                )

        raise NoCompatibleCohort(
            f"no powered comparison cohort is available for {demographics.sex}, "
            f"age {_display_band(band)}, with the allowed fallback chain"
        )

    def members_for(self, cohort_id: str) -> tuple[str, ...]:
        entry = self._entries.get(cohort_id)
        if entry is None:
            raise KeyError(cohort_id)
        return entry.members

    def _normalize_inventory(
        self,
        inventory: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, _Entry]:
        entries: dict[str, _Entry] = {}
        for cohort_id, record in inventory.items():
            n = record.get("n", record.get("sample_size"))
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise ValueError(f"cohort {cohort_id} has an invalid sample size")
            sex = _normalize_sex(record.get("sex", ""))
            ethnicity = str(record.get("ethnicity", record.get("race", ""))).strip().lower()
            if not ethnicity:
                raise ValueError(f"cohort {cohort_id} has no ethnicity label")
            band = _normalize_band(str(record.get("age_band", "")))
            if not _band_components(band):
                raise ValueError(f"cohort {cohort_id} has an invalid age band")
            members = tuple(str(item) for item in record.get("members", (cohort_id,)))
            entries[cohort_id] = _Entry(
                cohort_id=cohort_id,
                n=n,
                sex=sex,
                ethnicity=ethnicity,
                age_band=band,
                label=str(record.get("label") or cohort_id),
                merged=bool(record.get("merged", "_to_" in band)),
                meets_floor=bool(record.get("meets_floor", n >= self.floor)),
                members=members,
                virtual=bool(record.get("virtual", False)),
            )
        return entries

    def _add_virtual_pools(self) -> None:
        base = [
            entry
            for entry in self._entries.values()
            if not entry.merged
            and not entry.virtual
            and entry.age_band in AGE_BANDS
            and entry.sex in BINARY_SEX_PREFIX
            and entry.ethnicity != "all_ethnicities"
        ]
        same_sex: dict[tuple[str, str], list[_Entry]] = defaultdict(list)
        all_sex: dict[tuple[str, str], list[_Entry]] = defaultdict(list)
        all_sex_broad: dict[str, list[_Entry]] = defaultdict(list)
        for entry in base:
            same_sex[(entry.sex, entry.age_band)].append(entry)
            all_sex[(entry.ethnicity, entry.age_band)].append(entry)
            all_sex_broad[entry.age_band].append(entry)

        for (sex, band), entries in same_sex.items():
            prefix = BINARY_SEX_PREFIX[sex]
            self._add_virtual(
                f"{prefix}_all_ethnicities_{band}",
                entries,
                sex=sex,
                ethnicity="all_ethnicities",
                band=band,
                label=f"{sex}, all available ethnicity labels, age {_display_band(band)}",
            )
        for (ethnicity, band), entries in all_sex.items():
            self._add_virtual(
                f"all_{ethnicity}_{band}",
                entries,
                sex="all",
                ethnicity=ethnicity,
                band=band,
                label=f"all available sex labels, {ethnicity}, age {_display_band(band)}",
            )
        for band, entries in all_sex_broad.items():
            self._add_virtual(
                f"all_all_ethnicities_{band}",
                entries,
                sex="all",
                ethnicity="all_ethnicities",
                band=band,
                label=(
                    "all available sex and ethnicity labels, "
                    f"age {_display_band(band)}"
                ),
            )

    def _add_virtual(
        self,
        cohort_id: str,
        entries: list[_Entry],
        *,
        sex: str,
        ethnicity: str,
        band: str,
        label: str,
    ) -> None:
        if cohort_id in self._entries or not entries:
            return
        members = tuple(
            dict.fromkeys(member for entry in entries for member in entry.members)
        )
        n = sum(entry.n for entry in entries)
        self._entries[cohort_id] = _Entry(
            cohort_id=cohort_id,
            n=n,
            sex=sex,
            ethnicity=ethnicity,
            age_band=band,
            label=label,
            merged=False,
            meets_floor=n >= self.floor,
            members=members,
            virtual=True,
        )

    def _nearest(
        self,
        *,
        age: int,
        sex: str,
        ethnicity: str,
        exclude_band: str,
    ) -> _Entry | None:
        candidates = [
            entry
            for entry in self._entries.values()
            if entry.sex == sex
            and entry.ethnicity == ethnicity
            and entry.age_band != exclude_band
            and self._powered(entry)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda entry: (
                _band_distance(age, entry.age_band),
                len(_band_components(entry.age_band)),
                entry.age_band,
                entry.cohort_id,
            ),
        )

    def _powered(self, entry: _Entry | None) -> bool:
        return bool(entry and entry.meets_floor and entry.n >= self.floor)

    @staticmethod
    def _single_exact_ethnicity(demographics: Demographics) -> str | None:
        if len(demographics.ethnicities) != 1:
            return None
        ethnicity = demographics.ethnicities[0]
        if ethnicity in GENERIC_ETHNICITIES:
            return None
        return ethnicity

    @staticmethod
    def _requested_id(
        demographics: Demographics,
        ethnicity: str | None,
        band: str,
    ) -> str:
        prefix = BINARY_SEX_PREFIX.get(demographics.sex, "x")
        ethnicity_id = ethnicity or "multi_or_unlisted"
        return f"{prefix}_{ethnicity_id}_{band}"

    @staticmethod
    def _requested_label(demographics: Demographics, band: str) -> str:
        ethnicities = ", ".join(item.replace("_", " ") for item in demographics.ethnicities)
        return f"{demographics.sex}, {ethnicities}, age {_display_band(band)}"

    @staticmethod
    def _result(
        requested_id: str,
        requested_label: str,
        entry: _Entry,
        fallback_level: str,
    ) -> CohortUse:
        return CohortUse(
            requested_id=requested_id,
            used_id=entry.cohort_id,
            fallback_level=fallback_level,
            sample_size=entry.n,
            requested_label=requested_label,
            used_label=entry.label,
            member_ids=entry.members,
        )

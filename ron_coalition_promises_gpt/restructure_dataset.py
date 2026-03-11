"""
Dataset restructuring script for ron_coalition_promises_gpt.

This script:
1. Scans all party platform JSON files inside the ron/ source directory.
2. Detects duplicate party platforms (same party, same election).
3. Keeps the version with the most extracted promises.
4. Moves the canonical version into the correct election period directory.
5. Merges coalition treaty + valitsuskava promises per government.
6. Produces the final layered dataset structure.

Usage:
    python restructure_dataset.py [--source-dir PATH] [--dry-run]

If --source-dir is not provided, the script looks for a 'ron/' directory
in the same folder as this script.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ministers_data import ELECTION_PERIODS, GOVERNMENTS, Government
from merge_coalition_promises import merge_all_governments


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_json(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file, return None on failure."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [WARN] Could not load {filepath}: {e}")
        return None


def save_json(data: Any, filepath: Path) -> None:
    """Write JSON to file with UTF-8 encoding."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def count_promises(data: Optional[Dict[str, Any]]) -> int:
    """Count promises in a loaded JSON document."""
    if data is None:
        return 0
    if isinstance(data, list):
        return len(data)
    for key in ("promises", "lubadused", "items"):
        if key in data and isinstance(data[key], list):
            return len(data[key])
    if "data" in data and isinstance(data["data"], dict):
        for key in ("promises", "lubadused", "items"):
            if key in data["data"] and isinstance(data["data"][key], list):
                return len(data["data"][key])
    return 0


def identify_party_platform(data: Dict[str, Any], filename: str) -> Optional[Tuple[str, int]]:
    """
    Attempt to identify the party name and election year from a JSON document
    or its filename.

    Returns (party_name, election_year) or None if unidentifiable.
    """
    party = data.get("party") or data.get("erakond") or data.get("party_name")
    year = data.get("election_year") or data.get("valimiste_aasta")

    # Try to extract from filename: e.g. "Reformierakond_2019.json"
    if not party or not year:
        stem = Path(filename).stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            name_part, year_part = parts
            if not party:
                party = name_part.replace("_", " ")
            if not year:
                try:
                    year = int(year_part)
                except ValueError:
                    pass

    if party and year:
        return (str(party), int(year))
    return None


def election_year_to_period(year: int) -> Optional[str]:
    """Map an election year to the corresponding election period key."""
    year_to_period = {
        2003: "2003-2007",
        2007: "2007-2015",
        2011: "2007-2015",
        2015: "2015-2019",
        2019: "2019-2023",
    }
    return year_to_period.get(year)


# ---------------------------------------------------------------------------
# Layer 1: Election periods & party platforms
# ---------------------------------------------------------------------------

def scan_party_platforms(source_dir: Path) -> List[Dict[str, Any]]:
    """
    Recursively scan source_dir for JSON files that look like party platforms.

    Returns a list of dicts with metadata about each file found.
    """
    results = []
    for json_file in source_dir.rglob("*.json"):
        data = load_json(json_file)
        if data is None:
            continue

        identity = identify_party_platform(data, json_file.name)
        if identity is None:
            # Check if it looks like a party platform by structure
            promise_count = count_promises(data)
            if promise_count == 0:
                continue
            # Still include it but mark as unidentified
            results.append({
                "path": json_file,
                "party": None,
                "election_year": None,
                "election_period": None,
                "promise_count": promise_count,
                "data": data,
            })
        else:
            party, year = identity
            period = election_year_to_period(year)
            results.append({
                "path": json_file,
                "party": party,
                "election_year": year,
                "election_period": period,
                "promise_count": count_promises(data),
                "data": data,
            })

    return results


def deduplicate_platforms(platforms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group platforms by (party, election_year) and keep only the version
    with the most promises per group.

    Returns the deduplicated list.
    """
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)

    unidentified = []
    for p in platforms:
        if p["party"] is None or p["election_year"] is None:
            unidentified.append(p)
            continue
        key = (p["party"].lower().strip(), p["election_year"])
        groups[key].append(p)

    canonical = []
    for key, entries in groups.items():
        if len(entries) == 1:
            canonical.append(entries[0])
        else:
            # Sort by promise count descending; keep the one with most promises
            entries.sort(key=lambda x: x["promise_count"], reverse=True)
            best = entries[0]
            duplicates = entries[1:]
            print(f"  [DEDUP] '{key[0]}' ({key[1]}): keeping {best['path'].name} "
                  f"({best['promise_count']} promises), "
                  f"removing {len(duplicates)} duplicate(s)")
            for dup in duplicates:
                print(f"          - {dup['path']} ({dup['promise_count']} promises)")
            canonical.append(best)

    # Include unidentified files as-is (no dedup key available)
    canonical.extend(unidentified)
    return canonical


def place_platforms_in_periods(
    platforms: List[Dict[str, Any]],
    base_dir: Path,
    dry_run: bool = False,
) -> None:
    """
    Copy each canonical party platform JSON into the correct
    election period directory.
    """
    for p in platforms:
        period = p["election_period"]
        if period is None:
            print(f"  [SKIP] No election period for: {p['path']}")
            continue

        target_dir = base_dir / "election_periods" / period
        target_file = target_dir / p["path"].name

        if dry_run:
            print(f"  [DRY-RUN] Would copy {p['path']} -> {target_file}")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(p["path"]), str(target_file))
            print(f"  [COPY] {p['path'].name} -> {target_file}")


def update_election_json_files(
    platforms: List[Dict[str, Any]],
    base_dir: Path,
    dry_run: bool = False,
) -> None:
    """
    Update the Elections_<year>.json files to include references to the
    party platforms placed in that election period directory.
    """
    # Group platforms by election period
    period_platforms: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in platforms:
        if p["election_period"]:
            period_platforms[p["election_period"]].append(p)

    for period, plist in period_platforms.items():
        period_dir = base_dir / "election_periods" / period

        # Find existing Elections_*.json files in this directory
        election_files = list(period_dir.glob("Elections_*.json"))

        for ef in election_files:
            data = load_json(ef)
            if data is None:
                continue

            # Build party_platforms entries
            platform_refs = []
            for p in plist:
                ref = {
                    "party": p["party"] or "unknown",
                    "election_year": p["election_year"],
                    "filename": p["path"].name,
                    "promise_count": p["promise_count"],
                }
                # Only include platforms matching this election file's year
                file_year = data.get("election_year")
                if file_year and p["election_year"] and p["election_year"] == file_year:
                    platform_refs.append(ref)

            if platform_refs:
                data["party_platforms"] = platform_refs
                if dry_run:
                    print(f"  [DRY-RUN] Would update {ef} with {len(platform_refs)} platform refs")
                else:
                    save_json(data, ef)
                    print(f"  [UPDATE] {ef.name}: {len(platform_refs)} platform references")


# ---------------------------------------------------------------------------
# Layer 2: Coalition agreements (delegation to merge script)
# ---------------------------------------------------------------------------

def generate_coalition_agreements(
    source_dir: Path,
    base_dir: Path,
    dry_run: bool = False,
) -> None:
    """
    Generate merged coalition agreement JSONs for each government.

    Delegates to merge_coalition_promises.merge_all_governments().
    """
    output_dir = base_dir / "coalition_agreements"

    if dry_run:
        print(f"  [DRY-RUN] Would merge coalition agreements into {output_dir}")
        for gov in GOVERNMENTS:
            print(f"    - {gov.government_id}.json")
        return

    created = merge_all_governments(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
    )
    print(f"\n  Generated {len(created)} coalition agreement files.")


# ---------------------------------------------------------------------------
# Cleanup: remove redundant duplicates from source
# ---------------------------------------------------------------------------

def remove_duplicates_from_source(
    all_platforms: List[Dict[str, Any]],
    canonical: List[Dict[str, Any]],
    dry_run: bool = False,
) -> None:
    """
    Remove non-canonical duplicate files from the source directory.
    """
    canonical_paths = {str(p["path"]) for p in canonical}

    for p in all_platforms:
        if str(p["path"]) not in canonical_paths:
            if dry_run:
                print(f"  [DRY-RUN] Would remove duplicate: {p['path']}")
            else:
                try:
                    os.remove(str(p["path"]))
                    print(f"  [DELETE] Removed duplicate: {p['path']}")
                except OSError as e:
                    print(f"  [ERROR] Could not remove {p['path']}: {e}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def restructure(source_dir: str, dry_run: bool = False) -> None:
    """Run the full dataset restructuring pipeline."""
    src = Path(source_dir)
    base = Path(__file__).resolve().parent

    print("=" * 60)
    print("RON Coalition Promises - Dataset Restructuring")
    print("=" * 60)
    print(f"Source directory: {src}")
    print(f"Output base:     {base}")
    print(f"Dry run:         {dry_run}")
    print()

    # ---------------------------------------------------------------
    # Step 1: Scan party platforms
    # ---------------------------------------------------------------
    print("[Step 1] Scanning party platform JSON files...")
    if src.exists():
        all_platforms = scan_party_platforms(src)
        print(f"  Found {len(all_platforms)} platform file(s)")
    else:
        print(f"  Source directory does not exist: {src}")
        print("  Creating empty structure (no party platforms to process)")
        all_platforms = []

    # ---------------------------------------------------------------
    # Step 2: Deduplicate
    # ---------------------------------------------------------------
    print("\n[Step 2] Deduplicating party platforms...")
    canonical = deduplicate_platforms(all_platforms)
    print(f"  Canonical files: {len(canonical)}")

    # ---------------------------------------------------------------
    # Step 3: Place into election period directories
    # ---------------------------------------------------------------
    print("\n[Step 3] Placing platforms into election period directories...")
    place_platforms_in_periods(canonical, base, dry_run=dry_run)

    # ---------------------------------------------------------------
    # Step 4: Update Elections_*.json with platform references
    # ---------------------------------------------------------------
    print("\n[Step 4] Updating election period JSON files...")
    update_election_json_files(canonical, base, dry_run=dry_run)

    # ---------------------------------------------------------------
    # Step 5: Generate coalition agreement merged JSONs
    # ---------------------------------------------------------------
    print("\n[Step 5] Generating coalition agreement files...")
    generate_coalition_agreements(src, base, dry_run=dry_run)

    # ---------------------------------------------------------------
    # Step 6: Remove redundant duplicates from source
    # ---------------------------------------------------------------
    if all_platforms:
        print("\n[Step 6] Cleaning up duplicate source files...")
        remove_duplicates_from_source(all_platforms, canonical, dry_run=dry_run)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Restructuring complete.")
    print()
    print("Directory structure:")
    print(f"  {base}/")
    print(f"    election_periods/")
    for period in sorted(ELECTION_PERIODS.keys()):
        print(f"      {period}/")
        period_dir = base / "election_periods" / period
        if period_dir.exists():
            for f in sorted(period_dir.iterdir()):
                print(f"        {f.name}")
    print(f"    coalition_agreements/")
    ca_dir = base / "coalition_agreements"
    if ca_dir.exists():
        for f in sorted(ca_dir.iterdir()):
            print(f"      {f.name}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Restructure RON coalition promises dataset into layered format."
    )
    parser.add_argument(
        "--source-dir",
        default=str(Path(__file__).resolve().parent / "ron"),
        help="Path to the source ron/ directory with raw party platform JSONs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    args = parser.parse_args()

    restructure(source_dir=args.source_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

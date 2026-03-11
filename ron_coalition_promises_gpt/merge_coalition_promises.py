"""
Merge coalition treaty promises and government action plan (valitsuskava) promises
into a single JSON file per government.

The merged output keeps the two promise sources in separate arrays so that
downstream GPT agents can immediately determine the origin of each promise.

Output schema per government:
{
    "government_id": "...",
    "government_name": "...",
    "start_date": "...",
    "end_date": "...",
    "coalition_parties": [...],
    "election_period": "...",
    "coalition_treaty_promises": [...],
    "valitsuskava_promises": [...]
}
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow importing ministers_data from the same package
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ministers_data import GOVERNMENTS, Government


def load_json_file(filepath: str) -> Optional[Dict[str, Any]]:
    """Load and return parsed JSON from a file, or None on failure."""
    path = Path(filepath)
    if not path.exists():
        print(f"  [SKIP] File not found: {filepath}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [ERROR] Could not load {filepath}: {e}")
        return None


def extract_promises(data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract the promises array from a loaded JSON document.

    Supports multiple common structures:
      - {"promises": [...]}
      - {"lubadused": [...]}
      - A bare list [...]
      - {"data": {"promises": [...]}}
    """
    if data is None:
        return []

    # Direct list
    if isinstance(data, list):
        return data

    # Top-level promises key (English or Estonian)
    for key in ("promises", "lubadused", "items"):
        if key in data and isinstance(data[key], list):
            return data[key]

    # Nested under "data"
    if "data" in data and isinstance(data["data"], dict):
        for key in ("promises", "lubadused", "items"):
            if key in data["data"] and isinstance(data["data"][key], list):
                return data["data"][key]

    return []


def merge_government_promises(
    government: Government,
    coalition_treaty_path: Optional[str] = None,
    valitsuskava_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Merge coalition treaty and valitsuskava promises for a single government.

    Returns the merged document with promises kept in separate arrays.
    """
    coalition_data = load_json_file(coalition_treaty_path) if coalition_treaty_path else None
    valitsuskava_data = load_json_file(valitsuskava_path) if valitsuskava_path else None

    coalition_promises = extract_promises(coalition_data)
    valitsuskava_promises = extract_promises(valitsuskava_data)

    merged = {
        "government_id": government.government_id,
        "government_name": government.government_name,
        "start_date": government.start_date,
        "end_date": government.end_date,
        "coalition_parties": government.coalition_parties,
        "election_period": government.election_period,
        "coalition_treaty_promises": coalition_promises,
        "valitsuskava_promises": valitsuskava_promises,
    }

    return merged


def save_merged_json(merged: Dict[str, Any], output_path: str) -> None:
    """Write merged government JSON to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Saved: {output_path}")


def merge_all_governments(
    source_dir: str,
    output_dir: str,
) -> List[str]:
    """
    Merge promises for every government defined in ministers_data.

    Source directory layout expected:
      source_dir/
        <government_id>/
          coalition_treaty.json
          valitsuskava.json

    Output:
      output_dir/
        <government_id>.json

    Returns list of output file paths created.
    """
    src = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    created_files = []

    for gov in GOVERNMENTS:
        print(f"\nProcessing: {gov.government_name} ({gov.government_id})")

        gov_src = src / gov.government_id

        # Look for coalition treaty file
        treaty_path = None
        for candidate in ("coalition_treaty.json", "koalitsioonileping.json"):
            p = gov_src / candidate
            if p.exists():
                treaty_path = str(p)
                break

        # Also check government-level overrides from ministers_data
        if treaty_path is None and gov.coalition_treaty_file:
            if Path(gov.coalition_treaty_file).exists():
                treaty_path = gov.coalition_treaty_file

        # Look for valitsuskava file
        vk_path = None
        for candidate in ("valitsuskava.json", "government_action_plan.json"):
            p = gov_src / candidate
            if p.exists():
                vk_path = str(p)
                break

        if vk_path is None and gov.valitsuskava_file:
            if Path(gov.valitsuskava_file).exists():
                vk_path = gov.valitsuskava_file

        # Merge
        merged = merge_government_promises(gov, treaty_path, vk_path)

        output_path = str(out / f"{gov.government_id}.json")
        save_merged_json(merged, output_path)
        created_files.append(output_path)

    return created_files


if __name__ == "__main__":
    base = Path(__file__).resolve().parent

    # Default source: raw data directory (to be populated with processed JSONs)
    default_source = str(base / "ron")
    # Default output: coalition_agreements layer
    default_output = str(base / "coalition_agreements")

    source_dir = sys.argv[1] if len(sys.argv) > 1 else default_source
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output

    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    created = merge_all_governments(source_dir, output_dir)

    print("\n" + "=" * 60)
    print(f"Done. Created {len(created)} merged government files.")

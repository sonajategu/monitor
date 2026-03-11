"""
Estonian government coalition data for the period 2003-2023.

This module defines all coalition governments, their composition,
election periods, and associated metadata used for dataset structuring.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Government:
    government_id: str
    government_name: str
    prime_minister: str
    start_date: str
    end_date: str
    coalition_parties: List[str]
    election_year: int
    election_period: str
    coalition_treaty_file: Optional[str] = None
    valitsuskava_file: Optional[str] = None


# Election periods
ELECTION_PERIODS = {
    "2003-2007": {
        "election_year": 2003,
        "start_date": "2003-03-02",
        "end_date": "2007-03-04",
        "description": "XI Riigikogu valimised",
    },
    "2007-2015": {
        "election_year": 2007,
        "start_date": "2007-03-04",
        "end_date": "2015-03-01",
        "description": "XII-XIII Riigikogu valimised",
    },
    "2015-2019": {
        "election_year": 2015,
        "start_date": "2015-03-01",
        "end_date": "2019-03-03",
        "description": "XIII Riigikogu valimised",
    },
    "2019-2023": {
        "election_year": 2019,
        "start_date": "2019-03-03",
        "end_date": "2023-03-05",
        "description": "XIV Riigikogu valimised",
    },
}


# All coalition governments of Estonia 2003-2023
GOVERNMENTS = [
    # === Election period 2003-2007 ===
    Government(
        government_id="parts_2003",
        government_name="Juhan Partsi valitsus",
        prime_minister="Juhan Parts",
        start_date="2003-04-10",
        end_date="2005-04-13",
        coalition_parties=["Res Publica", "Reformierakond", "Rahvaliit"],
        election_year=2003,
        election_period="2003-2007",
    ),
    Government(
        government_id="ansip_i_2005",
        government_name="Andrus Ansipi I valitsus",
        prime_minister="Andrus Ansip",
        start_date="2005-04-13",
        end_date="2007-04-05",
        coalition_parties=["Reformierakond", "Keskerakond", "Rahvaliit"],
        election_year=2003,
        election_period="2003-2007",
    ),

    # === Election period 2007-2015 ===
    Government(
        government_id="ansip_ii_2007",
        government_name="Andrus Ansipi II valitsus",
        prime_minister="Andrus Ansip",
        start_date="2007-04-05",
        end_date="2009-06-04",
        coalition_parties=["Reformierakond", "Isamaa ja Res Publica Liit", "Sotsiaaldemokraatlik Erakond"],
        election_year=2007,
        election_period="2007-2015",
    ),
    Government(
        government_id="ansip_iii_2009",
        government_name="Andrus Ansipi III valitsus",
        prime_minister="Andrus Ansip",
        start_date="2009-06-04",
        end_date="2011-04-06",
        coalition_parties=["Reformierakond", "Isamaa ja Res Publica Liit"],
        election_year=2007,
        election_period="2007-2015",
    ),
    Government(
        government_id="ansip_iv_2011",
        government_name="Andrus Ansipi IV valitsus",
        prime_minister="Andrus Ansip",
        start_date="2011-04-06",
        end_date="2014-03-26",
        coalition_parties=["Reformierakond", "Isamaa ja Res Publica Liit"],
        election_year=2011,
        election_period="2007-2015",
    ),
    Government(
        government_id="roivas_i_2014",
        government_name="Taavi Rõivase I valitsus",
        prime_minister="Taavi Rõivas",
        start_date="2014-03-26",
        end_date="2015-04-09",
        coalition_parties=["Reformierakond", "Sotsiaaldemokraatlik Erakond"],
        election_year=2011,
        election_period="2007-2015",
    ),

    # === Election period 2015-2019 ===
    Government(
        government_id="roivas_ii_2015",
        government_name="Taavi Rõivase II valitsus",
        prime_minister="Taavi Rõivas",
        start_date="2015-04-09",
        end_date="2016-11-23",
        coalition_parties=["Reformierakond", "Sotsiaaldemokraatlik Erakond", "Isamaa ja Res Publica Liit"],
        election_year=2015,
        election_period="2015-2019",
    ),
    Government(
        government_id="ratas_i_2016",
        government_name="Jüri Ratase I valitsus",
        prime_minister="Jüri Ratas",
        start_date="2016-11-23",
        end_date="2019-04-29",
        coalition_parties=["Keskerakond", "Sotsiaaldemokraatlik Erakond", "Isamaa ja Res Publica Liit"],
        election_year=2015,
        election_period="2015-2019",
    ),

    # === Election period 2019-2023 ===
    Government(
        government_id="ratas_ii_2019",
        government_name="Jüri Ratase II valitsus",
        prime_minister="Jüri Ratas",
        start_date="2019-04-29",
        end_date="2021-01-26",
        coalition_parties=["Keskerakond", "Eesti Konservatiivne Rahvaerakond", "Isamaa"],
        election_year=2019,
        election_period="2019-2023",
    ),
    Government(
        government_id="kallas_i_2021",
        government_name="Kaja Kallase I valitsus",
        prime_minister="Kaja Kallas",
        start_date="2021-01-26",
        end_date="2022-07-18",
        coalition_parties=["Reformierakond", "Keskerakond"],
        election_year=2019,
        election_period="2019-2023",
    ),
    Government(
        government_id="kallas_ii_2022",
        government_name="Kaja Kallase II valitsus",
        prime_minister="Kaja Kallas",
        start_date="2022-07-18",
        end_date="2023-04-17",
        coalition_parties=["Reformierakond", "Sotsiaaldemokraatlik Erakond", "Isamaa"],
        election_year=2019,
        election_period="2019-2023",
    ),
]


def get_governments_by_election_period(period: str) -> List[Government]:
    """Return all governments belonging to a given election period."""
    return [g for g in GOVERNMENTS if g.election_period == period]


def get_government_by_id(government_id: str) -> Optional[Government]:
    """Return a government by its unique ID."""
    for g in GOVERNMENTS:
        if g.government_id == government_id:
            return g
    return None


def get_all_election_periods() -> List[str]:
    """Return sorted list of all election period keys."""
    return sorted(ELECTION_PERIODS.keys())


def get_all_coalition_parties() -> List[str]:
    """Return deduplicated list of all coalition parties across all governments."""
    parties = set()
    for g in GOVERNMENTS:
        parties.update(g.coalition_parties)
    return sorted(parties)

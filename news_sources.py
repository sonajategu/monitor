from dataclasses import dataclass
from typing import List, Dict

@dataclass
class NewsSource:
    url: str
    category: str
    subcategory: str
    name: str
    priority: int = 1
    rate_limit: int = 5  # seconds between requests

    def to_dict(self) -> dict:
        """Convert NewsSource to dictionary for serialization"""
        return {
            'url': self.url,
            'category': self.category,
            'subcategory': self.subcategory,
            'name': self.name,
            'priority': self.priority,
            'rate_limit': self.rate_limit
        }

    def __str__(self) -> str:
        """String representation for logging"""
        return f"{self.name} ({self.category}/{self.subcategory})"

NEWS_SOURCES = [
    # Political Parties (High Priority)
    NewsSource(
        url='https://www.err.ee/rss/keyword/10962',
        category='political_party',
        subcategory='coalition',
        name='SDE',
        priority=1
    ),
    NewsSource(
        url='https://www.err.ee/rss/keyword/16777',
        category='political_party',
        subcategory='opposition',
        name='Keskerakond',
        priority=1
    ),
    NewsSource(
        url='https://www.err.ee/rss/keyword/16776',
        category='political_party',
        subcategory='coalition',
        name='Reformierakond',
        priority=1
    ),
    NewsSource(
        url='https://www.err.ee/rss/keyword/16780',
        category='political_party',
        subcategory='opposition',
        name='Rohelised',
        priority=1
    ),
    NewsSource(
        url='https://www.err.ee/rss/keyword/134679',
        category='political_party',
        subcategory='opposition',
        name='Parempoolsed',
        priority=1
    ),
    NewsSource(
        url='https://www.err.ee/rss/keyword/616226',
        category='political_party',
        subcategory='coalition',
        name='Isamaa',
        priority=1
    ),
    NewsSource(
        url='https://www.err.ee/rss/keyword/1575',
        category='political_party',
        subcategory='opposition',
        name='EKRE',
        priority=1
    ),
    NewsSource(
        url='https://www.err.ee/rss/keyword/131711',
        category='political_party',
        subcategory='opposition',
        name='EKRE',  # Alternative name/URL for EKRE
        priority=1
    ),
    NewsSource(
        url='https://www.err.ee/rss/keyword/32675372',
        category='political_party',
        subcategory='opposition',
        name='ERK',
        priority=1
    ),
    NewsSource(
        url='https://www.err.ee/rss/keyword/16504123',
        category='political_party',
        subcategory='opposition',
        name='KOOS',
        priority=1
    ),
    NewsSource(
        url='https://www.err.ee/rss/keyword/800395',
        category='political_party',
        subcategory='coalition',
        name='Eesti 200',
        priority=1
    ),

    # Government Institutions (Medium Priority)
    NewsSource(
        url='https://www.err.ee/rss/keyword/18434',
        category='institution',
        subcategory='defense',
        name='Kaitsevägi',
        priority=2
    ),
    NewsSource(
        url='https://www.err.ee/rss/keyword/99355',
        category='institution',
        subcategory='municipality',
        name='Tartu linnavalitsus',
        priority=2
    ),

    # Politicians (Medium Priority)
    NewsSource(
        url='https://www.err.ee/rss/keyword/5938',
        category='politician',
        subcategory='minister',
        name='Hanno Pevkur',
        priority=2
    ),
    NewsSource(
        url='https://www.err.ee/rss/keyword/23206',
        category='politician',
        subcategory='minister',
        name='Kristen Michal',
        priority=2
    ),
]

# Group sources by category for easier management
SOURCES_BY_CATEGORY: Dict[str, List[NewsSource]] = {}
for source in NEWS_SOURCES:
    if source.category not in SOURCES_BY_CATEGORY:
        SOURCES_BY_CATEGORY[source.category] = []
    SOURCES_BY_CATEGORY[source.category].append(source)

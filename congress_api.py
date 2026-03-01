"""
Layer 1: Congress.gov API Client for ingesting political speech data.

Pulls Congressional Record entries (floor speeches, remarks) and 
associates them with specific members of Congress.

API Docs: https://api.congress.gov/
Sign up for key: https://api.congress.gov/sign-up/
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

import config


class CongressAPI:
    """
    Client for the Congress.gov API v3.
    
    Handles rate limiting, pagination, and data normalization
    for Congressional Record and member data.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.CONGRESS_API_KEY
        self.base_url = config.CONGRESS_API_BASE
        self.session = requests.Session()
        self.session.params = {"api_key": self.api_key, "format": "json"}
        self._rate_limit_delay = 0.5  # seconds between requests
    
    def _get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make a rate-limited GET request to the API."""
        url = f"{self.base_url}/{endpoint}"
        try:
            time.sleep(self._rate_limit_delay)
            resp = self.session.get(url, params=params or {}, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"[ERROR] API request failed: {e}")
            return None
    
    def _paginate(self, endpoint: str, params: dict = None, 
                  max_results: int = 250) -> List[dict]:
        """Paginate through API results."""
        params = params or {}
        params["limit"] = min(max_results, 250)
        all_results = []
        
        while len(all_results) < max_results:
            data = self._get(endpoint, params)
            if not data:
                break
            
            # Congress.gov nests results differently per endpoint
            # Try common result keys
            results = None
            for key in ["results", "congressionalRecord", "members", 
                       "dailyDigest", "articles"]:
                if key in data:
                    results = data[key]
                    break
            
            if not results or len(results) == 0:
                break
            
            all_results.extend(results)
            
            # Check for pagination
            pagination = data.get("pagination", {})
            next_url = pagination.get("next")
            if not next_url:
                break
            
            # Extract offset from next URL
            params["offset"] = params.get("offset", 0) + len(results)
        
        return all_results[:max_results]
    
    # =========================================================================
    # MEMBER DATA
    # =========================================================================
    
    def get_members(self, congress: int = None, chamber: str = None) -> List[dict]:
        """
        Get list of members for a given Congress.
        
        Args:
            congress: Congress number (e.g., 118 for 2023-2025)
            chamber: 'house' or 'senate'
        
        Returns:
            List of member dicts with bioguide_id, name, party, state
        """
        congress = congress or config.CURRENT_CONGRESS
        endpoint = f"member/congress/{congress}"
        if chamber:
            endpoint += f"/{chamber}"
        
        raw_members = self._paginate(endpoint, max_results=600)
        
        members = []
        for m in raw_members:
            member = {
                "bioguide_id": m.get("bioguideId", ""),
                "name": m.get("name", ""),
                "first_name": m.get("firstName", ""),
                "last_name": m.get("lastName", ""),
                "party": m.get("partyName", ""),
                "state": m.get("state", ""),
                "chamber": m.get("terms", {}).get("item", [{}])[0].get("chamber", "") 
                          if m.get("terms") else "",
                "district": m.get("district", ""),
            }
            members.append(member)
        
        return members
    
    def get_member_by_name(self, name: str, congress: int = None) -> Optional[dict]:
        """Search for a member by name."""
        members = self.get_members(congress)
        name_lower = name.lower()
        for m in members:
            if (name_lower in m["name"].lower() or 
                name_lower in f"{m['first_name']} {m['last_name']}".lower()):
                return m
        return None
    
    # =========================================================================
    # CONGRESSIONAL RECORD (speeches)
    # =========================================================================
    
    def get_daily_record(self, date: str = None, congress: int = None) -> List[dict]:
        """
        Get Congressional Record entries for a specific date.
        
        Args:
            date: Date string 'YYYY-MM-DD' (default: today)
            congress: Congress number
        
        Returns:
            List of record entries (articles/speeches)
        """
        congress = congress or config.CURRENT_CONGRESS
        endpoint = f"congressional-record"
        
        params = {}
        if date:
            # The API uses year, month, day
            params["y"] = date[:4]
            params["m"] = date[5:7]
            params["d"] = date[8:10]
        
        return self._paginate(endpoint, params)
    
    def get_daily_record_range(self, start_date: str, end_date: str,
                                congress: int = None) -> List[dict]:
        """
        Get Congressional Record entries for a date range.
        
        Args:
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
        
        Returns:
            List of all record entries in range
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_records = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            print(f"  Fetching Congressional Record for {date_str}...")
            records = self.get_daily_record(date_str, congress)
            all_records.extend(records)
            current += timedelta(days=1)
        
        return all_records
    
    # =========================================================================
    # BILL DATA (for topic context)
    # =========================================================================
    
    def get_recent_bills(self, congress: int = None, 
                         limit: int = 20) -> List[dict]:
        """Get recently introduced bills."""
        congress = congress or config.CURRENT_CONGRESS
        endpoint = f"bill/{congress}"
        return self._paginate(endpoint, max_results=limit)
    
    def get_bill_details(self, congress: int, bill_type: str, 
                         bill_number: int) -> Optional[dict]:
        """Get details for a specific bill."""
        endpoint = f"bill/{congress}/{bill_type}/{bill_number}"
        return self._get(endpoint)


class SpeechRecord:
    """
    Normalized representation of a political speech/statement.
    
    This is the standard data structure that flows through the pipeline.
    """
    
    def __init__(self, text: str, speaker_id: str, speaker_name: str,
                 date: str, source: str, chamber: str = "",
                 party: str = "", state: str = "", 
                 context: str = "", url: str = ""):
        self.text = text
        self.speaker_id = speaker_id      # bioguide_id
        self.speaker_name = speaker_name
        self.date = date                   # YYYY-MM-DD
        self.source = source               # 'congressional_record', 'twitter', etc.
        self.chamber = chamber
        self.party = party
        self.state = state
        self.context = context             # bill number, hearing title, etc.
        self.url = url
        self.word_count = len(text.split())
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "date": self.date,
            "source": self.source,
            "chamber": self.chamber,
            "party": self.party,
            "state": self.state,
            "context": self.context,
            "url": self.url,
            "word_count": self.word_count,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "SpeechRecord":
        return cls(**{k: v for k, v in d.items() if k != "word_count"})
    
    def __repr__(self):
        return (f"SpeechRecord({self.speaker_name}, {self.date}, "
                f"{self.word_count} words, {self.source})")


class TextPreprocessor:
    """
    Cleans and normalizes raw text from various sources.
    """
    
    @staticmethod
    def clean_congressional_record(raw_text: str) -> str:
        """Clean text from Congressional Record HTML/XML."""
        import re
        
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", raw_text)
        
        # Remove procedural markers
        procedural = [
            r"\[.*?\]",                    # [brackets content]
            r"Mr\./Mrs\./Ms\. SPEAKER",    # Speaker references
            r"PARLIAMENTARY INQUIRY",
            r"POINT OF ORDER",
            r"The PRESIDING OFFICER",
            r"The SPEAKER pro tempore",
        ]
        for pattern in procedural:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        # Remove very short fragments (likely procedural)
        sentences = text.split(".")
        substantive = [s.strip() for s in sentences if len(s.strip().split()) > 5]
        text = ". ".join(substantive)
        
        return text
    
    @staticmethod
    def clean_social_media(raw_text: str) -> str:
        """Clean text from Twitter/X or other social media."""
        import re
        
        # Remove URLs
        text = re.sub(r"https?://\S+", "", raw_text)
        # Remove @mentions (keep the name part)
        text = re.sub(r"@(\w+)", r"\1", text)
        # Remove hashtag symbols (keep the word)
        text = re.sub(r"#(\w+)", r"\1", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text


# =============================================================================
# SAMPLE DATA FOR TESTING (when API key not available)
# =============================================================================

SAMPLE_SPEECHES = [
    SpeechRecord(
        text="""Mr. Speaker, I rise today in strong support of this legislation. 
        We must act now to protect the American people from the devastating effects 
        of rising healthcare costs. I have spoken with families in my district who 
        cannot afford their prescription medications. This is unacceptable. We will 
        pass this bill, we will deliver relief, and we will not rest until every 
        American has access to affordable care. The time for half measures is over. 
        We need bold action, and this bill delivers exactly that. I urge my 
        colleagues on both sides of the aisle to support this critical legislation. 
        Our constituents are counting on us, and we must not let them down. Together, 
        we can build a healthcare system that works for everyone, not just the 
        wealthy and the well-connected. I yield back the balance of my time.""",
        speaker_id="P000197",
        speaker_name="Nancy Pelosi",
        date="2024-03-15",
        source="congressional_record",
        chamber="house",
        party="Democratic",
        state="CA",
        context="HR-4521 Healthcare Affordability Act",
    ),
    SpeechRecord(
        text="""Madam President, I want to address some concerns that have been 
        raised about this appropriations bill. I think we need to look at this 
        more carefully. There are some provisions that might have unintended 
        consequences for our rural communities. Perhaps we should consider 
        amendments that could address these issues. I believe there may be 
        a better way to structure the funding mechanisms. It seems like the 
        current approach could potentially create problems down the road. 
        I'm not entirely opposed to the bill, but I think we owe it to our 
        constituents to get this right. I would suggest that we maybe take 
        another week to review the language. Some of my colleagues have 
        expressed similar concerns, and I think it would be wise to address 
        them before we proceed to a vote. I hope we can find common ground 
        on this important matter.""",
        speaker_id="M001153",
        speaker_name="Lisa Murkowski",
        date="2024-03-15",
        source="congressional_record",
        chamber="senate",
        party="Republican",
        state="AK",
        context="S-1234 Appropriations Bill",
    ),
    SpeechRecord(
        text="""I rise in opposition to this reckless spending bill. The American 
        people are tired of Washington wasting their hard-earned money. This 
        bill adds billions to our already crushing national debt. Every family 
        in America has to balance their budget, but apparently Congress doesn't 
        think the same rules apply to us. This is outrageous. We cannot continue 
        down this path of fiscal irresponsibility. I will absolutely vote no on 
        this legislation, and I urge every one of my colleagues who cares about 
        the future of this country to do the same. We need to cut spending, not 
        increase it. We need to reduce the deficit, not add to it. The American 
        taxpayer deserves better than this bloated, wasteful bill. I will fight 
        this bill with everything I have, and I will not back down. The fiscal 
        future of our nation depends on it.""",
        speaker_id="C001098",
        speaker_name="Ted Cruz",
        date="2024-03-16",
        source="congressional_record",
        chamber="senate",
        party="Republican",
        state="TX",
        context="HR-5678 Federal Budget",
    ),
    SpeechRecord(
        text="""Thank you, Mr. Chairman. I want to discuss the implications of 
        this trade agreement for American workers. The data suggests that similar 
        agreements in the past have resulted in significant job displacement in 
        manufacturing sectors. However, proponents argue that the net economic 
        benefits outweigh these costs. I think we need to examine both sides 
        carefully. On one hand, increased market access could boost our export 
        industries. On the other hand, we risk accelerating the offshoring of 
        production. The Congressional Budget Office analysis indicates mixed 
        results. Some sectors would gain, while others would face increased 
        competition. I believe we should consider stronger worker protection 
        provisions before moving forward. Perhaps a phased implementation 
        approach would help mitigate the negative impacts while still capturing 
        the economic benefits. I look forward to working with my colleagues 
        on amendments that address these concerns.""",
        speaker_id="W000187",
        speaker_name="Elizabeth Warren",
        date="2024-03-17",
        source="congressional_record",
        chamber="senate",
        party="Democratic",
        state="MA",
        context="Trade Agreement Review Committee Hearing",
    ),
]


def load_sample_data() -> List[SpeechRecord]:
    """Load sample data for testing without API access."""
    return SAMPLE_SPEECHES

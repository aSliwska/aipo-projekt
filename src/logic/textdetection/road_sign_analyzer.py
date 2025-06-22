"""
Road sign analysis and country detection.
"""

import logging
from typing import List, Optional, Dict, Any

# Try to import API client, fallback to local patterns only if not available
try:
    from .api_client import PatternAPIClient
    API_AVAILABLE = True
except ImportError:
    logging.warning("API client not available, using local patterns only")
    API_AVAILABLE = False


class RoadSignAnalyzer:
    def __init__(self, use_api: bool = True):
        """
        Initialize RoadSignAnalyzer with optional API integration.

        Args:
            use_api: Whether to use API for pattern updates (default: True)
        """
        self.use_api = use_api and API_AVAILABLE
        self.api_client = None

        if self.use_api:
            try:
                self.api_client = PatternAPIClient()
                logging.info("RoadSignAnalyzer initialized with API support")
            except Exception as e:
                logging.warning("Failed to initialize API client: %s", e)
                self.use_api = False

        # Load patterns (API first, then fallback to hardcoded)
        self._load_patterns()

    def _load_patterns(self):
        """Load road sign patterns from API or use hardcoded fallback."""
        if self.use_api and self.api_client:
            try:
                api_patterns = self.api_client.get_enhanced_road_sign_patterns()
                if api_patterns:
                    logging.info(
                        "Loaded road sign patterns for %d countries from API", len(api_patterns))
                    self.sign_patterns = {
                        **self._get_fallback_patterns(), **api_patterns}
                    return
            except Exception as e:
                logging.warning(
                    "Failed to load road sign patterns from API: %s", e)

        # Use fallback patterns
        self.sign_patterns = self._get_fallback_patterns()
        logging.info("Using fallback road sign patterns for %d countries", len(
            self.sign_patterns))

    def _get_fallback_patterns(self) -> Dict[str, Dict]:
        """Get hardcoded fallback road sign patterns."""
        return {
            # European patterns
            "Poland": {
                "keywords": ["KRAKÓW", "WARSZAWA", "GDANSK", "WROCŁAW", "POZNAŃ", "km", "DK", "A1", "A2", "A4"],
                "distance_units": ["km"],
                "highway_prefixes": ["A", "DK", "S"]
            },
            "Germany": {
                "keywords": ["BERLIN", "MÜNCHEN", "HAMBURG", "KÖLN", "FRANKFURT", "km", "BAB"],
                "distance_units": ["km"],
                "highway_prefixes": ["A", "B"]
            },
            "France": {
                "keywords": ["PARIS", "LYON", "MARSEILLE", "TOULOUSE", "NICE", "km", "A", "N", "D"],
                "distance_units": ["km"],
                "highway_prefixes": ["A", "N", "D"]
            },
            "Italy": {
                "keywords": ["ROMA", "MILANO", "NAPOLI", "TORINO", "FIRENZE", "km"],
                "distance_units": ["km"],
                "highway_prefixes": ["A", "SS"]
            },
            "Spain": {
                "keywords": ["MADRID", "BARCELONA", "VALENCIA", "SEVILLA", "BILBAO", "km"],
                "distance_units": ["km"],
                "highway_prefixes": ["A", "AP", "N"]
            },
            "Czech Republic": {
                "keywords": ["PRAHA", "BRNO", "OSTRAVA", "PLZEŇ", "km"],
                "distance_units": ["km"],
                "highway_prefixes": ["D", "R"]
            },
            # English-speaking countries
            "United Kingdom": {
                "keywords": ["LONDON", "MANCHESTER", "BIRMINGHAM", "GLASGOW", "CARDIFF", "miles", "M", "A"],
                "distance_units": ["miles", "m"],
                "highway_prefixes": ["M", "A", "B"]
            },
            "United States": {
                "keywords": ["MILE", "MILES", "EXIT", "INTERSTATE", "US", "STATE"],
                "distance_units": ["miles", "mi", "m"],
                "highway_prefixes": ["I-", "US", "SR", "CR"]
            },
            "Canada": {
                "keywords": ["km", "TORONTO", "MONTREAL", "VANCOUVER", "CALGARY"],
                "distance_units": ["km", "miles"],
                "highway_prefixes": ["TCH", "QC", "ON", "BC"]
            },
            "Australia": {
                "keywords": ["km", "SYDNEY", "MELBOURNE", "BRISBANE", "PERTH"],
                "distance_units": ["km"],
                "highway_prefixes": ["M", "A", "B"]
            },
            # Asian countries
            "Japan": {
                "keywords": ["km", "キロ", "東京", "大阪", "名古屋", "福岡"],
                "distance_units": ["km", "キロ"],
                "highway_prefixes": ["E", "C"]
            },
            "China": {
                "keywords": ["公里", "北京", "上海", "广州", "深圳", "km"],
                "distance_units": ["公里", "km"],
                "highway_prefixes": ["G", "S"]
            }
        }

    def analyze_sign_text(self, text: str) -> List[str]:
        """
        Analyze road sign text and return possible countries based on patterns.
        """
        if not text or len(text.strip()) < 2:
            return []

        text_upper = text.upper()
        possible_countries = []
        logging.debug("Analyzing sign text: '%s'", text)

        for country, patterns in self.sign_patterns.items():
            score = 0

            # Check keywords (city names, common terms)
            for keyword in patterns["keywords"]:
                if keyword.upper() in text_upper:
                    score += 2
                    logging.debug("Found keyword '%s' for %s",
                                  keyword, country)

            # Check distance units
            for unit in patterns["distance_units"]:
                if unit.upper() in text_upper:
                    score += 1
                    logging.debug(
                        "Found distance unit '%s' for %s", unit, country)

            # Check highway prefixes
            for prefix in patterns["highway_prefixes"]:
                if prefix.upper() in text_upper:
                    score += 1
                    logging.debug(
                        "Found highway prefix '%s' for %s", prefix, country)

            if score > 0:
                possible_countries.append(country)
                logging.info("Sign text '%s' matches %s (score: %d)",
                             text, country, score)

        return possible_countries

    def is_country_specific_sign(self, text: str) -> Optional[str]:
        """
        Check if a road sign is clearly from a specific country.
        Returns the country name if clearly identifiable, None otherwise.
        """
        countries = self.analyze_sign_text(text)

        # If only one country matches with high confidence
        if len(countries) == 1:
            return countries[0]

        # If multiple countries match, try to find the most specific one
        if len(countries) > 1:
            text_upper = text.upper()

            # Priority for major city names (more specific)
            major_cities = {
                "WARSZAWA": "Poland",
                "KRAKÓW": "Poland",
                "BERLIN": "Germany",
                "MÜNCHEN": "Germany",
                "PARIS": "France",
                "LONDON": "United Kingdom",
                "ROMA": "Italy",
                "MADRID": "Spain",
                "PRAHA": "Czech Republic",
                "TOKYO": "Japan",
                "東京": "Japan",
                "BEIJING": "China",
                "北京": "China"
            }

            for city, country in major_cities.items():
                if city in text_upper and country in countries:
                    logging.info(
                        "Found major city '%s' indicating %s", city, country)
                    return country

        return None

    def update_patterns_from_api(self) -> bool:
        """Update patterns from API and return success status."""
        if not self.use_api or not self.api_client:
            logging.warning("API not available for pattern updates")
            return False

        try:
            success = self.api_client.update_patterns()
            if success:
                self._load_patterns()  # Reload patterns after API update
                logging.info(
                    "Successfully updated road sign patterns from API")
            return success
        except Exception as e:
            logging.error(
                "Failed to update road sign patterns from API: %s", e)
            return False

    def get_pattern_info(self) -> Dict[str, Any]:
        """Get information about current patterns and API status."""
        return {
            "api_enabled": self.use_api,
            "api_available": API_AVAILABLE,
            "total_countries": len(self.sign_patterns),
            "countries": list(self.sign_patterns.keys()),
            "last_api_update": getattr(self.api_client, 'cache', {}).get('last_updated', {}) if self.api_client else {}
        }

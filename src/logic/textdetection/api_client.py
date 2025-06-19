"""
API client for fetching road sign and license plate pattern data from free APIs.
"""

import logging
import json
import yaml
import requests
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


class PatternAPIClient:
    def __init__(self, config_path: str = None):
        """Initialize API client with configuration."""
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "api_config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.cache_file = Path(__file__).parent / "config" / \
            self.config["fallback"]["local_cache_file"]
        self.cache = self._load_cache()

        logging.info(
            "PatternAPIClient initialized with config from %s", config_path)

    def _load_config(self) -> Dict[str, Any]:
        """Load API configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error("Failed to load API config: %s", e)
            return {"apis": {}, "fallback": {"use_local_patterns": True}}

    def _load_cache(self) -> Dict[str, Any]:
        """Load cached API responses."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning("Failed to load cache: %s", e)
        return {"plate_patterns": {}, "road_signs": {}, "last_updated": {}}

    def _save_cache(self):
        """Save cache to file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error("Failed to save cache: %s", e)

    def _is_cache_valid(self, cache_key: str, hours: int) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache["last_updated"]:
            return False

        last_updated = datetime.fromisoformat(
            self.cache["last_updated"][cache_key])
        return datetime.now() - last_updated < timedelta(hours=hours)

    def _make_request(self, url: str, params: Dict = None, headers: Dict = None) -> Optional[Dict]:
        """Make HTTP request with retry logic."""
        retry_config = self.config.get("retry", {})
        max_attempts = retry_config.get("max_attempts", 3)
        backoff_factor = retry_config.get("backoff_factor", 2)
        timeout = retry_config.get("timeout_seconds", 30)

        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    url, params=params, headers=headers, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == max_attempts - 1:
                    logging.error(
                        "Request failed after %d attempts: %s", max_attempts, e)
                    return None

                wait_time = backoff_factor ** attempt
                logging.warning("Request failed (attempt %d/%d), retrying in %ds: %s",
                                attempt + 1, max_attempts, wait_time, e)
                time.sleep(wait_time)

        return None

    def fetch_country_data(self) -> Dict[str, Any]:
        """Fetch country data from REST Countries API."""
        cache_key = "countries"
        cache_duration = self.config["apis"]["rest_countries"]["cache_duration_hours"]

        if self._is_cache_valid(cache_key, cache_duration):
            logging.debug("Using cached country data")
            return self.cache.get(cache_key, {})

        if not self.config["apis"]["rest_countries"]["enabled"]:
            logging.info("REST Countries API disabled")
            return {}

        url = self.config["apis"]["rest_countries"]["base_url"] + "/all"
        params = {"fields": "name,cca2,cca3,region,subregion,languages,car"}

        logging.info("Fetching country data from REST Countries API")
        data = self._make_request(url, params)

        if data:
            # Process country data for our needs
            countries = {}
            for country in data:
                try:
                    name = country["name"]["common"]
                    countries[name] = {
                        "code2": country.get("cca2", ""),
                        "code3": country.get("cca3", ""),
                        "region": country.get("region", ""),
                        "subregion": country.get("subregion", ""),
                        "languages": list(country.get("languages", {}).values()),
                        "driving_side": country.get("car", {}).get("side", "right")
                    }
                except KeyError as e:
                    logging.warning("Incomplete country data for %s: %s",
                                    country.get("name", {}).get("common", "unknown"), e)
                    continue

            self.cache[cache_key] = countries
            self.cache["last_updated"][cache_key] = datetime.now().isoformat()
            self._save_cache()
            logging.info("Fetched data for %d countries", len(countries))
            return countries

        return self.cache.get(cache_key, {})

    def fetch_highway_systems(self) -> Dict[str, Any]:
        """Fetch highway system data using Wikidata SPARQL."""
        cache_key = "highway_systems"
        cache_duration = self.config["apis"]["wikidata"]["cache_duration_hours"]

        if self._is_cache_valid(cache_key, cache_duration):
            logging.debug("Using cached highway systems data")
            return self.cache.get(cache_key, {})

        if not self.config["apis"]["wikidata"]["enabled"]:
            logging.info("Wikidata API disabled")
            return {}

        # SPARQL query to get highway systems by country
        sparql_query = """
        SELECT ?country ?countryLabel ?highway ?highwayLabel ?prefix WHERE {
          ?highway wdt:P31/wdt:P279* wd:Q34442 .  # highway system
          ?highway wdt:P17 ?country .             # country
          OPTIONAL { ?highway wdt:P3913 ?prefix } # highway prefix
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        """

        url = self.config["apis"]["wikidata"]["sparql_endpoint"]
        headers = {"Accept": "application/json"}
        params = {"query": sparql_query, "format": "json"}

        logging.info("Fetching highway systems from Wikidata")
        data = self._make_request(url, params, headers)

        if data and "results" in data:
            highway_systems = {}
            for result in data["results"]["bindings"]:
                try:
                    country = result["countryLabel"]["value"]
                    highway = result["highwayLabel"]["value"]
                    prefix = result.get("prefix", {}).get("value", "")

                    if country not in highway_systems:
                        highway_systems[country] = {
                            "highways": [], "prefixes": []}

                    highway_systems[country]["highways"].append(highway)
                    if prefix and prefix not in highway_systems[country]["prefixes"]:
                        highway_systems[country]["prefixes"].append(prefix)

                except KeyError as e:
                    logging.warning("Incomplete highway data: %s", e)
                    continue

            self.cache[cache_key] = highway_systems
            self.cache["last_updated"][cache_key] = datetime.now().isoformat()
            self._save_cache()
            logging.info("Fetched highway systems for %d countries",
                         len(highway_systems))
            return highway_systems

        return self.cache.get(cache_key, {})

    def fetch_license_plate_patterns(self) -> Dict[str, Any]:
        """Fetch license plate pattern data from GitHub datasets."""
        cache_key = "license_plates"
        cache_duration = self.config["apis"]["github"]["cache_duration_hours"]

        if self._is_cache_valid(cache_key, cache_duration):
            logging.debug("Using cached license plate data")
            return self.cache.get(cache_key, {})

        if not self.config["apis"]["github"]["enabled"]:
            logging.info("GitHub API disabled")
            return {}

        # Try to fetch from a public dataset on GitHub
        # This is a hypothetical dataset - in reality you'd need to find or create one
        base_url = self.config["apis"]["github"]["endpoints"]["raw_content"]
        url = f"{base_url}/datasets/license-plates/master/plates.json"

        logging.info("Fetching license plate patterns from GitHub")
        data = self._make_request(url)

        if data:
            self.cache[cache_key] = data
            self.cache["last_updated"][cache_key] = datetime.now().isoformat()
            self._save_cache()
            logging.info(
                "Fetched license plate patterns for %d countries", len(data))
            return data

        return self.cache.get(cache_key, {})

    def get_enhanced_plate_patterns(self) -> Dict[str, Any]:
        """Get enhanced license plate patterns combining API data with fallbacks."""
        api_patterns = self.fetch_license_plate_patterns()
        country_data = self.fetch_country_data()

        # If API data is available, use it; otherwise fall back to built-in patterns
        if api_patterns:
            logging.info("Using API-sourced license plate patterns")
            return api_patterns

        logging.info("Using fallback license plate patterns")
        # Return enhanced patterns based on country data
        enhanced_patterns = {}

        for country, info in country_data.items():
            # Create basic patterns based on known country characteristics
            if country in ["Poland", "Germany", "France", "Italy", "Spain"]:
                enhanced_patterns[country] = {
                    # Generic European pattern
                    "patterns": [r"^[A-Z0-9\s\-]{4,12}$"],
                    "regions": {},
                    "driving_side": info.get("driving_side", "right"),
                    "region": info.get("region", ""),
                    "languages": info.get("languages", [])
                }

        return enhanced_patterns

    def get_enhanced_road_sign_patterns(self) -> Dict[str, Any]:
        """Get enhanced road sign patterns combining API data."""
        country_data = self.fetch_country_data()
        highway_systems = self.fetch_highway_systems()

        enhanced_patterns = {}

        for country, info in country_data.items():
            # Create road sign patterns based on API data
            patterns = {
                "keywords": [],
                "distance_units": ["km"] if info.get("driving_side") == "right" else ["miles", "km"],
                "highway_prefixes": highway_systems.get(country, {}).get("prefixes", []),
                "languages": info.get("languages", []),
                "region": info.get("region", ""),
                "driving_side": info.get("driving_side", "right")
            }

            # Add major cities as keywords (this would ideally come from another API)
            if "languages" in info:
                for lang in info["languages"]:
                    if "English" in lang:
                        patterns["distance_units"] = ["miles", "mi", "m", "km"]

            enhanced_patterns[country] = patterns

        return enhanced_patterns

    def update_patterns(self) -> bool:
        """Force update of all patterns from APIs."""
        logging.info("Forcing pattern update from APIs")

        # Clear cache timestamps to force refresh
        self.cache["last_updated"] = {}

        # Fetch fresh data
        try:
            self.fetch_country_data()
            self.fetch_highway_systems()
            self.fetch_license_plate_patterns()
            logging.info("Pattern update completed successfully")
            return True
        except Exception as e:
            logging.error("Pattern update failed: %s", e)
            return False

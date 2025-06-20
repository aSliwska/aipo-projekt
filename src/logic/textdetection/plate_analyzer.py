"""
License plate analysis and country/region detection.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Try to import API client, fallback to local patterns only if not available
try:
    from .api_client import PatternAPIClient
    API_AVAILABLE = True
except ImportError:
    logging.warning("API client not available, using local patterns only")
    API_AVAILABLE = False

# Try to import GLPD recognizer for enhanced recognition
try:
    from .glpd_recognizer import GLPDPlateAnalyzer, GLPDPlateRecognizer
    GLPD_AVAILABLE = True
    logging.info("GLPD (Global License Plate Dataset) integration available")
except ImportError:
    logging.info("GLPD integration not available, using standard patterns")
    GLPD_AVAILABLE = False


class PlateAnalyzer:
    def __init__(self, use_api: bool = True, use_glpd: bool = True):
        """
        Initialize PlateAnalyzer with optional API and GLPD integration.

        Args:
            use_api: Whether to use API for pattern updates (default: True)
            use_glpd: Whether to use GLPD (Global License Plate Dataset) for enhanced recognition (default: True)
        """
        self.use_api = use_api and API_AVAILABLE
        self.use_glpd = use_glpd and GLPD_AVAILABLE
        self.api_client = None
        self.glpd_analyzer = None

        # Initialize GLPD analyzer if available
        if self.use_glpd:
            try:
                self.glpd_analyzer = GLPDPlateAnalyzer()
                logging.info("PlateAnalyzer initialized with GLPD support (74 countries, 5M images dataset)")
            except Exception as e:
                logging.warning("Failed to initialize GLPD analyzer: %s", e)
                self.use_glpd = False

        if self.use_api:
            try:
                self.api_client = PatternAPIClient()
                logging.info("PlateAnalyzer initialized with API support")
            except Exception as e:
                logging.warning("Failed to initialize API client: %s", e)
                self.use_api = False
        self._load_patterns()

    def _load_patterns(self):
        """Load license plate patterns from API or use hardcoded fallback."""
        if self.use_api and self.api_client:
            try:
                api_patterns = self.api_client.get_enhanced_plate_patterns()
                if api_patterns:
                    logging.info("Loaded %d countries from API",
                                 len(api_patterns))
                    self.plate_patterns = {
                        **self._get_fallback_patterns(), **api_patterns}
                    return
            except Exception as e:
                logging.warning("Failed to load patterns from API: %s", e)

        # Use fallback patterns
        self.plate_patterns = self._get_fallback_patterns()
        logging.info("Using fallback patterns for %d countries",
                     len(self.plate_patterns))

    def _get_fallback_patterns(self) -> Dict[str, Dict]:
        """Get hardcoded fallback patterns."""
        return {
            # European patterns
            "Poland": {
                "patterns": [
                    # AB 12345, ABC 1234A
                    r"^[A-Z]{2,3}[\s\-]?[0-9]{3,5}[A-Z]?$",
                    r"^[A-Z]{1,2}[0-9]{4,5}$",  # A12345, AB12345
                ],
                "regions": {
                    "KR": "Kraków",
                    "WA": "Warszawa",
                    "GD": "Gdańsk",
                    "DW": "Wrocław",
                    "PO": "Poznań",
                    "SK": "Katowice",
                    "LU": "Lublin",
                    "BI": "Białystok",
                    "ZS": "Szczecin",
                    "RZ": "Rzeszów",
                    "TK": "Kielce",
                    "OP": "Opole",
                    "FZ": "Zielona Góra",
                    "NO": "Olsztyn",
                    "CT": "Toruń",
                }
            },
            "Germany": {
                "patterns": [
                    # B AB 1234
                    r"^[A-Z]{1,3}[\s\-]?[A-Z]{1,2}[\s\-]?[0-9]{1,4}[A-Z]?$",
                ],
                "regions": {
                    "B": "Berlin",
                    "M": "München",
                    "HH": "Hamburg",
                    "K": "Köln",
                    "F": "Frankfurt",
                    "S": "Stuttgart",
                    "D": "Düsseldorf",
                    "H": "Hannover",
                    "DD": "Dresden",
                    "L": "Leipzig"
                }
            },
            "France": {
                "patterns": [
                    # 123 AB 75
                    r"^[0-9]{1,4}[\s\-]?[A-Z]{2,3}[\s\-]?[0-9]{2}$",
                ],
                "regions": {
                    "75": "Paris",
                    "13": "Marseille",
                    "69": "Lyon",
                    "31": "Toulouse",
                    "06": "Nice",
                    "44": "Nantes",
                    "67": "Strasbourg",
                    "59": "Lille",
                    "33": "Bordeaux"
                }
            },
            "Italy": {
                "patterns": [
                    r"^[A-Z]{2}[\s\-]?[0-9]{3}[\s\-]?[A-Z]{2}$",  # AB 123 CD
                ],
                "regions": {
                    "RM": "Roma",
                    "MI": "Milano",
                    "NA": "Napoli",
                    "TO": "Torino",
                    "FI": "Firenze",
                    "BA": "Bari",
                    "PA": "Palermo",
                    "GE": "Genova",
                    "BO": "Bologna",
                    "VE": "Venezia"
                }
            },
            "Czech Republic": {
                "patterns": [
                    r"^[0-9][A-Z][0-9][\s\-]?[0-9]{4}$",  # 1A2 3456
                ],
                "regions": {
                    "A": "Praha",
                    "B": "Brno",
                    "C": "České Budějovice",
                    "E": "Pardubice",
                    "H": "Hradec Králové",
                    "J": "Jihlava",
                    "K": "Karlovy Vary",
                    "L": "Liberec",
                    "M": "Ostrava",
                    "P": "Plzeň",
                    "S": "Ústí nad Labem",
                    "T": "Olomouc",
                    "U": "Ústí nad Labem",
                    "Z": "Zlín"
                }
            },
            # UK patterns
            "United Kingdom": {
                "patterns": [
                    r"^[A-Z]{2}[0-9]{2}[\s\-]?[A-Z]{3}$",  # AB12 CDE
                ],
                "regions": {
                    "L": "London",
                    "M": "Manchester",
                    "B": "Birmingham",
                    "G": "Glasgow",
                    "E": "Edinburgh",
                    "V": "Severn Valley",
                    "W": "West of England",
                    "Y": "Yorkshire",
                    "N": "North"
                }
            },
            # US patterns
            "United States": {
                "patterns": [
                    r"^[A-Z0-9]{2,8}$",  # Varies by state
                ],
                "regions": {
                    "NY": "New York",
                    "CA": "California",
                    "TX": "Texas",
                    "FL": "Florida",
                    "IL": "Illinois",
                    "PA": "Pennsylvania",
                    "OH": "Ohio",
                    "MI": "Michigan",
                    "GA": "Georgia",
                    "NC": "North Carolina"
                }
            },
            # Other European countries
            "Spain": {
                "patterns": [
                    r"^[0-9]{4}[\s\-]?[A-Z]{3}$",  # 1234 ABC
                ],
                "regions": {
                    "M": "Madrid",
                    "B": "Barcelona",
                    "V": "Valencia",
                    "S": "Sevilla",
                    "BI": "Bilbao"
                }
            },
            "Ukraine": {
                "patterns": [
                    r"^[A-Z]{2}[\s\-]?[0-9]{4}[\s\-]?[A-Z]{2}$",  # AA 1234 BB
                ],
                "regions": {
                    "AA": "Kyiv",
                    "AB": "Vinnytsia",
                    "AC": "Volyn",
                    "AE": "Dnipropetrovsk",
                    "AH": "Donetsk",
                    "AI": "Zhytomyr",
                    "AK": "Zakarpattia",
                    "AM": "Zaporizhia",
                    "AO": "Ivano-Frankivsk",
                    "AP": "Kyiv Oblast",
                    "AT": "Kirovohrad",
                    "AX": "Kharkiv"
                }
            }
        }

    def analyze_plate(self, plate_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Analyze license plate text and return (country, region).
        Uses GLPD (Global License Plate Dataset) if available for enhanced accuracy.
        Returns (None, None) if no match found.
        """
        if not plate_text or len(plate_text.strip()) < 3:
            return None, None

        # Try GLPD analysis first if available
        if self.use_glpd and self.glpd_analyzer:
            try:
                glpd_results = self.glpd_analyzer.analyze_plates([plate_text])
                if glpd_results["countries"] and glpd_results["confidence"] > 0.5:
                    country = glpd_results["countries"][0]
                    region = glpd_results["regions"][0] if glpd_results["regions"] else None
                    logging.info("GLPD analysis: Plate '%s' -> %s, %s (confidence: %.2f)", 
                                plate_text, country, region, glpd_results["confidence"])
                    return country, region
            except Exception as e:
                logging.warning("GLPD analysis failed: %s", e)

        # Fallback to traditional pattern matching

        plate_clean = re.sub(r'[^A-Z0-9\s\-]', '', plate_text.upper().strip())
        logging.debug("Analyzing plate: '%s' (cleaned: '%s')",
                      plate_text, plate_clean)

        for country, data in self.plate_patterns.items():
            for pattern in data["patterns"]:
                if re.match(pattern, plate_clean):
                    logging.info("Plate '%s' matches %s pattern",
                                 plate_text, country)

                    # Try to extract region
                    region = self._extract_region(plate_clean, data["regions"])
                    if region:
                        logging.info("Detected region: %s", region)
                        return country, region
                    else:
                        return country, None

        logging.debug("No pattern match found for plate: '%s'", plate_text)
        return None, None

    def _extract_region(self, plate_clean: str, regions: Dict[str, str]) -> Optional[str]:
        """Extract region from plate based on region codes."""
        for code, region_name in regions.items():
            # Check if region code appears at the beginning of the plate
            if plate_clean.startswith(code):
                return region_name

            # For some formats, check if region code appears after numbers
            parts = re.split(r'[\s\-]', plate_clean)
            for part in parts:
                if part == code:
                    return region_name

        return None

    def get_countries_for_plates(self, plates: List[str]) -> Tuple[List[str], List[str]]:
        """
        Analyze multiple plates and return lists of countries and regions.
        """
        countries = []
        regions = []

        for plate in plates:
            country, region = self.analyze_plate(plate)
            if country:
                if country not in countries:
                    countries.append(country)
            if region:
                if region not in regions:
                    regions.append(region)

        return countries, regions

    def update_patterns_from_api(self) -> bool:
        """Update patterns from API and return success status."""
        if not self.use_api or not self.api_client:
            logging.warning("API not available for pattern updates")
            return False

        try:
            success = self.api_client.update_patterns()
            if success:
                self._load_patterns()  # Reload patterns after API update
                logging.info("Successfully updated patterns from API")
            return success
        except Exception as e:
            logging.error("Failed to update patterns from API: %s", e)
            return False

    def get_pattern_info(self) -> Dict[str, Any]:
        """Get information about current patterns and API status."""
        return {
            "api_enabled": self.use_api,
            "api_available": API_AVAILABLE,
            "total_countries": len(self.plate_patterns),
            "countries": list(self.plate_patterns.keys()),
            "last_api_update": getattr(self.api_client, 'cache', {}).get('last_updated', {}) if self.api_client else {}
        }

    def get_glpd_info(self) -> Dict[str, Any]:
        """Get information about GLPD (Global License Plate Dataset) integration."""
        if not self.use_glpd or not self.glpd_analyzer:
            return {"available": False, "reason": "GLPD integration not available"}
        
        return {
            "available": True,
            "dataset_info": {
                "name": "Global License Plate Dataset (GLPD)",
                "description": "Enhanced license plate recognition using 5 million images from 74 countries",
                "publication_date": "March 2024",
                "total_images": "5,000,000",
                "countries_count": 74,
                "features": ["segmentation", "text recognition", "country detection", "region extraction"]
            },
            "supported_countries": self.glpd_analyzer.recognizer.get_supported_countries() if self.glpd_analyzer else [],
            "capabilities": [
                "Enhanced pattern recognition",
                "Multi-country license plate support", 
                "Regional code extraction",
                "Confidence scoring",
                "Fallback pattern matching"
            ]
        }
    
    def analyze_plates_batch(self, plate_texts: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple license plates using GLPD for improved accuracy.
        
        Args:
            plate_texts: List of license plate texts to analyze
            
        Returns:
            Comprehensive analysis results
        """
        if not plate_texts:
            return {"countries": [], "regions": [], "confidence": 0.0}
        
        # Use GLPD for batch analysis if available
        if self.use_glpd and self.glpd_analyzer:
            try:
                results = self.glpd_analyzer.analyze_plates(plate_texts)
                results["method"] = "GLPD (Global License Plate Dataset)"
                results["dataset_size"] = "5M images, 74 countries"
                return results
            except Exception as e:
                logging.warning("GLPD batch analysis failed: %s", e)
        
        # Fallback to individual analysis
        countries = []
        regions = []
        
        for plate_text in plate_texts:
            country, region = self.analyze_plate(plate_text)
            if country and country not in countries:
                countries.append(country)
            if region and region not in regions:
                regions.append(region)
        
        return {
            "countries": countries,
            "regions": regions,
            "confidence": 0.7 if countries else 0.0,
            "analysis_count": len(plate_texts),
            "method": "Traditional pattern matching"
        }

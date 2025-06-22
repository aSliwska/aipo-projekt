"""
Simplified text detection pipeline.
Uses robust detection during frame analysis and returns detected countries at the end.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from logic.textdetection.text_objects_detection import process_frame
from logic.textdetection.object_detector import MultiObjectDetector
from logic.textdetection.plate_recognizer import PlateRecognizer
from logic.textdetection.text_extractor import TextExtractor
from logic.textdetection.text_classifier import TextClassifier
from logic.textdetection.plate_analyzer import PlateAnalyzer
from logic.textdetection.road_sign_analyzer import RoadSignAnalyzer

class DeferredTextDetection:
    """
    Simplified text detection that stores detected plates and signs during frame analysis.
    Returns detected countries at the end without any GLPD downloads.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.detected_plates: List[str] = []
        self.detected_plates_meta: List[dict] = []  # Store plate, frame, country guess
        self.detected_signs: List[str] = []
        self.detected_sign_countries: Set[str] = set()
        self.detected_billboard_locations: List[str] = []
        self.frame_count = 0
        # Robust detection pipeline components - initialize once
        self.detector = MultiObjectDetector("yolov8n.pt")
        self.plate_recognizer = PlateRecognizer()
        self.text_extractor = TextExtractor()
        self.text_clf = TextClassifier()
        self.road_sign_analyzer = RoadSignAnalyzer()
        self.plate_analyzer = PlateAnalyzer()

    def analyze_frame_lightweight(self, frame) -> Dict[str, Any]:
        """
        Robust frame analysis using the original detection pipeline.
        Detects plates and signs reliably and stores results.
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            Analysis results with detected countries
        """
        self.frame_count += 1
        
        # Use the robust detection pipeline from text_objects_detection.py
        result = process_frame(
            frame,
            self.frame_count,
            self.detector,
            self.plate_recognizer,
            self.text_extractor,
            self.text_clf,
            self.road_sign_analyzer,
            self.plate_analyzer,
            save_to_file=False,
            outline_objects=False,
        )
        
        # Collect all detected data
        frame_plate_countries = []
        frame_plate_regions = []
        frame_sign_countries = []
        frame_billboard_locations = []
        
        # Process each detection from the robust pipeline
        for det in result.get("detections", []): 
            # License plates
            if det.get("plates"):
                for plate in det["plates"]:
                    self.detected_plates.append(plate)
                    countries = det.get("plate_countries", [])
                    regions = det.get("plate_regions", [])
                    
                    # Store metadata
                    self.detected_plates_meta.append({
                        "plate": plate,
                        "frame": self.frame_count,
                        "countries": countries,
                        "regions": regions
                    })
                    
                    # Add to frame results
                    frame_plate_countries.extend(countries)
                    frame_plate_regions.extend(regions)
                    
                    self.logger.debug(f"Frame {self.frame_count}: Plate '{plate}' -> countries: {countries}, regions: {regions}")
            
            # Road signs
            if det.get("sign_countries"):
                sign_countries = det["sign_countries"]
                self.detected_sign_countries.update(sign_countries)
                frame_sign_countries.extend(sign_countries)
                self.logger.debug(f"Frame {self.frame_count}: Sign countries: {sign_countries}")
            
            # Billboard/text locations
            if det.get("billboard_locations"):
                billboard_locs = det["billboard_locations"]
                self.detected_billboard_locations.extend(billboard_locs)
                frame_billboard_locations.extend(billboard_locs)
                self.logger.debug(f"Frame {self.frame_count}: Billboard locations: {billboard_locs}")
        
        # Return results for this frame
        return {
            "plate_countries": frame_plate_countries,
            "plate_regions": frame_plate_regions,
            "sign_countries": frame_sign_countries,
            "billboard_locations": frame_billboard_locations,
            "deferred": False,  # We're doing real detection now
        }
    
    def finalize_analysis(self) -> Dict[str, Any]:
        """
        Finalize analysis by returning all detected countries.
        No GLPD downloads - just return what was actually detected.
        
        Returns:
            Final analysis results with detected countries
        """
        self.logger.info("=== FINALIZING ANALYSIS ===")
        self.logger.info(f"Processed {self.frame_count} frames")
        self.logger.info(f"Detected {len(self.detected_plates)} license plates")
        self.logger.info(f"Detected {len(self.detected_sign_countries)} unique sign countries")
        self.logger.info(f"Detected {len(self.detected_billboard_locations)} billboard locations")
        
        # Collect all unique countries from detected plates
        all_plate_countries = set()
        all_plate_regions = set()
        for meta in self.detected_plates_meta:
            all_plate_countries.update(meta.get("countries", []))
            all_plate_regions.update(meta.get("regions", []))
        
        # Log what we found
        self.logger.info(f"Unique plate countries: {sorted(all_plate_countries)}")
        self.logger.info(f"Unique plate regions: {sorted(all_plate_regions)}")
        self.logger.info(f"Unique sign countries: {sorted(self.detected_sign_countries)}")
        
        # Return the detected countries - this is what the system needs
        final_results = {
            "plate_countries": sorted(all_plate_countries),
            "plate_regions": sorted(all_plate_regions),
            "sign_countries": sorted(self.detected_sign_countries),
            "billboard_locations": self.detected_billboard_locations,
            "total_plates_detected": len(self.detected_plates),
            "total_frames_processed": self.frame_count,
            "status": "completed"
        }
        
        self.logger.info("=== ANALYSIS FINALIZATION COMPLETE ===")
        return final_results


# Global instance for text detection
_text_detector = None

def get_text_detector() -> DeferredTextDetection:
    """Get global text detection instance."""
    global _text_detector
    if _text_detector is None:
        _text_detector = DeferredTextDetection()
    return _text_detector

def analyze_frame_for_location(frame):
    """
    Robust frame analysis using the original text detection pipeline.
    Compatible with existing interface.
    """
    detector = get_text_detector()
    return detector.analyze_frame_lightweight(frame)

def finalize_text_detection() -> Dict[str, Any]:
    """
    Finalize text detection and return detected countries.
    Call this at the end of video analysis.
    """
    detector = get_text_detector()
    return detector.finalize_analysis()

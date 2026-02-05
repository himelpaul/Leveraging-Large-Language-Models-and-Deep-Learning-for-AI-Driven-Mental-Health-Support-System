"""
Crisis Detection Module
Identifies high-risk content using keyword matching and regex patterns.
"""

import json
import re
import os
import logging
from typing import Tuple, List, Dict
from datetime import datetime

# Set up logging (WARNING level to avoid startup noise)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class CrisisDetector:
    def __init__(self, rules_path="detection/rules.json"):
        self.rules = self._load_rules(rules_path)
        self.detection_log = []

    def _load_rules(self, path):
        if not os.path.exists(path):
            logger.warning(f"Rules file not found: {path}. Using empty rules.")
            return {"high": [], "medium": [], "low": [], "regex_patterns": {"high": [], "medium": []}}
        
        with open(path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        
        # Compile regex patterns
        if "regex_patterns" in rules:
            for level in ["high", "medium"]:
                if level in rules["regex_patterns"]:
                    rules["regex_patterns"][level] = [
                        re.compile(pattern) for pattern in rules["regex_patterns"][level]
                    ]
        
        logger.info(f"Loaded {len(rules.get('high', []))} high-risk phrases, "
                   f"{len(rules.get('medium', []))} medium-risk phrases")
        return rules

    def classify_risk(self, text: str, llm_fallback_fn=None) -> Tuple[str, float, List[str]]:
        """
        Classify risk level of input text.
        
        Args:
            text: User input text
            llm_fallback_fn: Optional LLM function for ambiguous cases
            
        Returns:
            Tuple of (risk_level, confidence, matched_phrases)
            - risk_level: "high", "medium", "low", or "safe"
            - confidence: float between 0 and 1
            - matched_phrases: list of matched phrases/patterns
        """
        text_lower = text.lower()
        matched_phrases = []
        
        # Check high-risk keywords first
        for phrase in self.rules.get("high", []):
            if phrase in text_lower:
                matched_phrases.append(phrase)
        
        # Check high-risk regex patterns
        if "regex_patterns" in self.rules and "high" in self.rules["regex_patterns"]:
            for pattern in self.rules["regex_patterns"]["high"]:
                if pattern.search(text):
                    matched_phrases.append(f"pattern: {pattern.pattern[:30]}...")
        
        if matched_phrases:
            confidence = min(1.0, 0.7 + (len(matched_phrases) * 0.1))
            self._log_detection("high", confidence, matched_phrases, text)
            return "high", confidence, matched_phrases
        
        # Check medium-risk keywords
        for phrase in self.rules.get("medium", []):
            if phrase in text_lower:
                matched_phrases.append(phrase)
        
        # Check medium-risk regex patterns
        if "regex_patterns" in self.rules and "medium" in self.rules["regex_patterns"]:
            for pattern in self.rules["regex_patterns"]["medium"]:
                if pattern.search(text):
                    matched_phrases.append(f"pattern: {pattern.pattern[:30]}...")
        
        if matched_phrases:
            confidence = min(1.0, 0.6 + (len(matched_phrases) * 0.1))
            self._log_detection("medium", confidence, matched_phrases, text)
            return "medium", confidence, matched_phrases
        
        # Check low-risk keywords
        for phrase in self.rules.get("low", []):
            if phrase in text_lower:
                matched_phrases.append(phrase)
        
        if matched_phrases:
            confidence = min(1.0, 0.5 + (len(matched_phrases) * 0.05))
            self._log_detection("low", confidence, matched_phrases, text)
            return "low", confidence, matched_phrases
        
        # If LLM fallback is provided, use it for ambiguous cases
        if llm_fallback_fn:
            llm_result = llm_fallback_fn(text)
            if llm_result:
                return llm_result, 0.5, ["llm_classification"]
        
        return "safe", 1.0, []
    
    def _log_detection(self, level: str, confidence: float, matched: List[str], text: str):
        """Log detection event for analysis."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "confidence": confidence,
            "matched": matched,
            "text_length": len(text),
            "text_preview": text[:100] if len(text) > 100 else text
        }
        self.detection_log.append(log_entry)
        
        # Log to file for thesis analysis
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "crisis_detections.jsonl")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.warning(f"Crisis detection: level={level}, confidence={confidence:.2f}, "
                      f"matched={len(matched)} phrases")
    
    def get_detection_stats(self) -> Dict:
        """Get statistics about detections for thesis reporting."""
        stats = {
            "total_detections": len(self.detection_log),
            "high_risk": sum(1 for d in self.detection_log if d["level"] == "high"),
            "medium_risk": sum(1 for d in self.detection_log if d["level"] == "medium"),
            "low_risk": sum(1 for d in self.detection_log if d["level"] == "low"),
        }
        
        if stats["total_detections"] > 0:
            avg_confidence = sum(d["confidence"] for d in self.detection_log) / stats["total_detections"]
            stats["average_confidence"] = avg_confidence
        
        return stats


# Singleton instance
_detector = None

def classify_risk(text: str) -> Tuple[str, float, List[str]]:
    """
    Convenience function for quick risk classification.
    
    Returns:
        Tuple of (risk_level, confidence, matched_phrases)
    """
    global _detector
    if _detector is None:
        _detector = CrisisDetector()
    return _detector.classify_risk(text)


def get_detector_stats() -> Dict:
    """Get detection statistics from singleton detector."""
    global _detector
    if _detector is None:
        return {"total_detections": 0}
    return _detector.get_detection_stats()


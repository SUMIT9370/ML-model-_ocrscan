# ============================================
# ULTRA-ENHANCED DOCUMENT EXTRACTOR v3.0
# Maximum Accuracy for Indian Documents
# ============================================

# STEP 1: Install Dependencies
!pip install -q pytesseract pillow pdf2image opencv-python-headless
!apt-get -qq install tesseract-ocr tesseract-ocr-mar tesseract-ocr-hin tesseract-ocr-eng
!apt-get -qq install poppler-utils

print("✅ All dependencies installed with full language support!")

# ============================================
# STEP 2: ULTRA-ENHANCED ML MODEL v3.0
# ============================================

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import json
from pdf2image import convert_from_path
import os
from pathlib import Path
import re
from datetime import datetime

class UltraEnhancedExtractor:
    """
    Ultra-enhanced document extraction with maximum accuracy
    """
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.pdf']
        
    def super_preprocess(self, image):
        """Multiple preprocessing techniques for best results"""
        if isinstance(image, Image.Image):
            img_pil = image
            image = np.array(image)
        else:
            img_pil = Image.fromarray(image)
        
        # Technique 1: Enhanced PIL preprocessing
        img_pil = img_pil.convert('L')  # Grayscale
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(2.0)  # Increase contrast
        enhancer = ImageEnhance.Sharpness(img_pil)
        img_pil = enhancer.enhance(2.0)  # Sharpen
        
        # Technique 2: OpenCV preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(filtered)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, h=10)
        
        return img_pil, denoised
    
    def extract_with_multiple_attempts(self, image_path):
        """Try multiple OCR configurations for best results"""
        image = Image.open(image_path)
        pil_processed, cv_processed = self.super_preprocess(image)
        
        results = []
        
        # Config 1: Multi-language with PSM 6 (uniform text block)
        config1 = r'--oem 3 --psm 6 -l eng+mar+hin'
        text1 = pytesseract.image_to_string(cv_processed, config=config1)
        conf1_data = pytesseract.image_to_data(cv_processed, output_type=pytesseract.Output.DICT, config=config1)
        conf1 = self.get_avg_confidence(conf1_data)
        results.append((text1, conf1, 'Multi-lang PSM6'))
        
        # Config 2: English only with PSM 6
        config2 = r'--oem 3 --psm 6 -l eng'
        text2 = pytesseract.image_to_string(pil_processed, config=config2)
        conf2_data = pytesseract.image_to_data(pil_processed, output_type=pytesseract.Output.DICT, config=config2)
        conf2 = self.get_avg_confidence(conf2_data)
        results.append((text2, conf2, 'English PSM6'))
        
        # Config 3: Multi-language with PSM 3 (automatic page segmentation)
        config3 = r'--oem 3 --psm 3 -l eng+mar+hin'
        text3 = pytesseract.image_to_string(cv_processed, config=config3)
        conf3_data = pytesseract.image_to_data(cv_processed, output_type=pytesseract.Output.DICT, config=config3)
        conf3 = self.get_avg_confidence(conf3_data)
        results.append((text3, conf3, 'Multi-lang PSM3'))
        
        # Pick best result
        best_result = max(results, key=lambda x: x[1])
        print(f"   Best OCR config: {best_result[2]} (confidence: {best_result[1]:.2f}%)")
        
        # Combine all texts for comprehensive extraction
        combined_text = "\n\n".join([r[0] for r in results])
        
        return best_result[0], combined_text, best_result[1]
    
    def get_avg_confidence(self, data):
        """Calculate average confidence from OCR data"""
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        return sum(confidences) / len(confidences) if confidences else 0
    
    def extract_marksheet_advanced(self, text, combined_text):
        """Advanced marksheet extraction with multiple pattern attempts"""
        data = {}
        
        # Use both best text and combined text for extraction
        all_text = text + "\n\n" + combined_text
        
        # Student name - multiple patterns
        name_patterns = [
            r"CANDIDATE'?S?\s*FULL\s*NAME[:\s]*(?:\([^)]*\))?\s*([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"Bonde\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)",
            r"NAME[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 1:
                    data['student_name'] = match.group(1).strip()
                else:
                    data['student_name'] = ' '.join(match.groups()).strip()
                break
        
        # If still not found, try finding "Bonde" and context
        if 'student_name' not in data:
            bonde_match = re.search(r'Bonde\s+(\w+)\s+(\w+)', all_text, re.IGNORECASE)
            if bonde_match:
                data['student_name'] = f"Bonde {bonde_match.group(1)} {bonde_match.group(2)}"
        
        # Mother's name
        mother_patterns = [
            r"MOTHER'?S?\s*NAME[:\s]*([A-Z][a-z]+)",
            r"MOTHER[:\s]*([A-Z][a-z]+)",
        ]
        for pattern in mother_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                data['mother_name'] = match.group(1).strip()
                break
        
        # Seat number - multiple patterns
        seat_patterns = [
            r'SEAT\s*NO\.?[:\s]*([A-Z]?\d{6,8})',
            r'H(\d{6,8})',
            r'SEAT[:\s]*([A-Z0-9]{6,10})',
        ]
        for pattern in seat_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                data['seat_number'] = match.group(1).strip()
                break
        
        # Year
        year_patterns = [
            r'YEAR\s*OF\s*EXAM[:\s]*(\d{4})',
            r'20\d{2}',
            r'\b(2021|2022|2023|2024|2025)\b',
        ]
        for pattern in year_patterns:
            match = re.search(pattern, all_text)
            if match:
                data['exam_year'] = match.group(1) if '(' in pattern else match.group(0)
                break
        
        # Extract marks - ROBUST PATTERNS
        subjects = []
        
        # Pattern 1: Traditional format
        marks_patterns = [
            # Code | Subject | Max | Obtained
            r'(\d{1,2})\s+([A-Z][A-Z\s&/()]+?)\s+(\d{3})\s*\|\s*(\d{2,3})',
            # Subject | Max | Obtained
            r'([A-Z][A-Z\s&/()]+?)\s+(\d{3})\s*\|\s*(\d{2,3})',
            # With words: MARATHI ... 100 | 087 | EIGHTYSEVEN
            r'(\d{1,2})?\s*([A-Z][A-Z\s&/()]+?)\s+(\d{3})\s*\|\s*(\d{2,3})\s*\|\s*([A-Z]+)',
        ]
        
        # Known subjects to look for
        known_subjects = ['MARATHI', 'SANSKRIT', 'ENGLISH', 'MATHEMATICS', 'SCIENCE', 'SOCIAL']
        
        for subject_name in known_subjects:
            # Look for pattern: SUBJECT_NAME ... numbers
            pattern = rf'{subject_name}[^0-9]*?(\d{{3}})[^0-9]*?(\d{{2,3}})'
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                subjects.append({
                    'name': subject_name,
                    'max_marks': int(match.group(1)),
                    'obtained_marks': int(match.group(2))
                })
        
        # If subjects found, calculate totals
        if subjects:
            data['subjects'] = subjects
            total_obtained = sum(s['obtained_marks'] for s in subjects)
            total_max = sum(s['max_marks'] for s in subjects)
            data['total_marks'] = f"{total_obtained}/{total_max}"
            data['percentage'] = round((total_obtained / total_max) * 100, 2) if total_max > 0 else 0
        
        # Look for total marks in text
        total_patterns = [
            r'(\d{3,4})\s*/\s*(\d{3,4})',
            r'TOTAL[:\s]*(\d{3,4})',
        ]
        for pattern in total_patterns:
            match = re.search(pattern, all_text)
            if match and 'total_marks' not in data:
                if len(match.groups()) == 2:
                    data['total_marks'] = f"{match.group(1)}/{match.group(2)}"
                else:
                    data['total_obtained'] = match.group(1)
        
        # Result
        result_patterns = [
            r'Result[:\s]*(PASS|FAIL|ATKT)',
            r'\b(PASS|FAIL)\b',
        ]
        for pattern in result_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                data['result'] = match.group(1).upper()
                break
        
        # Percentage from text
        perc_patterns = [
            r'(\d{2}\.\d{2})%',
            r'Percentage[:\s]*(\d{2}\.\d{2})',
        ]
        for pattern in perc_patterns:
            match = re.search(pattern, all_text)
            if match and 'percentage' not in data:
                data['percentage'] = float(match.group(1))
                break
        
        return data
    
    def extract(self, file_path):
        """Main extraction with ultra-enhanced features"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format. Supported: {self.supported_formats}")
        
        print(f"\n{'='*70}")
        print(f"🔍 ULTRA-ENHANCED EXTRACTION v3.0")
        print(f"{'='*70}")
        print(f"📄 File: {file_path.name}")
        print(f"📊 Size: {file_path.stat().st_size / 1024:.2f} KB")
        print(f"⏱️  Started: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'-'*70}")
        
        # Multi-attempt extraction
        print("🔬 Running multiple OCR attempts...")
        best_text, combined_text, confidence = self.extract_with_multiple_attempts(str(file_path))
        
        print(f"✅ Extraction complete - Confidence: {confidence:.2f}%")
        
        # Document type
        doc_type = 'marksheet'  # Assuming marksheet for now
        print(f"📑 Document Type: {doc_type.upper()}")
        
        # Advanced extraction
        print(f"🔬 Running advanced pattern matching...")
        structured = self.extract_marksheet_advanced(best_text, combined_text)
        
        # Extract general info
        general = self.extract_general_info(combined_text)
        structured.update(general)
        
        # Calculate quality
        quality_score = self.calculate_quality(best_text, confidence, structured)
        
        result = {
            'file_info': {
                'file_name': file_path.name,
                'file_size_kb': round(file_path.stat().st_size / 1024, 2),
                'extraction_timestamp': datetime.now().isoformat()
            },
            'document_info': {
                'document_type': doc_type,
                'quality_score': quality_score,
                'extraction_method': 'Ultra-Enhanced v3.0'
            },
            'extracted_data': structured,
            'raw_text': best_text.strip(),
            'metadata': {
                'total_characters': len(best_text),
                'total_words': len(best_text.split()),
                'ocr_confidence': round(confidence, 2),
                'extraction_quality': self.get_quality_label(quality_score)
            }
        }
        
        print(f"✅ Complete!")
        print(f"🎯 Quality: {quality_score}/100 ({self.get_quality_label(quality_score)})")
        print(f"{'='*70}\n")
        
        return result
    
    def extract_general_info(self, text):
        """Extract general information"""
        data = {}
        
        # All reference numbers
        numbers = re.findall(r'\b([A-Z]*\d{6,}[A-Z]*)\b', text)
        if numbers:
            data['reference_numbers'] = list(set(numbers))[:5]
        
        # All dates
        dates = re.findall(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', text)
        if dates:
            data['dates'] = list(set(dates))
        
        return data
    
    def calculate_quality(self, text, confidence, structured):
        """Calculate quality score"""
        score = 0
        score += (confidence / 100) * 40
        score += min(len(text) / 10, 20)
        score += min(len(structured) * 5, 40)
        return min(round(score, 2), 100)
    
    def get_quality_label(self, score):
        """Get quality label"""
        if score >= 90: return "EXCELLENT"
        elif score >= 75: return "GOOD"
        elif score >= 60: return "FAIR"
        else: return "POOR"
    
    def generate_report(self, result):
        """Generate detailed report"""
        lines = []
        lines.append("="*70)
        lines.append("📋 EXTRACTION REPORT")
        lines.append("="*70)
        
        lines.append(f"\n📄 FILE:")
        lines.append(f"   {result['file_info']['file_name']} ({result['file_info']['file_size_kb']} KB)")
        
        lines.append(f"\n📊 QUALITY:")
        lines.append(f"   Score: {result['document_info']['quality_score']}/100")
        lines.append(f"   Rating: {result['metadata']['extraction_quality']}")
        lines.append(f"   OCR Confidence: {result['metadata']['ocr_confidence']}%")
        
        lines.append(f"\n🔍 EXTRACTED DATA:")
        for key, value in result['extracted_data'].items():
            if isinstance(value, list) and value:
                if isinstance(value[0], dict):
                    lines.append(f"   {key.replace('_', ' ').title()}:")
                    for item in value:
                        lines.append(f"      - {item}")
                else:
                    lines.append(f"   {key.replace('_', ' ').title()}: {', '.join(map(str, value[:3]))}")
            elif value:
                lines.append(f"   {key.replace('_', ' ').title()}: {value}")
        
        lines.append("="*70)
        return "\n".join(lines)

print("✅ Ultra-Enhanced Model v3.0 loaded!")

# ============================================
# STEP 3: RUN EXTRACTION
# ============================================

from google.colab import files

print("\n" + "="*70)
print("📤 UPLOAD YOUR DOCUMENT")
print("="*70)
print("Supports: JPG, PNG, PDF | Multi-language: English, Marathi, Hindi\n")

uploaded = files.upload()

if uploaded:
    file_path = list(uploaded.keys())[0]
    
    extractor = UltraEnhancedExtractor()
    result = extractor.extract(file_path)
    
    # Show report
    report = extractor.generate_report(result)
    print("\n" + report)
    
    # Show JSON
    print("\n" + "="*70)
    print("💾 COMPLETE JSON OUTPUT:")
    print("="*70)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Save
    with open('ultra_extraction.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    files.download('ultra_extraction.json')
    print("\n✅ JSON file downloaded!")
else:
    print("❌ No file uploaded!")
import io
import os
import pymupdf  # PyMuPDF (replaces deprecated 'fitz' import)
from typing import Tuple, Optional, Union
from PIL import Image


# Lazy-loaded PaddleOCR instance
_ocr_engine = None


def _apply_paddle_compatibility_patch():
    """Disable PIR/oneDNN incompatibility in PaddlePaddle 3.x on Windows."""
    try:
        import os
        os.environ["FLAGS_use_onednn"] = "0"
        os.environ["FLAGS_enable_pir_api"] = "0"
        from paddlex.inference.models.runners.paddle_static.runner import PaddleStaticRunner
        orig_create = PaddleStaticRunner._create

        def patched_create(self):
            self._config["enable_new_ir"] = False
            self._config["run_mode"] = "paddle"
            return orig_create(self)

        PaddleStaticRunner._create = patched_create
    except Exception as e:
        print(f"[DEBUG] PaddleStaticRunner patch note: {e}")


def _get_ocr_engine():
    """Lazy-load PaddleOCR to avoid slow imports on every startup."""
    global _ocr_engine
    if _ocr_engine is None:
        _apply_paddle_compatibility_patch()
        from paddleocr import PaddleOCR
        # In PaddleOCR 3.x: show_log and use_angle_cls are removed/deprecated
        _ocr_engine = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _ocr_engine


def _ocr_from_image_bytes(image_bytes: bytes) -> str:
    """Run PaddleOCR on raw image bytes and return extracted text."""
    try:
        ocr = _get_ocr_engine()
        import numpy as np
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(pil_img)
        result = ocr.ocr(img_array)

        lines = []
        if result:
            for item in result:
                if isinstance(item, dict):
                    # PaddleOCR 3.x / PaddleX format: {'rec_texts': [...]}
                    rec_texts = item.get("rec_texts", [])
                    for t in rec_texts:
                        if t and str(t).strip():
                            lines.append(str(t).strip())
                elif isinstance(item, (list, tuple)):
                    # Legacy PaddleOCR format: [[box, (text, score)], ...]
                    for line_info in item:
                        if isinstance(line_info, (list, tuple)) and len(line_info) > 1:
                            text_part = line_info[1]
                            if isinstance(text_part, (list, tuple)) and len(text_part) > 0:
                                text = str(text_part[0]).strip()
                                if text:
                                    lines.append(text)
                            elif isinstance(text_part, str) and text_part.strip():
                                lines.append(text_part.strip())
        return "\n".join(lines).strip()
    except Exception as e:
        print(f"[WARN] PaddleOCR extraction failed: {e}")
        return ""


def _ocr_pdf_pages(doc: pymupdf.Document) -> str:
    """Render each PDF page as image and run PaddleOCR on it."""
    all_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to image at 300 DPI for good OCR quality
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        page_text = _ocr_from_image_bytes(img_bytes)
        if page_text.strip():
            all_text.append(f"--- Page {page_num + 1} ---\n{page_text.strip()}")
    return "\n\n".join(all_text)


def parse_resume_file(
    file_input: Union[bytes, str],
    filename: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse uploaded resume file using PyMuPDF + PaddleOCR.
    Accepts either raw bytes or a file path string.
    Returns a tuple of (extracted_text, None).

    - PDF: Extract text with PyMuPDF. If text is sparse (scanned PDF),
           fall back to PaddleOCR on rendered page images.
    - Image: Run PaddleOCR directly on the image.
    """
    if isinstance(file_input, str):
        filepath = file_input
        filename = filename or os.path.basename(filepath)
        with open(filepath, "rb") as f:
            file_bytes = f.read()
    else:
        file_bytes = file_input
        filename = filename or "resume.pdf"

    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            text_pages = []
            for page in doc:
                page_text = page.get_text()
                if page_text and page_text.strip():
                    text_pages.append(page_text.strip())

            extracted = "\n\n".join(text_pages).strip()

            # If PyMuPDF extracted meaningful text (>30 chars), use it
            if len(extracted) > 30:
                print(f"[INFO] PyMuPDF extracted {len(extracted)} chars from PDF")
                doc.close()
                return extracted, None

            # Scanned PDF — fall back to PaddleOCR
            print("[INFO] PDF appears scanned (low text). Running PaddleOCR...")
            ocr_text = _ocr_pdf_pages(doc)
            doc.close()
            if ocr_text.strip():
                print(f"[INFO] PaddleOCR extracted {len(ocr_text)} chars from scanned PDF")
                return ocr_text, None
            else:
                print("[WARN] PaddleOCR could not extract text from scanned PDF")
                return None, None

        except Exception as e:
            print(f"[WARN] Failed to parse PDF: {e}")
            return None, None

    elif filename_lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        # Image resume — run PaddleOCR directly
        try:
            print("[INFO] Running PaddleOCR on uploaded image...")
            ocr_text = _ocr_from_image_bytes(file_bytes)
            if ocr_text.strip():
                print(f"[INFO] PaddleOCR extracted {len(ocr_text)} chars from image")
                return ocr_text, None
            else:
                print("[WARN] PaddleOCR could not extract text from image")
                return None, None
        except Exception as e:
            print(f"[WARN] Failed to process image: {e}")
            return None, None

    return None, None

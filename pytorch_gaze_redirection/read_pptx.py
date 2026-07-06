import zipfile
import xml.etree.ElementTree as ET
import os
import re

pptx_path = r'd:\AI\EDTalk\pytorch_gaze_redirection\說明PPT.pptx'

# PPTX namespace mapping
ns = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
}

def get_text_from_xml_element(elem):
    """Extract all text from an XML element recursively."""
    texts = []
    for t in elem.iter():
        if t.tag.endswith('}t') and t.text:
            texts.append(t.text)
    return ''.join(texts)

def extract_slide_content(slide_xml):
    """Extract text from slide shapes."""
    root = ET.fromstring(slide_xml)
    texts = []
    for sp in root.findall('.//p:sp', ns):
        # Skip placeholders and shapes without text body
        text = ''
        for txBody in sp.findall('.//p:txBody', ns):
            text += get_text_from_xml_element(txBody)
        if text.strip():
            texts.append(text.strip())
    return '\n'.join(texts)

def extract_notes_content(notes_xml):
    """Extract notes text."""
    root = ET.fromstring(notes_xml)
    texts = []
    for cSld in root.findall('.//p:cSld', ns):
        text = get_text_from_xml_element(cSld)
        if text.strip():
            texts.append(text.strip())
    return '\n'.join(texts)

with zipfile.ZipFile(pptx_path, 'r') as z:
    # List all files
    all_files = z.namelist()
    
    # Get slide files sorted
    slide_files = sorted([f for f in all_files if f.startswith('ppt/slides/slide') and f.endswith('.xml')],
                         key=lambda x: int(re.search(r'slide(\d+)\.xml', x).group(1)))
    
    print(f"Total slides: {len(slide_files)}\n")
    print("=" * 80)
    
    for slide_file in slide_files:
        slide_num = int(re.search(r'slide(\d+)\.xml', slide_file).group(1))
        slide_xml = z.read(slide_file)
        slide_text = extract_slide_content(slide_xml)
        
        # Try to find corresponding notes slide
        notes_file = f'ppt/notesSlides/notesSlide{slide_num}.xml'
        notes_text = ''
        if notes_file in all_files:
            notes_xml = z.read(notes_file)
            notes_text = extract_notes_content(notes_xml)
        
        print(f"\n## Slide {slide_num}")
        print("-" * 80)
        if slide_text:
            print("[CONTENT]")
            print(slide_text)
        if notes_text:
            print("\n[NOTES]")
            print(notes_text)
        print("\n" + "=" * 80)

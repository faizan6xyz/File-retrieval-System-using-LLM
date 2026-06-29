import os 
from data_coversion.image_to_text_chunking import image_text
from data_coversion.pdf_to_text_chunking import save_pdf
from data_coversion.docx_to_text_chuncking import save_docx
def addfile(name):
    if name.lower().endswith('.docx'):
        save_docx(name)
    elif name.lower().endswith('.pdf'):
        save_pdf(name)
    elif name.lower().endswith(('.png', '.jpeg', '.jpg')): 
        image_text(name)
    else:
        print(f"Unsupported file format: {name}")
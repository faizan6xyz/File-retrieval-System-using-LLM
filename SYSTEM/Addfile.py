import os 
from data_coversion.image_to_text_chunking import image_text
from data_coversion.pdf_to_text_chunking import save_pdf
from data_coversion.docx_to_text_chuncking import save_docx
from data_coversion.video_to_image_byCV2 import video_text
def addfile(name):
    if name.lower().endswith('.docx'):
        save_docx(name)
    elif name.lower().endswith('.pdf'):
        save_pdf(name)
    elif name.lower().endswith(('.png', '.jpeg', '.jpg')): 
        image_text(name)
    elif name.lower().endswith(".mp4",):
        video_text(name)
    else:
        print(f"Unsupported file format: {name}")
        
if __name__ == "__main__" :
    files = os.listdir("SYSTEM/Data")
    for file in files :
        addfile(file)
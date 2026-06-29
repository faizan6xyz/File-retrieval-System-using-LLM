import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vison_model_Qwen3b_4bit_qunatization import describe_image
from Rag_create import build_index_from_text
def image_text(name,folder="SYSTEM/data"):
    image_name = os.path.join(folder, name)
    result, source = describe_image(image_name)
    if not result:
        print(f"Nothing returned for {image_name} by the Qwen")
        return None 
    print(f"About : {result}")
    build_index_from_text(result, source_name=source)
    print(f"{image_name} saved successfully")
    return result  
image_text("h.png")
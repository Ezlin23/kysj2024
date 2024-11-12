
import os
from urllib import response
from PIL import Image
import fitz
import pandas as pd
import json

def pdf_to_pngPath(pdf_path):
    # Extract the base name of the PDF file without extension
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Create a directory with the same name as the PDF file
    output_folder = os.path.join(os.path.dirname(pdf_path), pdf_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    # Iterate through each page
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        image_path = os.path.join(output_folder, f'{pdf_name}_page_{page_num + 1}.png')
        pix.save(image_path)
        print(f'Saved: {image_path}')

    pdf_document.close()

    return output_folder

def pdf_to_images(pdfPath):

    path = pdf_to_pngPath(pdfPath)

    images = []

    for image_file in sorted(os.listdir(path)):
        if image_file.endswith(".png"):
            images.append(Image.open(os.path.join(path, image_file)))

    return images

def images_to_user(
    images=list, 
    instruction = """
        As a paper-reading assistant, please read the provided paper and give a detailed description of the solution-based inorganic synthesis processes it discusses. If there are multiple syntheses, provide separate summaries for each. For multi-step syntheses, describe each step in detail. Limit your response to a maximum of 16,382 new tokens.

        Please structure your response **STRICTLY** using the following format. **DO NOT** include any content outside of this format, and **DO NOT** add extra headings or sub-points within the "Steps" section:

        # Target 1: *[Target Production Name]*

        ## Overview

        Brief summary.

        ## Steps

        - **Step 1:** Description.
        - **Step 2:** Description.
        - *Continue for all steps.*

        # Target 2: *[Target Production Name]*

        ## Overview

        Brief summary.

        ## Steps

        - **Step 1:** Description.
        - **Step 2:** Description.
        - *Continue for all steps.*

        *Repeat this format for each additional target production discussed in the paper.*
    """,
    ):

    message = [
        {"role": "user", "content": [
            *[{"type": "image"} for _ in range(len(images))],
            {"type": "text", "text": instruction}
        ]}
    ]

    return message

def add_response(message, response):
    if len(message) == 1:
       message.append(
           {"role": "assistant", "content": 
            [
                   {"type": "text", "text": response}
            ]})
    return message

def save_responses(json_file,text, num):
    df = pd.read_json(json_file)
    df.at[num-1, 'responses'] = text
    df.to_json(json_file)
    print(df.loc[num-1])

def save_prompt(response_json,dataset_json_file):
    df = pd.read_json(dataset_json_file)
    if response_json in os.listdir():
      os.remove(response_json)
    data = []
    for index, row in df.iterrows():
       if row['responses'] is not None:
          data.append({
               "file_name": row['file_name'],
              "prompt": add_response(row['user_messages'], row['responses'])
          })
    with open(response_json, 'w') as f:
        json.dump(data, f)
    df_r = pd.read_json(response_json)
    print(df_r.iloc[0].prompt)

def get_images(path):
    
    images = []

    for image_file in sorted(os.listdir(path)):
        if image_file.endswith(".png"):
            images.append(Image.open(os.path.join(path, image_file)))

    return images

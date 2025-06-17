import base64
import requests
import os
import json
from tqdm import tqdm
import cv2

import sys
from tool.utils import read_ocr
import random
from openai import OpenAI
from PIL import Image, ImageDraw,ImageFont
from google import genai

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
def request_gemini(images,client):
    prompt = """
You are a professional UI/UX analyst specializing in identifying components in mobile UI screenshots.
Inputs:

Screenshot: A visual representation of the mobile UI.
Screenshot with Mask: A version of the screenshot where all text elements are masked. Each mask has a unique id.

Your goal is to analyze the masked regions by referencing the original screenshot and detect if there are following windows in the UI:
a: Keyboard: A standard virtual keyboard interface.
b: Numberpad: A standard virtual numberpad interface, typically resembling the layout of a calculator or telephone keypad
c: Clockface: the front face of an Analog Clock.
d: Clock other: other element in the clockface windows, incuding the time display
e: Timepicker: digital time picker with scrollable lists.
f: Calendar Cells: the date and weekday cell in the calendar view
g: Calendar others:  other element in the calendar view, like the title, selected date, selected time.

If you detect those windows, please return the masks included in the window:

by the format:{ "id": "<window_type>",...}

You need to reference the original screenshot to verify the content of each masked region.

Only reply the masks with classification of [ "Keyboard","Numberpad","Clockface","Clock other","Timepicker","Calendar cell","Calendar other"],ignore other elements.
Please reply an empty map '{}' for UI with no such elements.
Ensure there is no additional formatting, code blocks or placeholders in your response; return only a clean JSON without any comments.

"""
    image_input= [Image.open(image) for image in images]
    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[image_input, prompt])
    output =   response.text
    print(output)
    try:
        output = json.loads(output)
    except json.JSONDecodeError:
        output = output.replace("```json", "").replace("```", "")
        print(output)
        output = json.loads(output)
    return output

def request_gpt(images,client):
   
    prompt = """
You are a professional UI/UX analyst specializing in identifying components in mobile UI screenshots.
Inputs:

Screenshot: A visual representation of the mobile UI.
Screenshot with Mask: A version of the screenshot where all text elements are masked. Each mask has a unique id.

Your goal is to analyze the masked regions by referencing the original screenshot and detect if there are following windows in the UI:
a: Keyboard: A standard virtual keyboard interface.
b: Numberpad: A standard virtual numberpad interface, typically resembling the layout of a calculator or telephone keypad
c: Clockface: the front face of an Analog Clock.
d: Clock other: other element in the clockface windows, incuding the time display
e: Timepicker: digital time picker with scrollable lists.
f: Calendar Cells: the date and weekday cell in the calendar view
g: Calendar others:  other element in the calendar view, like the title, selected date, selected time.

If you detect those windows, please return the masks included in the window:

by the format:{ "id": "<window_type>",...}

You need to reference the original screenshot to verify the content of each masked region.

Only reply the masks with classification of [ "Keyboard","Numberpad","Clockface","Clock other","Timepicker","Calendar cell","Calendar other"],ignore other elements.
Please reply an empty map '{}' for UI with no such elements.
Ensure there is no additional formatting, code blocks or placeholders in your response; return only a clean JSON without any comments.

"""
    content_array =  [{
        "type": "text",
        'text': f"""
 Screenshot is shown in the first image. 
 Screenshot with Mask in the second image.
"""
    }]
    for image_path in images:

        base64_image = encode_image(image_path)

        dic_img =      {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
                }
            }
        content_array.append(dic_img)  
        
    input_messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content":
                content_array},

        ]
    response = client.chat.completions.create(
                model='gpt-4o',
                messages=input_messages
              
            )
    output = [item.message.content for item in response.choices][0]

    try:
        output = json.loads(output)
    except json.JSONDecodeError:
        output = output.replace("```json", "").replace("```", "")
        print(output)
        output = json.loads(output)
    return output


def detect_text_save(image_path,output_path,ocr_results):
    image = Image.open(image_path).convert("RGBA")  

    draw = ImageDraw.Draw(image, "RGBA")

    outer_border_color = (0, 0, 0, 255)
    fill_color = (255, 255, 255, 255)
    output = []
    for idx, item in enumerate(ocr_results):
        x_min, y_min, x_max, y_max = item["Location"]
        draw.rectangle(((x_min-5, y_min-5), (x_max+5, y_max+5)), fill=fill_color)
        draw.rectangle(((x_min-5, y_min-5), (x_max+5, y_max+5)), outline=outer_border_color, width=10)

        output.append({
            "id": idx+1,
            "text": item["text"],
            "Location": item["Location"]
        })
  

    image = image.convert("RGB")
    image.save(output_path) 

    return output

def reform_ocr(ocr):
    converted_data = [(item['id'], item['text'], item['Location']) for item in ocr]
    return converted_data
def remove_exist(input_path,out_path):
        if os.path.exists(input_path) ==False:
            return
        import shutil
        

if __name__=='__main__':
    #json_path for the dataset
    json_path = '../android_control/control_dic.json'
     #image_path for the dataset
    data_path = '../android_control/images'
    #ocr_detected image path for the dataset, can be generated by generate_ocr.py

    ocr_root_image_id ='../android_control/paddleocr_image_id/'

    output_root= '../android_control/image_layout'
    os.makedirs(output_root,exist_ok=True)

    #ocr_detected ocr json path for the dataset, can be generated by generate_ocr.py

    ocr_root = '../android_control/paddleocr_json/'

 

    with open(json_path, 'r') as f:
      json_data = json.load(f) 
 

    id_list = sorted(list(json_data.keys()))
    random.shuffle(id_list)
    for episode in tqdm(id_list):
        output_root_ep = os.path.join(output_root,episode)
        os.makedirs(output_root_ep,exist_ok=True)
        ocr_path = os.path.join(ocr_root,episode+'.json')

        with open(ocr_path, 'r') as f:
            ocr_data = json.load(f) 
        
    
        step_instructions = json_data[episode]['step_instructions']
        image_root_ep = os.path.join(data_path,episode)
        image_root_ep_id = os.path.join(ocr_root_image_id,episode)
        images = os.listdir(image_root_ep)
        images = sorted(images)
  
        assert len(ocr_data) == len(images),print(len(ocr_data),len(images))
          
        exluded_array = ['Calendar other','Timepicker','Clock other']
        for i,image in enumerate(images):
                output_path = os.path.join(output_root_ep,image)
                if os.path.exists(output_path):
                    continue
                ocr_results = reform_ocr(ocr_data[i])
              
                ui_image = os.path.join(image_root_ep,image)
                id_image = os.path.join(image_root_ep_id,image)
                cv_image = cv2.imread(ui_image)
                if cv_image is not None:
                    height, width, _ = cv_image.shape
           
                client = genai.Client(api_key="")

                results = request_gemini([ui_image,id_image],client)
                if not results: 
                    detect_text_save(ui_image,output_path,ocr_data[i])
                else:
                    remove_ids = [int(key) for key in results.keys() if results[key] not in exluded_array]
                    filtered_data = [item for item in ocr_data[i] if item['id'] not in remove_ids]
                    detect_text_save(ui_image,output_path,filtered_data)



        
        
  
  



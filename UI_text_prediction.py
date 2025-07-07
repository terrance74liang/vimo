import os
import json
import subprocess
import cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw
import base64
import requests
from openai import OpenAI
import random
from io import BytesIO
from google import genai

def calculate_average_color(image, box):
  
    x1, y1, x2, y2 = box
    cropped_image = image.crop((x1, y1, x2, y2))
    np_image = np.array(cropped_image)
    avg_color = np.mean(np_image, axis=(0, 1))  # 按RGB通道计算平均值
    return avg_color

def is_color_in_range(avg_color, lower_bound, upper_bound):
    
    return np.all(avg_color >= lower_bound) and np.all(avg_color <= upper_bound)

def select_text_color(avg_color, lower_bound, upper_bound):
   
    if is_color_in_range(avg_color, lower_bound, upper_bound):
        return 'black' 
    return 'white'  
def merge_overlapping_boxes_withborder(boxes, border):
    
        expanded_boxes = [
            (
                max(0, x - border), 
                max(0, y - border), 
                x + w + border, 
                y + h + border
            )
            for x, y, w, h in boxes
        ]

        merged_boxes = []
        for x_min, y_min, x_max, y_max in expanded_boxes:
            merged = False
            for i, (mx_min, my_min, mx_max, my_max) in enumerate(merged_boxes):
                if not (x_max < mx_min or x_min > mx_max or y_max < my_min or y_min > my_max):
                    merged_boxes[i] = (
                        min(mx_min, x_min), 
                        min(my_min, y_min), 
                        max(mx_max, x_max), 
                        max(my_max, y_max)
                    )
                    merged = True
                    break
            if not merged:
                merged_boxes.append((x_min, y_min, x_max, y_max))

        return [
            (x_min, y_min, x_max - x_min, y_max - y_min)
            for x_min, y_min, x_max, y_max in merged_boxes
        ]
def remove_boxes_and_replace_color(input_image_path, boxes,save_path=None):
  
    image = Image.open(input_image_path).convert("RGB")
    np_image = np.array(image)
    border = 5
    merged_boxes = merge_overlapping_boxes_withborder(boxes, border)

    for box in merged_boxes:
        x, y, w, h = box  
        x_min = max(0, x )
        y_min = max(0, y )
        x_max = min(np_image.shape[1], x + w )
        y_max = min(np_image.shape[0], y + h )

        surrounding_pixels = []

        if y_min > 0:
            surrounding_pixels.extend(np_image[y_min-1, x_min:x_max])

        if y_max < np_image.shape[0]:
            surrounding_pixels.extend(np_image[y_max, x_min:x_max])

        if x_min > 0:
            surrounding_pixels.extend(np_image[y_min:y_max, x_min-1])

        if x_max < np_image.shape[1]:
            surrounding_pixels.extend(np_image[y_min:y_max, x_max])

        surrounding_pixels = np.array(surrounding_pixels)
        mean_color = np.mean(surrounding_pixels, axis=0).astype(int)

        np_image[y_min:y_max, x_min:x_max] = mean_color

    output_image = Image.fromarray(np_image)
    if save_path != None:
        output_image.save(save_path)
    return output_image
def extract_and_merge_boxes(last_image_path, input_mask_image_path, boxes,save_path=None):
  
    last_image = Image.open(last_image_path)
    input_mask_image = Image.open(input_mask_image_path)

    last_image = last_image.convert("RGBA")
    input_mask_image = input_mask_image.convert("RGBA")

    overlay = Image.new("RGBA", input_mask_image.size, (0, 0, 0, 0))

    for box in boxes:
        x_min, y_min, width, height = box
        x_max = x_min + width
        y_max = y_min + height

        cropped_box = last_image.crop((x_min, y_min, x_max, y_max))
        
        overlay.paste(cropped_box, (x_min, y_min))

    combined_image = Image.alpha_composite(input_mask_image, overlay)

    combined_image = combined_image.convert("RGB")
    if save_path !=None:
        combined_image.save(save_path)
    return combined_image
def is_overlap(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    return not (
        x1 + w1 <= x2 or  
        x2 + w2 <= x1 or  
        y1 + h1 <= y2 or  
        y2 + h2 <= y1     
    )
def merge_boxes(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    new_x = min(x1, x2)
    new_y = min(y1, y2)
    new_w = max(x1 + w1, x2 + w2) - new_x
    new_h = max(y1 + h1, y2 + h2) - new_y

    return (new_x, new_y, new_w, new_h)


def merge_overlapping_boxes(boxes):

    merged = []
    while boxes:
        current = boxes.pop(0)
        to_merge = []
        for box in boxes:
            if is_overlap(current, box):
                to_merge.append(box)
        
        for box in to_merge:
            current = merge_boxes(current, box)
            boxes.remove(box)
        
        merged.append(current)
    return merged
def encode_image_pil(image):
  
    buffer = BytesIO()  

    image.save(buffer, format="PNG")  

    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode('utf-8')
def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
def request_exist_gemini(layout_images,user_ins,box_list,client,UI_size):
    prompt = '''
You are a professional UI/UX analyst and your goal is to compare the two UI screenshot and return their overlapping layout.

### Inputs:
1. **Current Screenshot**: The current mobile UI as an image.
2. **Next UI Layout Screenshot**:  
   - An image of the next mobile UI layout with all text replaced by light yellow boxes.
   - Each box has a unique red ID label.
3. Use action: a user action described by lanugage

Next UI Layout Screenshot is a result on an user action on the current screenshot, but the text element are masked.
Please help me to identify those layout that locate in the same position, so I can predict their text directly from the current screenshot.
Usually, the system bar information should be included.

Exclude elements from the result if:
The content (text) changes as a result of the user action, even if the element exists in both screenshots.
You can find a element in the the current screenshot that they share similar context layout, but their absolute are not the same.

Please be very very caution about putting id the list, which means you are very very confident with this task. if you are unsure about some elements, please ignore it and do not put it into the list.

### Output the list of existing element:
Return the result id in the following JSON format:
["1","2",...]

### Notes:
- Ensure the detected elements appear in both UI screenshot, which means their surrounding context are the same.
- Ensure identify those elements that their text will change by the user action and exclude them from your response.
- Ensure identify those elements that share similar context layout, but their absolute are not the same, and them from your response.
- If its an action of "save a page", please return a empty list. 
- Ensure only reply with pure json format, no placeholders or comments.
 '''
    user_prompt =  f"""
        Current UI screenshot is shown in the first image. 
        Next UI screenshot is shown in the second image.
        user action: {user_ins}.
"""

    image_input= [Image.open(image) for image in layout_images]

    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[image_input, prompt+user_prompt])
        
 
    output =   response.text
    print(output)
    try:
        output = json.loads(output)
    except json.JSONDecodeError:
        output = output.replace("```json", "").replace("```", "")
        print(output)
        output = json.loads(output)
    return output
def request_exist(layout_images,user_ins,box_list,client,UI_size,model_str = 'gpt-4o'):
    prompt = '''
You are a professional UI/UX analyst and your goal is to compare the two UI screenshot and return their overlapping layout.

### Inputs:
1. **Current Screenshot**: The current mobile UI as an image.
2. **Next UI Layout Screenshot**:  
   - An image of the next mobile UI layout with all text replaced by light yellow boxes.
   - Each box has a unique red ID label.
3. Use action: a user action described by lanugage

Next UI Layout Screenshot is a result on an user action on the current screenshot, but the text element are masked.
Please help me to identify those layout that locate in the same position, so I can predict their text directly from the current screenshot.
Usually, the system bar information should be included.

Exclude elements from the result if:
The content (text) changes as a result of the user action, even if the element exists in both screenshots.
You can find a element in the the current screenshot that they share similar context layout, but their absolute are not the same.

Please be very very caution about putting id the list, which means you are very very confident with this task. if you are unsure about some elements, please ignore it and do not put it into the list.

### Output the list of existing element:
Return the result id in the following JSON format:
["1","2",...]

### Notes:
- Ensure the detected elements appear in both UI screenshot, which means their surrounding context are the same.
- Ensure identify those elements that their text will change by the user action and exclude them from your response.
- Ensure identify those elements that share similar context layout, but their absolute are not the same, and them from your response.
- If its an action of "save a page", please return a empty list. 
- Ensure only reply with pure json format, no placeholders or comments.
 '''
    content_array =  [{
        "type": "text",
        "text": f"""
        Current UI screenshot is shown in the first image. 
        Next UI screenshot is shown in the second image.
        user action: {user_ins}.
     
"""
    }]
    for base64_image in layout_images:
        

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
                model=model_str,
                messages=input_messages,
                temperature=0.2,
                top_p=0.7
            )
    output = [item.message.content for item in response.choices][0]  
    print(output)
    try:
        output = json.loads(output)
    except json.JSONDecodeError:
        output = output.replace("```json", "").replace("```", "")
        output = json.loads(output)
    return output

def request_semantics_gemini(layout_images,user_ins,box_list,client,UI_size):
    prompt = '''
You are a professional UI/UX analyst assigned to structure and analyze the semantics of mobile UI screenshots. 
Your goal is to segment the UI and annotate box elements in a way that enhances understanding of their roles and relationships within the interface.  

Inputs:  
- Current Screenshot: A visual representation of the mobile UI. 
- Next UI layout screenshot: A visual representation of the next UI layout  with all the text masked with a light yellow box. Each box has a id number on it in red color.
- User Action: A action put on the current UI will result in the next UI. 
- Box locations: a list of box locations to better help you to locate the boxes in the format of {'id': id, 'Location':[x1,y1,width,height]}. Id indicates their id number in the UI screenshot. 
- UI_size: the width/height of the input images. They are in the same size. The image you received might be resized. Please rescale it back for the locations. 

Task:  
Structure the boxes in the Next UI  layout screenshot with semantics based on the visual input by following these steps:  
 1,Divide the UI into Semantic Windows Group the UI into functional sections with a specific name (e.g., "Header Windows," "Time Selector Panel"). 
 2. Structure Text Elements in Each Semantic Window.
    - Assign box element to windows based on logical, visual relationships or semantic roles.
    - For every element, structure output as : 
        **id: corresponding box retrieved from the box list and the Next UI layout screenshot.
        **Role: A brief explanation to the role of this box. You should consider their [x1,y1] to indicate their location, [w,h] to indicate their size to decide the role. It is important consider context for the role prediction. For example, if there is a colon ":" in between elements, you should consider these two elements are hour and minute display.  The role should be in detail to distinguish with other items in the same category. 

 Output Format: 
 {     
    "Window Name": {             
    "Category Name": [
            { 
            "id":id,                     
            "Role": "Role"              
            },               
            {                     
            "id":id,                     
            "Role": "Role"         
            },                 
            ...            
            ],             
    "Category Name": [
            {                    
            "id":id,                    
             "Role": "Role"          
             },           
             ...
            ]       
        }, 
     ...     
    }  
 
 Key Guidelines:
  - Ensure to retrive id from the given screenshot and box list. 
  - Avoid duplicating or omitting ids.
  - Every box element in the box location list must be included in the structured output.
  - Ensure there is no additional formatting, code blocks or placeholders in your response; return only a clean JSON without any comments.
 '''
    user_prompt = f"""
        Current UI screenshot is shown in the first image. 
        Next UI screenshot is shown in the second image.
        User Action: {user_ins}.
        Box Locations: {box_list}.
        UI_size:{UI_size}
"""
    
    image_input= [Image.open(image) for image in layout_images]

    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[image_input, prompt+user_prompt])
        
 
    output =   response.text
    print(output)
    try:
        output = json.loads(output)
    except json.JSONDecodeError:
        output = output.replace("```json", "").replace("```", "")
        print(output)
        output = json.loads(output)
    return output

  
def request_semantics(layout_images,user_ins,box_list,client,UI_size,model_str = 'gpt-4o'):
    prompt = '''
You are a professional UI/UX analyst assigned to structure and analyze the semantics of mobile UI screenshots. 
Your goal is to segment the UI and annotate box elements in a way that enhances understanding of their roles and relationships within the interface.  

Inputs:  
- Current Screenshot: A visual representation of the mobile UI. 
- Next UI layout screenshot: A visual representation of the next UI layout  with all the text masked with a light yellow box. Each box has a id number on it in red color.
- User Action: A action put on the current UI will result in the next UI. 
- Box locations: a list of box locations to better help you to locate the boxes in the format of {'id': id, 'Location':[x1,y1,width,height]}. Id indicates their id number in the UI screenshot. 
- UI_size: the width/height of the input images. They are in the same size. The image you received might be resized. Please rescale it back for the locations. 

Task:  
Structure the boxes in the Next UI  layout screenshot with semantics based on the visual input by following these steps:  
 1,Divide the UI into Semantic Windows Group the UI into functional sections with a specific name (e.g., "Header Windows," "Time Selector Panel"). 
 2. Structure Text Elements in Each Semantic Window.
    - Assign box element to windows based on logical, visual relationships or semantic roles.
    - For every element, structure output as : 
        **id: corresponding box retrieved from the box list and the Next UI layout screenshot.
        **Role: A brief explanation to the role of this box. You should consider their [x1,y1] to indicate their location, [w,h] to indicate their size to decide the role. It is important consider context for the role prediction. For example, if there is a colon ":" in between elements, you should consider these two elements are hour and minute display.  The role should be in detail to distinguish with other items in the same category. 

 Output Format: 
 {     
    "Window Name": {             
    "Category Name": [
            { 
            "id":id,                     
            "Role": "Role"              
            },               
            {                     
            "id":id,                     
            "Role": "Role"         
            },                 
            ...            
            ],             
    "Category Name": [
            {                    
            "id":id,                    
             "Role": "Role"          
             },           
             ...
            ]       
        }, 
     ...     
    }  
 
 Key Guidelines:
  - Ensure to retrive id from the given screenshot and box list. 
  - Avoid duplicating or omitting ids.
  - Every box element in the box location list must be included in the structured output.
  - Ensure there is no additional formatting, code blocks or placeholders in your response; return only a clean JSON without any comments.
 '''
    content_array =  [{
        "type": "text",
        'text': f"""
        Current UI screenshot is shown in the first image. 
        Next UI screenshot is shown in the second image.
        User Action: {user_ins}.
        Box Locations: {box_list}.
        UI_size:{UI_size}
"""
    }]
    for base64_image in layout_images:

        # base64_image = encode_image(image_path)

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
                model=model_str,
                messages=input_messages
                # temperature=0.2,
                # top_p=0.7
            )
    output = [item.message.content for item in response.choices][0]  

    try:
        output = json.loads(output)
    except json.JSONDecodeError:
        output = output.replace("```json", "").replace("```", "")
        print(output)
        output = json.loads(output)
    return output


def request_api_gemini(images,step_instruction,layout_text,client,model_str = 'gpt-4o'):

    prompt = """

Task: Plan the content for the next UI screen based on the provided inputs and instructions.

Inputs:

Current Screenshot: A visual representation of the mobile UI.
Next UI layout screenshot: A visual representation of the next UI  layout with all light yellow box indicates a text place. Each box has a id number on it.
User Instruction: A specific action or command that transitions the current UI to the next UI state.
Semantics for the masks in Next UI screenshot: A structured map.

Goal:
Predict the content (text) for each masked area in the next UI layout screenshot based on the following steps:

Map Affected Elements to the Next UI:
Align the affected elements with the yellow box coordinates on the next UI.
Predict the text for each yellow box based on the user instruction and the context of the current UI.

If you can not find any information about the text, predict predict a plausible text based on its context.
Ensure to use the semnatics to help you understand the layout and predict the text. If you think the semantic is wrong, please modify it in your output.

Output:
Return the predictions in JSON format with the structure:
{
    "Window Name": {
        "Category Name": [
                {
                    "id":id,
                    "text": "text",
                    "role":"role"
                },
               {
                    "id":id,
                    "text": "text",
                    "role":"role"
                    }
            ],
        },
        ...
}
Ensure predict text based on the context.
Do not include any special characters.
Ensure there is no additional formatting, code blocks or placeholders in your response; return only a clean JSON without any comments.
 """
    user_prompt = f"""
        Current UI screenshot is shown in the first image. 
        Next UI screenshot is shown in the second image.
        User Action: {step_instruction}.
        Yellow box semantics: {layout_text}.
"""
    
   
    image_input= [Image.open(image) for image in images]

    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[image_input, prompt+user_prompt])
        
 
    output =   response.text
    print(output)
    try:
        output = json.loads(output)
    except json.JSONDecodeError:
        output = output.replace("```json", "").replace("```", "")
        print(output)
        output = json.loads(output)
    return output
    

def request_api(images,step_instruction,layout_text,client,model_str = 'gpt-4o'):

    prompt = """

Task: Plan the content for the next UI screen based on the provided inputs and instructions.

Inputs:

Current Screenshot: A visual representation of the mobile UI.
Next UI layout screenshot: A visual representation of the next UI  layout with all light yellow box indicates a text place. Each box has a id number on it.
User Instruction: A specific action or command that transitions the current UI to the next UI state.
Semantics for the masks in Next UI screenshot: A structured map.

Goal:
Predict the content (text) for each masked area in the next UI layout screenshot based on the following steps:

Map Affected Elements to the Next UI:
Align the affected elements with the yellow box coordinates on the next UI.
Predict the text for each yellow box based on the user instruction and the context of the current UI.

If you can not find any information about the text, predict predict a plausible text based on its context.
Ensure to use the semnatics to help you understand the layout and predict the text. If you think the semantic is wrong, please modify it in your output.

Output:
Return the predictions in JSON format with the structure:
{
    "Window Name": {
        "Category Name": [
                {
                    "id":id,
                    "text": "text",
                    "role":"role"
                },
               {
                    "id":id,
                    "text": "text",
                    "role":"role"
                    }
            ],
        },
        ...
}
Ensure predict text based on the context.
Do not include any special characters.
Ensure there is no additional formatting, code blocks or placeholders in your response; return only a clean JSON without any comments.
 """
    content_array =  [{
        "type": "text",
        'text': f"""
        Current UI screenshot is shown in the first image. 
        Next UI screenshot is shown in the second image.
        User Action: {step_instruction}.
        Yellow box semantics: {layout_text}.
"""
    }]
    for base64_image in images:

        # base64_image = encode_image(image_path)

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
                model=model_str,
                messages=input_messages,
                temperature=0.2,
                top_p=0.7
            )
    output = [item.message.content for item in response.choices][0]  
    print(output)
    try:
        output = json.loads(output)
    except json.JSONDecodeError:
        output = output.replace("```json", "").replace("```", "")
        output = json.loads(output)
    
    return output

def gpt_to_image(gpt,image_path,save_path=None):
    image = Image.open(image_path).convert("RGB")
    lower_yellow_white = np.array([20, 20, 120]) 
    upper_yellow_white = np.array([255, 255, 255]) 
    width, height = image.size

    for data in gpt:

        x,y,w,h = data['Location']

        x0 = max(0, x - 7)
        y0 = max(0, y - 7)
        x1 = min(width, x + w + 14)
        y1 = min(height, y + h + 14)
        box =[x0,y0,x1,y1]
        avg_color = calculate_average_color(image, box)
        text_color = select_text_color(avg_color, lower_yellow_white, upper_yellow_white)
        if data['text'] =="":
            continue
        image = add_text_to_box(image, [x,y,x+w,y+h],data['text'],text_color)
    if save_path != None:
        image.save(save_path)
    return image
def add_text_to_box(image, location, text,color):
   
    draw = ImageDraw.Draw(image)

    x_min, y_min, x_max, y_max = location
    width = x_max-x_min
    height = y_max-y_min
    font_size = 2  
    font_path = "./Arial.ttf" 
    font = ImageFont.truetype(font_path, font_size)

    while True:
        bbox = font.getbbox(text)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if text_width >= width or text_height >= height:
            break
        font_size += 1
        font = ImageFont.truetype(font_path, font_size)

    font_size -= 1
    font = ImageFont.truetype(font_path, font_size)

    bbox = font.getbbox(text)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    ascent, descent = font.getmetrics()

    text_x = x_min + (width - text_width) // 2
    text_y = y_min + (height - text_height) // 2 -descent

    draw.text((text_x, text_y), text, font=font, fill=color)
    return image

def is_vertical_overlap(box1, box2):
        _, y1, _, h1 = box1
        _, y2, _, h2 = box2
        return y1 + h1 > y2 and y2 + h2 > y1

def detect_and_save_steps(input_image, save_image_path=None):


    image = cv2.imread(input_image)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

   
    lower_black = np.array([0, 0, 0])   
    upper_black = np.array([50, 50, 50])
    mask = cv2.inRange(image, lower_black, upper_black)
    
 
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 

    valid_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10: 
            area = cv2.contourArea(contour)
            rect_area = w * h
            if area / rect_area > 0.8:  
                roi = image[y:y+h, x:x+w]
    
                lower_yellow_white = np.array([200, 200, 200]) 
                upper_yellow_white = np.array([255, 255, 255]) 
                
                yellow_white_mask = cv2.inRange(roi, lower_yellow_white, upper_yellow_white)
                non_zero_pixels = cv2.countNonZero(yellow_white_mask)
                total_pixels = roi.shape[0] * roi.shape[1]
                
                if non_zero_pixels / total_pixels > 0.5: 
                    valid_boxes.append((x, y, w, h))
    valid_boxes = merge_overlapping_boxes(valid_boxes)

    rows = []
    for box in sorted(valid_boxes, key=lambda b: b[1]): 
        added = False
        for row in rows:
            if is_vertical_overlap(row[-1], box):  
                row.append(box)
                added = True
                break
        if not added:
            rows.append([box]) 
    sorted_boxes = []
    for row in rows:
        sorted_boxes.extend(sorted(row, key=lambda b: b[0]))
    box_result_list = []

    for idx, (x, y, w, h) in enumerate(sorted_boxes):
        colors = ["red"]
        chosen_color = random.choice(colors)
      
        pil_image = add_text_to_box(
            pil_image,
            (x, y, x + w, y + h),
            str(idx + 1),chosen_color
        )
        box_result_list.append({'id':idx+1,'Location':[x, y, w, h]})

    if save_image_path != None:
        pil_image.save(save_image_path)
    return box_result_list,pil_image

def detect_boxes(image_path,output_path):
    image = cv2.imread(image_path)

    if image is None:
        print("image is wrong")
        return
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 20, 120]) 
    upper_yellow = np.array([40, 255, 255])  

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    cv2.imwrite("yellow_mask.jpg", yellow_mask)  
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_black = np.array([0, 0, 0]) 
    upper_black = np.array([180, 255, 50])

    output_image = image.copy()
    valid_coordinates = []  

    edges = cv2.Canny(image, 50, 150)

    cv2.imwrite("edges_detected.jpg", edges)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        padding = 1  
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)

        top_edge = hsv[y_start:y_start + 1, x_start:x_end]
        bottom_edge = hsv[y_end - 1:y_end, x_start:x_end]
        left_edge = hsv[y_start:y_end, x_start:x_start + 1]
        right_edge = hsv[y_start:y_end, x_end - 1:x_end]

        is_top_black = np.all(cv2.inRange(top_edge, lower_black, upper_black) > 0)
        is_bottom_black = np.all(cv2.inRange(bottom_edge, lower_black, upper_black) > 0)
        is_left_black = np.all(cv2.inRange(left_edge, lower_black, upper_black) > 0)
        is_right_black = np.all(cv2.inRange(right_edge, lower_black, upper_black) > 0)

        if is_top_black and is_bottom_black and is_left_black and is_right_black:
            valid_coordinates.append((x, y, w, h))
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_image, f"({x},{y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(output_path, output_image)

    return  valid_coordinates
def expand_location(location, expansion=5):
    x, y, w, h = location
    return [x - expansion, y - expansion, w + 2 * expansion, h + 2 * expansion]

def generate_text_onestep(original_image,next_layout,action,result_path,client,model_str='gpt-4o'):

        #detect box and embed with id
        box_layout,pil_image_id = detect_and_save_steps(next_layout,result_path)
        location_dict = {str(loc['id']): loc['Location'] for loc in box_layout}
        location_dict = {
            key: expand_location(value,10) for key, value in location_dict.items()
            }

        with Image.open(original_image) as img:
            width, height = img.size
        base_images = [encode_image(original_image),encode_image_pil(pil_image_id)]
        if model_str =='gemini':
            output_fix = request_exist_gemini([original_image,result_path],action,box_layout,client,[width,height])
        else:
            output_fix = request_exist(base_images,action,box_layout,client,[width,height],model_str =model_str)
        # test
        # output_fix =[]
        extract_and_merge_boxes(original_image, next_layout,[location_dict[idx] for idx in output_fix],result_path)


       
            
        box_layout,pil_image_id = detect_and_save_steps(result_path,result_path)
        
        location_dict = {str(loc['id']): loc['Location'] for loc in box_layout}
        location_dict = {
            key: expand_location(value) for key, value in location_dict.items()
            }
        base_images = [encode_image(original_image),encode_image_pil(pil_image_id)]
        if len(box_layout) == 0:
            return
        if model_str =='gemini':

            next_semantics= request_semantics_gemini([original_image,result_path],action,box_layout,client,[width,height])
        else:
            next_semantics= request_semantics(base_images,action,box_layout,client,[width,height],model_str)

        if model_str =='gemini':

            gpt = request_api_gemini([original_image,result_path],action,next_semantics,client)
        else:
            gpt = request_api(base_images,action,next_semantics,client,model_str)

         

        extracted_data = []
        add_id = set()
        for window_name, window in gpt.items():
            if isinstance(window,list):
                stack = window

            else:
                stack = list(window.values())
            while stack:
                category = stack.pop()

                if isinstance(category, list):
                    for item in category:
                        try:
                            id_str = str(item.get('id', '')) 

                            if id_str not in location_dict:
                                continue
                            if id_str in add_id:
                                continue
                            add_id.add(id_str)
                            result = location_dict[id_str]

                          
                            extracted_data.append({'Location': result, 'text': item['text']})

                        except Exception as e:
                            print(f"Error processing item: {item}, Exception: {e}")

                elif isinstance(category, dict):
                    stack.extend(category.values())
                else:
                    print(f"Unhandled category type: {category}")
        remove_boxes_and_replace_color(result_path, [location_dict[re] for re in location_dict.keys()],result_path)

        gpt_to_image(extracted_data,result_path,result_path)
        # image.save(result_path)
        
          

if __name__=="__main__":
    client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
    original_image = './ep_0_sg_0.png'
    next_layout = './0_1.png'
    action = "Open the Zoho Meet app"
    output_path = 'result_zoho.png'
    generate_text_onestep(original_image,next_layout,action,output_path,client,model_str='gemini')
        
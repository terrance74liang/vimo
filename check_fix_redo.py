import base64
import os
import json
from tqdm import tqdm
from PIL import Image, ImageDraw
from google import genai
import time
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
    # print(output)
    try:
        output = json.loads(output)
    except json.JSONDecodeError:
        output = output.replace("```json", "").replace("```", "")
        # print(output)
        output = json.loads(output)
    return output

def detect_text_save(image_path,output_path,ocr_results):
    image = Image.open(image_path).convert("RGBA")  

    draw = ImageDraw.Draw(image, "RGBA")

    outer_border_color = (0, 0, 0, 255)
    fill_color = (255, 255, 255, 255)
    output = []
    for idx, item in enumerate(ocr_results):
        x_min, y_min, x_max, y_max = item["location"]
        draw.rectangle(((x_min-5, y_min-5), (x_max+5, y_max+5)), fill=fill_color)
        draw.rectangle(((x_min-5, y_min-5), (x_max+5, y_max+5)), outline=outer_border_color, width=10)

        output.append({
            "id": idx+1,
            "text": item["text"],
            "Location": item["location"]
        })
  

    image = image.convert("RGB")
    image.save(output_path) 

    return output

def reform_ocr(ocr):
    converted_data = [(item['id'], item['text'], item['Location']) for item in ocr]
    return converted_data

if __name__=='__main__':
    path = '/media/data/te_liang/android_control_tfrecords/structured'

    for episode in tqdm(sorted(os.listdir(path))):

        ocr_path = os.path.join(path, episode)
        episode_subset = episode.split('_')

        for subset in range(2):
            with open(os.path.join(path, episode, episode_subset[1] + f'_{subset}' + '.json'), 'r') as f:
                ocr_data = json.load(f) 
            
            images = os.path.join(path, episode, episode_subset[1] + f'_{subset}' + '.png')

            exluded_array = ['Calendar other','Timepicker','Clock other']

            client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

            results = request_gemini([images,os.path.join(path, episode,  f'og_{episode_subset[1]}_{subset}' + '.png')],client)

            time.sleep(7)
            
            if not results: 
                detect_text_save(os.path.join(path, episode,  f'og_{episode_subset[1]}_{subset}' + '.png'),images,ocr_data)
            else:
                remove_ids = [int(key) for key in results.keys() if results[key] not in exluded_array]
                filtered_data = [item for item in ocr_data if item['id'] not in remove_ids]
                detect_text_save(os.path.join(path, episode,  f'og_{episode_subset[1]}_{subset}' + '.png'),images,filtered_data)




        
        
  
  



import json
import os
import sys
sys.path.append(os.path.abspath('../tool'))

from utils import remove_text_save,remove_text_save_withid,detect_text_save
import random
from tqdm import tqdm
from PIL import Image

def detect_save():
    image_root = '../android_control/images'
    # save images where each text is removed
    output_root_image = '../android_control/paddleocr_image/'
    # save images where each text is removed and filled with an id toekn
    output_root_image_id = '../android_control/paddleocr_image_id/'

    #save ocr information
    output_root_json = '../android_control/paddleocr_json/'

    # dataset json 
    json_path = '../android_control/control_dic.json'
    os.makedirs(output_root_json,exist_ok=True)
    id_list = set()

    with open(json_path, 'r') as f:
        json_data = json.load(f) 
        id_list =set(json_data.keys())

    id_list = list(id_list)
    random.shuffle(id_list)
    for ep_id in tqdm(id_list):
        ep_path = os.path.join(image_root,ep_id)
        out_ep_path_image = os.path.join(output_root_image,ep_id)
        out_ep_path_image_id = os.path.join(output_root_image_id,ep_id)
        output_path_json = os.path.join(output_root_json,ep_id+'.json')
        if os.path.exists(output_path_json)==True:
             continue
        print(ep_id)
        os.makedirs(out_ep_path_image,exist_ok=True)
        os.makedirs(out_ep_path_image_id,exist_ok=True)
        images = os.listdir(ep_path)
        images = sorted(images)
        ocr_ep = []
        for img in images:
            image_path = os.path.join(ep_path,img)
            output_path_image = os.path.join(out_ep_path_image,img)
            output_path_image_id = os.path.join(out_ep_path_image_id,img)
            print(output_path_image_id)
            ocr_json = detect_text_save(image_path,output_path_image,min_width=30,min_height=30,output_path_id = output_path_image_id)

            ocr_ep.append(ocr_json)
        
        with open(output_path_json, "w", encoding="utf-8") as f:
            json.dump(ocr_ep, f, ensure_ascii=False)  # ensure_ascii=False 保留非 ASCII 字符

if __name__=='__main__':
    detect_save()

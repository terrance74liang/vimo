import tensorflow as tf
import os
import json
from paddleocr import TextDetection, TextRecognition
from PIL import Image, ImageDraw
from pathlib import Path
import shutil
import tempfile
from tqdm import tqdm
import argparse

# def str_representation(input_path, output_path, json_output, min_width, min_height, mask_matrix, text_matrix):
#     json_ocr = []
#     current_id = 0

#     pic = Image.open(input_path).convert("RGBA")
#     draw = ImageDraw.Draw(pic)

#     for row in range(mask_matrix.shape[0]):
#         x0 = mask_matrix[row][1]
#         x1 = mask_matrix[row][1] + mask_matrix[row][3]
#         y0 = mask_matrix[row][0]
#         y1 = mask_matrix[row][0] + mask_matrix[row][2]

#         if (x1 - x0) <= min_width or (y1 - y0) <= min_height:
#             print(mask_matrix[row][2])
#             print(mask_matrix[row][3])

#             continue

#         with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
#             current_id += 1
#             cropped = pic.crop((x0, y0, x1, y1))
#             cropped.save(tmp, format="PNG")
#             tmp.flush()
#             print(text_matrix[row])
#             json_ocr.append({
#                 'id': current_id,
#                 'text': text_matrix[row],
#                 'location': list(map(int, [x0, y0, x1, y1]))
#             })

#         draw.rectangle(((x0 - 5, y0 - 5), (x1 + 5, y1 + 5)), fill=(255, 255, 255, 255))
#         draw.rectangle(((x0 - 5, y0 - 5), (x1 + 5, y1 + 5)), outline=(0, 0, 0, 255), width=10)

#         bbox = draw.textbbox((0, 0), str(current_id))
#         text_width = bbox[2] - bbox[0]
#         text_height = bbox[3] - bbox[1]
#         text_x = x0 + ((x1 - x0) - text_width) / 2
#         text_y = y0 + ((y1 - y0) - text_height) / 2
#         draw.text(xy=(text_x, text_y), text=str(current_id), fill=(255, 0, 0, 255))

#     pic.convert("RGB").save(fp=output_path)

#     with open(json_output, 'w', encoding='utf-8') as f:
#         json.dump(json_ocr, f, ensure_ascii=False, indent=2)

def str_representation(input_path, output_path, json_output, min_width, min_height, detection_model, recognition_model):
    model_de = detection_model
    model_re = recognition_model
    results_de = model_de.predict(input_path, batch_size=1)
    json_ocr = []
    current_id = 0

    for res in results_de:
        pic = Image.open(input_path).convert("RGBA")
        draw = ImageDraw.Draw(pic)

        for coordinate in res.get('dt_polys'):
            x0 = min(coordinate[0][0], coordinate[2][0])
            x1 = max(coordinate[0][0], coordinate[2][0])
            y0 = min(coordinate[0][1], coordinate[2][1])
            y1 = max(coordinate[0][1], coordinate[2][1])

            if (x1 - x0) <= min_width or (y1 - y0) <= min_height:
                continue

            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                current_id += 1
                cropped = pic.crop((x0, y0, x1, y1))
                cropped.save(tmp, format="PNG")
                tmp.flush()
                results_re = model_re.predict(tmp.name, batch_size=1)
                json_ocr.append({
                    'id': current_id,
                    'text': results_re[0]['rec_text'],
                    'location': list(map(int, [x0, y0, x1, y1]))
                })

            draw.rectangle(((x0 - 5, y0 - 5), (x1 + 5, y1 + 5)), fill=(255, 255, 255, 255))
            draw.rectangle(((x0 - 5, y0 - 5), (x1 + 5, y1 + 5)), outline=(0, 0, 0, 255), width=10)

            bbox = draw.textbbox((0, 0), str(current_id))
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = x0 + ((x1 - x0) - text_width) / 2
            text_y = y0 + ((y1 - y0) - text_height) / 2
            draw.text(xy=(text_x, text_y), text=str(current_id), fill=(255, 0, 0, 255))

        pic.convert("RGB").save(fp=output_path)

        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(json_ocr, f, ensure_ascii=False, indent=2)

def _parse_fn(example_proto):
    feature_description = {
        'image/encoded': tf.io.VarLenFeature(tf.string),
        'episode_id' : tf.io.FixedLenFeature([1], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/channels': tf.io.FixedLenFeature([], tf.int64),
        'image/ui_annotations_positions': tf.io.VarLenFeature(tf.float32),
        'image/ui_annotations_text' : tf.io.VarLenFeature(tf.string),
        'step_id': tf.io.FixedLenFeature([1],tf.int64)
    }
    return tf.io.parse_single_example(example_proto, feature_description)



def filter_wrap(allowed_ids):
    def _filter_fn(example):
        return tf.reduce_any(tf.equal(example['episode_id'][0], allowed_ids))
    return _filter_fn


# input is a directory, output is also a directory, episode path is a file and split is the folder you want to name it for eitheer put training or validation or testing
def run(input_path = '../android_control',output_path = '../vimo_processed_data',episode_path = 'data/control_train.json', split = None, mask_path = 'data/aitw_dic.json'):

    data_path = input_path
    subfiles = os.listdir(data_path)
    output_dir = str(Path(output_path).joinpath(split))
    detection = TextDetection(model_name="PP-OCRv5_server_det")
    recognition = TextRecognition(model_name="PP-OCRv5_server_rec")

    seeds = []

    with open(Path(episode_path), 'r', encoding='utf-8') as f:
        ep = json.load(f)   
        episodes = tf.constant([item for sublist in ep.values() for item in sublist], dtype=tf.string)
        # episodes = tf.constant(['16257529840554122890'], dtype=tf.string)

    with open(mask_path,'r') as f:
        aitw_dic = json.load(f)
        
    for file in tqdm(subfiles):
        # if 'general-00234-of-00321' not in file:
        #     continue
        gzip_dataset = tf.data.TFRecordDataset(
            [os.path.join(data_path, file)],
            compression_type='GZIP'
        ).map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE).filter(filter_wrap(episodes))

        for record in gzip_dataset:
            p = Path(output_dir)
            screenshots = tf.sparse.to_dense(record['image/encoded']).numpy()[0]
            parsed_episode_id = record['episode_id'].numpy()[0].decode('utf-8')
            ui_masks = tf.reshape(tf.sparse.to_dense(record['image/ui_annotations_positions']), shape = (-1,4)).numpy()
            ui_text = [s.decode('utf-8') for s in tf.sparse.to_dense(record['image/ui_annotations_text']).numpy()]
            steps = record['step_id'].numpy().item()
            height = record['image/height'].numpy()
            width = record['image/width'].numpy()
            channels = record['image/channels'].numpy()

            image = tf.io.decode_raw(screenshots, out_type=tf.uint8)
            image = tf.reshape(image, (height, width, channels))

            encoded_png = tf.io.encode_png(image)

            if int(aitw_dic[parsed_episode_id]['step_ids'][-1]) != steps:

                ep_folder_name = p.joinpath(str(parsed_episode_id) + '_' + str(steps))
                seeds.append([str(parsed_episode_id) + '_' + str(steps),[str(steps)]])

                if ep_folder_name.exists():
                    shutil.rmtree(ep_folder_name)
                
                ep_folder_name.mkdir(parents=True, exist_ok=True)

                with open(ep_folder_name.joinpath('prompt.json'),'w') as f:
                    json.dump({'edit': aitw_dic[parsed_episode_id]['step_instructions'][steps]},fp = f)

            if steps != 0 and steps != int(aitw_dic[parsed_episode_id]['step_ids'][-1]):
                tf.io.write_file(str(p.joinpath(str(parsed_episode_id) + '_' + str(steps)).joinpath(f'og_{steps}_0.png')), encoded_png)
                tf.io.write_file(str(p.joinpath(str(parsed_episode_id) + '_' + str(steps -1)).joinpath(f'og_{steps - 1}_1.png')), encoded_png)
                str_representation(str(p.joinpath(str(parsed_episode_id) + '_' + str(steps)).joinpath(f'og_{steps}_0.png')),
                                   str(p.joinpath(str(parsed_episode_id) + '_' + str(steps)).joinpath(f'p1_{steps}_0.png')),
                                   str(p.joinpath(str(parsed_episode_id) + '_' + str(steps)).joinpath(f'p1_{steps}_0.json')),30,30, detection, recognition)
                str_representation(str(p.joinpath(str(parsed_episode_id) + '_' + str(steps - 1)).joinpath(f'og_{steps - 1}_1.png')),
                                   str(p.joinpath(str(parsed_episode_id) + '_' + str(steps - 1)).joinpath(f'p1_{steps - 1}_1.png')),
                                   str(p.joinpath(str(parsed_episode_id) + '_' + str(steps - 1)).joinpath(f'p1_{steps - 1}_1.json')),30,30, detection, recognition)

            elif steps == 0:
                tf.io.write_file(str(p.joinpath(str(parsed_episode_id) + '_' + str(steps)).joinpath(f'og_{steps}_0.png')), encoded_png)
                str_representation(str(p.joinpath(str(parsed_episode_id) + '_' + str(steps)).joinpath(f'og_{steps}_0.png')),
                                   str(p.joinpath(str(parsed_episode_id) + '_' + str(steps)).joinpath(f'p1_{steps}_0.png')),
                                   str(p.joinpath(str(parsed_episode_id) + '_' + str(steps)).joinpath(f'p1_{steps}_0.json')),30,30, detection, recognition)
            elif steps == int(aitw_dic[parsed_episode_id]['step_ids'][-1]):
                tf.io.write_file(str(p.joinpath(str(parsed_episode_id) + '_' + str(steps - 1)).joinpath(f'og_{steps - 1}_1.png')), encoded_png)
                str_representation(str(p.joinpath(str(parsed_episode_id) + '_' + str(steps - 1)).joinpath(f'og_{steps - 1}_1.png')),
                                   str(p.joinpath(str(parsed_episode_id) + '_' + str(steps - 1)).joinpath(f'p1_{steps - 1}_1.png')),
                                   str(p.joinpath(str(parsed_episode_id) + '_' + str(steps - 1)).joinpath(f'p1_{steps - 1}_1.json')),30,30, detection, recognition)
    
    with open(Path(output_dir).joinpath('seeds_2.json'), 'w', encoding='utf-8') as f:
        json.dump(seeds, f, ensure_ascii=False, indent=2)   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", required=False,
        help="Path to input file or directory"
    )
    parser.add_argument(
        "-o", "--output_path", required=False,
        help="Path to save output"
    )

    parser.add_argument(
        "-e", "--episode_path", required=False,
        help="Path to episodes"
    )
    parser.add_argument(
        "-s", "--split", required=True,
        help="path to episodes"
    )
    parser.add_argument(
        "-m", "--mask_path", required=True,
        help="path to STR boxes"
    )

    args = parser.parse_args()

    run(**args.__dict__)



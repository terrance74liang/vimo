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

def str_representation(input_path, output_path,json_output, min_width, min_height, detection_model, recognition_model):
    model_de = detection_model
    model_re = recognition_model
    results_de = model_de.predict(input_path, batch_size =1)
    json_ocr = []
    current_id = 0

    for res in results_de:

        pic = Image.open(input_path).convert("RGBA")

        for coordinate in res.get('dt_polys'):

            if abs(coordinate[0][0] - coordinate[2][0]) <= min_width or abs(coordinate[0][1] - coordinate[2][1]) <= min_height:
                continue

            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                current_id += 1
                cropped = pic.crop((coordinate[3][0],coordinate[1][1],coordinate[1][0],coordinate[3][1]))
                cropped.save(tmp, format="PNG")
                tmp.flush()
                results_re = model_re.predict(tmp.name, batch_size = 1)
                json_ocr.append({'id': current_id , 'text': results_re[0]['rec_text'], 'location': list(map(int,[coordinate[0][0],coordinate[0][1],coordinate[2][0],coordinate[2][1]]))})

            draw = ImageDraw.Draw(pic)
            draw.rectangle(((coordinate[0][0] -5, coordinate[0][1]-5), (coordinate[2][0]+5, coordinate[2][1]+5)), fill=(255, 255, 255, 255))
            draw.rectangle(((coordinate[0][0] -5, coordinate[0][1]-5), (coordinate[2][0]+5, coordinate[2][1]+5)), outline = (0, 0, 0, 255),width = 10)

            bbox = draw.textbbox((0, 0), str(current_id))
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            box_width = coordinate[2][0] - coordinate[0][0]
            box_height = coordinate[2][1] - coordinate[0][1]
            text_x = coordinate[0][0] + (box_width - text_width) / 2
            text_y = coordinate[0][1] + (box_height - text_height) / 2

            draw.text(xy=(text_x,text_y), text=str(current_id), fill = (255, 0, 0, 255), font_size=20)
        
        pic = pic.convert("RGB")

        pic.save(fp=output_path)

        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(json_ocr, f, ensure_ascii=False, indent=2)

def _parse_fn(example_proto):
    feature_description = {
        'screenshots': tf.io.VarLenFeature(tf.string),
        'episode_id' : tf.io.FixedLenFeature([1], tf.int64),
        'step_instructions': tf.io.VarLenFeature(tf.string)
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def filter_wrap(allowed_ids):
    def _filter_fn(example):
        return tf.reduce_any(tf.equal(example['episode_id'][0], allowed_ids))
    return _filter_fn


# input is a directory, output is also a directory, episode path is a file and split is the folder you want to name it for eitheer put training or validation or testing
def run(input_path = '../android_control_tfrecords/data',output_path = '../android_control_tfrecords/structured2',episode_path = None, split = None):

    data_path = input_path
    subfiles = os.listdir(data_path)
    output_dir = str(Path(output_path).joinpath(split))

    detection = TextDetection(model_name="PP-OCRv5_server_det")
    recognition = TextRecognition(model_name="PP-OCRv5_server_rec")

    seeds = []

    with open(Path(episode_path), 'r', encoding='utf-8') as f:
        ep = json.load( f)   
        episodes = tf.constant([int(item) for sublist in ep.values() for item in sublist] + [0,20,60], dtype=tf.int64)
    
    for file in tqdm(subfiles):
        if 'android' not in file:
            continue

        gzip_dataset = tf.data.TFRecordDataset([data_path + '/' +file],compression_type = 'GZIP').map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE).filter(filter_wrap(episodes))

        for record in iter(gzip_dataset):

            screenshots = tf.sparse.to_dense(record['screenshots'])

            parsed_episode_id = record['episode_id'].numpy()[0]

            step_instructions = tf.sparse.to_dense(record['step_instructions'])

            for i in range(len(screenshots) - 1):

                p = Path(output_dir)
                ep_folder_name = p.joinpath(str(parsed_episode_id) + '_' + str(i))
                seeds.append([str(parsed_episode_id) + '_' + str(i),[str(i)]])

                if ep_folder_name.exists():
                    shutil.rmtree(ep_folder_name)
                ep_folder_name.mkdir(parents=True, exist_ok=True)

                with open(ep_folder_name.joinpath('prompt.json'),'w') as f:
                    json.dump({'edit': step_instructions[i].numpy().decode('utf-8')},fp = f)

                sample1_path_og = str(ep_folder_name.joinpath(f'og_{i}_0'))
                sample2_path_og = str(ep_folder_name.joinpath(f'og_{i}_1'))
                sample1_path = str(ep_folder_name.joinpath(f'p1_{i}_0'))
                sample2_path = str(ep_folder_name.joinpath(f'p1_{i}_1'))

                png = '.png'
                json_txt = '.json'

                if i == 0:
                    tf.io.write_file(sample1_path_og + png, screenshots[i])
                    tf.io.write_file(sample2_path_og + png, screenshots[i + 1])

                    str_representation(sample1_path_og + png,sample1_path + png,sample1_path + json_txt,30,30, detection, recognition)
                    str_representation(sample2_path_og + png,sample2_path + png,sample2_path + json_txt,30,30, detection, recognition)
                else:
                    tf.io.write_file(sample2_path_og + png, screenshots[i + 1])
                    str_representation(sample2_path_og + png,sample2_path + png,sample2_path + json_txt,30,30, detection, recognition)

                    shutil.copy(p.joinpath(str(parsed_episode_id) + '_' + str(i -1)).joinpath(f'og_{i-1}_1' + png), p.joinpath(str(parsed_episode_id) + '_' + str(i)).joinpath(f'og_{i}_0' + png))
                    shutil.copy(p.joinpath(str(parsed_episode_id) + '_' + str(i -1)).joinpath(f'p1_{i-1}_1' + png), p.joinpath(str(parsed_episode_id) + '_' + str(i)).joinpath(f'p1_{i}_0' + png))
                    shutil.copy(p.joinpath(str(parsed_episode_id) + '_' + str(i -1)).joinpath(f'p1_{i-1}_1' + json_txt), p.joinpath(str(parsed_episode_id) + '_' + str(i)).joinpath(f'p1_{i}_0' + json_txt))


    with open(Path(output_dir).joinpath('seeds.json'), 'w', encoding='utf-8') as f:
        json.dump(seeds, f, ensure_ascii=False, indent=2)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", required=True,
        help="Path to input file or directory"
    )
    parser.add_argument(
        "-o", "--output_path", required=True,
        help="Path to save output"
    )

    parser.add_argument(
        "-e", "--episode_path", required=True,
        help="Path to episodes"
    )
    parser.add_argument(
        "-s", "--split", required=True,
        help="path to episodes"
    )

    args = parser.parse_args()

    run(**args.__dict__)



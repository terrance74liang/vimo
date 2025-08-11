import argparse
from pathlib import Path
import shutil
import os
import json 
from tqdm import tqdm

def merge_directories(target_dir: Path, source_dirs: list):
    seeds = []
    if any([not x.exists() for x in source_dirs]):
        print('path not found')
        return None
    else:
        for directory in tqdm(source_dirs):
            for sub_dir in os.listdir(directory):
                if 'seeds' in sub_dir:
                    with open(directory.joinpath(sub_dir),'r') as f:
                        seeds += json.load(f)
                else:
                    shutil.copytree(directory.joinpath(sub_dir), target_dir.joinpath(sub_dir), dirs_exist_ok=True)
        
    with open(target_dir.joinpath('seeds.json'),'w') as f:
        json.dump(seeds,f, indent=2)
    
    print('dataset done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple directories into a target directory."
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        required=True,
        help="Target directory to merge into."
    )
    parser.add_argument(
        "--source_dirs",
        type=Path,
        nargs='+',  
        required=True,
        help="Source directories to merge from."
    )

    args = parser.parse_args()

    merge_directories(**args.__dict__)

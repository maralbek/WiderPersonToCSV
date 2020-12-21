import cv2
import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm
import argparse

#annot_path = '/home/maralbek/Desktop/Codes/work/video_detections/WiderPerson/Annotations'
#img_path = '/home/maralbek/Desktop/Codes/work/video_detections/WiderPerson/Images'

parser = argparse.ArgumentParser(description='convert wideperson dataset annotations to CSV files for TF2')
parser.add_argument('-a', '--annot_path', type=str, required=True, help='Annotations Path')
parser.add_argument('-i', '--img_path', type=str, required=True, help='Path to images')
parser.add_argument('-t', '--text_path', type=str, required=True, help='Path to txt file with file names (test, train, val)')
parser.add_argument('-c', '--csv_path', type=str, required=True, help='Path to output csv file folder')

args = parser.parse_args()

info = []
with open(args.text_path) as f:
    for item in tqdm(f.readlines(), unit=' files'):
        file_name = item.rstrip('\n') + '.jpg'
        with (open(args.annot_path + '/' + file_name + '.txt')) as anno_f:
            next(anno_f)
            img = cv2.imread(args.img_path + '/' + file_name)
            h, w, c = img.shape
            for line in anno_f.readlines():
                elements = line.split(" ")
                class_names = ["empty", "pedestrians", "riders", "partially-visible persons", "ignore regions", "crowd"]
                img_info = (
                    file_name,
                    int(w),
                    int(h),
                    class_names[int(elements[0])],
                    int(elements[1]),
                    int(elements[2]),
                    int(elements[3]),
                    int(elements[4])
                )
                info.append(img_info)
            column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
            xml_df = pd.DataFrame(info, columns=column_name)

xml_df.to_csv(args.csv_path + 'labels.csv', index=None)




import numpy as np
import mxnet as mx
from mxnet.gluon.data.vision import transforms
from insightface.utils.face_align import norm_crop
from insightface import model_zoo
from pathlib import Path
import shutil
import os
from pathlib import Path
from tqdm import tqdm
import cv2
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from dataset import FamiliesDataset, PairDataset, ImgDataset
from random import shuffle,sample
from typing import List, Tuple, Callable
import mytypes as t
import pickle

"""
insightface==0.1.5
"""

def ensure_path(cur: Path) -> Path:
    if not cur.exists():
        os.makedirs(str(cur))
    return cur

def prepare_images(root_dir: Path, output_dir: Path) -> None:
    whitelist_dir = 'MID'
    detector = model_zoo.get_model('retinaface_r50_v1')
    detector.prepare(ctx_id=-1, nms=0.4)
    for family_path in tqdm(root_dir.iterdir()):
        for person_path in family_path.iterdir():
            if not person_path.is_dir() or not person_path.name.startswith(whitelist_dir):
                continue
            output_person = ensure_path(output_dir / person_path.relative_to(root_dir))
            for img_path in person_path.iterdir():
                img = cv2.imread(str(img_path))
                bbox, landmarks = detector.detect(img, threshold=0.5, scale=1.0)
                output_path = output_person / img_path.name
                if len(landmarks) < 1:
                    warped_img = cv2.resize(img, (112, 112))
                else:
                    warped_img = norm_crop(img, landmarks[0])
                cv2.imwrite(str(output_path), warped_img)


def prepare_test_images(root_dir: Path, output_dir: Path):
    detector = model_zoo.get_model('retinaface_r50_v1')
    detector.prepare(ctx_id=-1, nms=0.4)
    output_dir = ensure_path(output_dir)
    for img_path in root_dir.iterdir():
        img = cv2.imread(str(img_path))
        bbox, landmarks = detector.detect(img, threshold=0.5, scale=1.0)
        output_path = output_dir / img_path.name
        if len(landmarks) < 1:
            print(f'smth wrong with {img_path}')
            warped_img = cv2.resize(img, (112, 112))
        else:
            warped_img = norm_crop(img, landmarks[0])
        cv2.imwrite(str(output_path), warped_img)

if __name__ == '__main__':
    prepare_images(Path("/home/zxm/桌面/kinship/triplets/val-faces"),
                   Path("/home/zxm/桌面/kinship/triplets/Validation/val-faces"))


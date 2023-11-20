# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""
This script is used as a first step to build combined annotations for pre-training
data_path : path to original GQA annotations to be downloaded from https://cs.stanford.edu/people/dorarad/gqa/download.html
img_path : path to original GQA images to be downloaded from https://cs.stanford.edu/people/dorarad/gqa/download.html
sg_path : path to original GQA scene graphs to be downloaded from https://cs.stanford.edu/people/dorarad/gqa/download.html
vg_img_data_path : path to image info for VG images to be downloaded from https://visualgenome.org/static/data/dataset/image_data.json.zip
"""
import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import List
import sys
PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from tqdm import tqdm
from utils.dump import Annotation, Datapoint
from utils.spans import consolidate_spans


def parse_args():
    parser = argparse.ArgumentParser("Conversion script")

    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Path to the gqa dataset",
    )
    parser.add_argument(
        "--img_path",
        required=True,
        type=str,
        help="Path to the gqa image dataset",
    )
    parser.add_argument(
        "--sg_path",
        required=True,
        type=str,
        help="Path to the gqa dataset scene graph",
    )

    parser.add_argument(
        "--vg_img_data_path",
        required=True,
        type=str,
        help="Path to image meta data for VG"
    )

    parser.add_argument(
        "--out_path",
        default="",
        type=str,
        help="Path where to export the resulting dataset. ",
    )
    return parser.parse_args()


def convert(split, data_path, img_path, sg_path, output_path, imid2data):

    with open(data_path / f"{split}_balanced_questions.json", "r") as f:
        data = json.load(f)
    with open(sg_path / f"{split}_sceneGraphs.json", "r") as f:
        sg_data = json.load(f)

    img2ann = defaultdict(dict)
    for k, v in data.items():
        img2ann[v["imageId"]][k] = v

    # Add missing annotations by inspecting the semantic field
    regexp = re.compile(r"([0-9]+)")
    regexp2 = re.compile(r"([A-z]+)")
    count = 0

    for k, v in img2ann.items():
        for ann_id, annotations in v.items():
            expected_boxes = []
            for item in annotations["semantic"]:
                if item["operation"] == "select":
                    if len(regexp.findall(item["argument"])) > 0:
                        expected_boxes.append(
                            (regexp2.findall(item["argument"])[0].strip(), regexp.findall(item["argument"])[0])
                        )
            question_boxes = list(annotations["annotations"]["question"].values())

            for name, box_id in expected_boxes:
                if box_id not in question_boxes:
                    count += 1
                    beg = annotations["question"].find(name)
                    end = beg + len(name)
                    annotations["annotations"]["question"][(beg, end)] = box_id

    # Add annotations for the questions where there is a box for the answer but not for the question (what/where/who questions)
    for k, v in img2ann.items():
        for ann_id, ann in v.items():
            question_objects = list(ann["annotations"]["question"].values())
            answer_objects = list(ann["annotations"]["answer"].values())
            if len(set(answer_objects) - set(question_objects)) > 0:

                for box_id in answer_objects:
                    if box_id not in question_objects:

                        if ann["question"].find("What") > -1:
                            beg = ann["question"].find("What")
                            end = beg + len("What")
                        elif ann["question"].find("what") > -1:
                            beg = ann["question"].find("what")
                            end = beg + len("what")
                        elif ann["question"].find("Who") > -1:
                            beg = ann["question"].find("Who")
                            end = beg + len("Who")
                        elif ann["question"].find("who") > -1:
                            beg = ann["question"].find("who")
                            end = beg + len("who")
                        elif ann["question"].find("Where") > -1:
                            beg = ann["question"].find("Where")
                            end = beg + len("Where")
                        elif ann["question"].find("where") > -1:
                            beg = ann["question"].find("where")
                            end = beg + len("where")
                        else:
                            continue

                        ann["annotations"]["question"][(beg, end)] = box_id

    all_datapoints: List[Datapoint] = []
    d_name = "gqa"

    for k, v in tqdm(img2ann.items()):
        for ann_id, annotation in v.items():
            question = annotation["question"]
            cur_datapoint = Datapoint(
                image_id=k,
                dataset_name="gqa",
                original_id=ann_id,
                caption=question,
                annotations=[],
                tokens_negative=[(0, len(question))],
            )

            if len(annotation["annotations"]["question"]) > 0:

                for text_tok_id, box_anno_id in annotation["annotations"]["question"].items():
                    target_bbox = sg_data[k]["objects"][box_anno_id]
                    x, y, h, w = target_bbox["x"], target_bbox["y"], target_bbox["h"], target_bbox["w"]
                    target_bbox = [x, y, w, h]
                    converted_bbox = [
                        target_bbox[0],
                        target_bbox[1],
                        target_bbox[2] + target_bbox[0],
                        target_bbox[3] + target_bbox[1],
                    ]

                    if isinstance(text_tok_id, str):
                        if ":" in text_tok_id:
                            text_tok_id = text_tok_id.split(":")
                        if isinstance(text_tok_id, list) and len(text_tok_id) > 1:
                            beg = sum([len(x) for x in question.split()[: int(text_tok_id[0])]]) + int(text_tok_id[0])
                            end = (
                                sum([len(x) for x in question.split()[: int(text_tok_id[1]) - 1]])
                                + int(text_tok_id[1])
                                - 1
                            )
                            end = end + len(question.split()[int(text_tok_id[1]) - 1])
                        else:
                            beg = sum([len(x) for x in question.split()[: int(text_tok_id)]]) + int(text_tok_id)
                            end = beg + len(question.split()[int(text_tok_id)])
                    else:
                        beg, end = text_tok_id

                    cleaned_span = consolidate_spans([(beg, end)], question)

                    cur_ann = Annotation(
                        area=h * w,
                        iscrowd=0,
                        category_id=1,
                        bbox=target_bbox,
                        giou_friendly_bbox=converted_bbox,
                        tokens_positive=cleaned_span,
                    )
                    cur_datapoint.annotations.append(cur_ann)
            all_datapoints.append(cur_datapoint)

    with open(output_path / "gqa_dict.pkl", "wb") as f:
        pickle.dump(all_datapoints, f)


def main(args):
    data_path = Path(args.data_path)
    sg_path = Path(args.sg_path)
    output_path = Path(args.out_path) if args.out_path is not None else data_path
    with open(f"{args.vg_img_data_path}/image_data.json", "r") as f:
        image_data = json.load(f)
    imid2data = {x["image_id"]: x for x in image_data}

    os.makedirs(str(output_path), exist_ok=True)

    convert("train", data_path, args.img_path, sg_path, output_path, imid2data)


if __name__ == "__main__":
    main(parse_args())

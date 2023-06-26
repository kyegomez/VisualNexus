from ultralytics import YOLO
from FastSAM.utils.tools import fast_process, convert_box_xywh_to_xyxy, format_results, box_prompt, point_prompt, text_prompt
import argparse
import ast
import torch
import cv2
import numpy as np

class ImageSegmentation:

    def __init__(self, model_path='./weights/FastSAM.pt', img_path='./images/dogs.jpg', imgsz=1024, iou=0.9, text_prompt=None, conf=0.4,
                output='./output/', randomcolor=True, point_prompt="[[0,0]]", point_label="[0]", box_prompt="[0,0,0,0]", better_quality=False,
                device=None, retina=True, withContours=False):
        
        self.model_path = model_path
        self.img_path = img_path
        self.imgsz = imgsz
        self.iou = iou
        self.text_prompt = text_prompt
        self.conf = conf
        self.output = output
        self.randomcolor = randomcolor
        self.point_prompt = ast.literal_eval(point_prompt)
        self.point_label = ast.literal_eval(point_label)
        self.box_prompt = ast.literal_eval(box_prompt)
        self.better_quality = better_quality
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.retina = retina
        self.withContours = withContours

    def segment(self):
        # load model
        model = YOLO(self.model_path)
        results = model(
            self.img_path,
            imgsz=self.imgsz,
            device=self.device,
            retina_masks=self.retina,
            iou=self.iou,
            conf=self.conf,
            max_det=100,
        )
        if self.box_prompt[2] != 0 and self.box_prompt[3] != 0:
            annotations = self.prompt(results, box=True)
            annotations = np.array([annotations])
            fast_process(
                annotations=annotations,
                args=self,
                mask_random_color=self.randomcolor,
                bbox=convert_box_xywh_to_xyxy(self.box_prompt),
            )

        elif self.text_prompt is not None:
            results = format_results(results[0], 0)
            annotations = self.prompt(results, text=True)
            annotations = np.array([annotations])
            fast_process(
                annotations=annotations, args=self, mask_random_color=self.randomcolor
            )

        elif self.point_prompt[0] != [0, 0]:
            results = format_results(results[0], 0)
            annotations = self.prompt(results, point=True)
            # list to numpy
            annotations = np.array([annotations])
            fast_process(
                annotations=annotations,
                args=self,
                mask_random_color=self.randomcolor,
                points=self.point_prompt,
            )

        else:
            fast_process(
                annotations=results[0].masks.data,
                args=self,
                mask_random_color=self.randomcolor,
            )


    def prompt(self, results, box=None, point=None, text=None):
        ori_img = cv2.imread(self.img_path)
        ori_h = ori_img.shape[0]
        ori_w = ori_img.shape[1]
        if box:
            mask, idx = box_prompt(
                results[0].masks.data,
                convert_box_xywh_to_xyxy(self.box_prompt),
                ori_h,
                ori_w,
            )
        elif point:
            mask, idx = point_prompt(
                results, self.point_prompt, self.point_label, ori_h, ori_w
            )
        elif text:
            mask, idx = text_prompt(results, self.text_prompt, self.img_path,self.device)
        else:
            return None
        return mask


if __name__ == "__main__":
    img_seg = ImageSegmentation()
    img_seg.segment()

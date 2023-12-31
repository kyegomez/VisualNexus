import os
from pathlib import Path
from ultralytics import YOLO
from FastSAM.utils.tools import fast_process, convert_box_xywh_to_xyxy, format_results, box_prompt, point_prompt, text_prompt
import ast
import torch
import cv2
import numpy as np

from datasets import load_dataset
from pathlib import Path
import os

from datasets import Dataset
import pandas as pd


def load_hf_dataset(dataset_name):
    #load a dataset from hf dataset library
    #return s alist of file paths of the images in the dataset
    dataset = load_dataset(dataset_name)
    file_paths = []

    for example in dataset['train']:
        file_path = example['image']
        file_paths.append(file_path)

        return file_paths
    


class SAG_IMG:
    """
    SAG_IMG: Segment Anything for Image.

    This class handles the process of iterating over a dataset of images, segmenting them, and outputting a structured
    dataset ready for pre-training a model.

    ...

    Attributes
    ----------
    dataset_folder : str
        Path to the folder containing images for segmentation.
    model_path : str
        Path to the model to be used for segmentation (default is './weights/FastSAM.pt')
    imgsz : int
        Size of the image (default is 1024)
    iou : float
        IoU threshold for filtering the annotations (default is 0.9)
    text_prompt : str
        Text prompt to be used in the segmentation process (default is None)
    conf : float
        Object confidence threshold (default is 0.4)
    output : str
        Path to save the output (default is './output/')
    randomcolor : bool
        Indicates if mask random color should be used (default is True)
    point_prompt : str
        Point prompt for the segmentation process (default is '[[0,0]]')
    point_label : str
        Point label for the segmentation process (default is '[0]')
    box_prompt : str
        Box prompt for the segmentation process (default is '[0,0,0,0]')
    better_quality : bool
        Indicates if better quality using morphologyEx should be used (default is False)
    device : str
        Device to be used for processing (default is None, which leads to automatic selection of 'cuda' or 'cpu')
    retina : bool
        Indicates if high-resolution segmentation masks should be drawn (default is True)
    withContours : bool
        Indicates if the edges of the masks should be drawn (default is False)

    Methods
    -------
    segment():
        Processes all images in the dataset folder and saves the segmentation results to the output directory.
    prompt(results, box=None, text=None):
        Prompts the segmentation process based on the provided results and prompt type.
    fast_process(annotations, mask_random_color):
        Performs the fast process function from FastSAM.utils.tools on the given annotations.
    """

    def __init__(self, image_file_paths, model_path='./weights/FastSAM.pt', imgsz=1024, iou=0.9, text_prompt=None, conf=0.4,
                 output='./output/', randomcolor=True, point_prompt="[[0,0]]", point_label="[0]", box_prompt="[0,0,0,0]", better_quality=False,
                 device=None, retina=True, withContours=False):
        
        self.image_file_paths = image_file_paths
        self.model_path = model_path
        self.imgsz = imgsz
        self.iou = iou
        self.text_prompt = text_prompt
        self.conf = conf
        self.output = Path(output)
        self.randomcolor = randomcolor
        self.point_prompt = ast.literal_eval(point_prompt)
        self.point_label = ast.literal_eval(point_label)
        self.box_prompt = ast.literal_eval(box_prompt)
        self.better_quality = better_quality
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.retina = retina
        self.withContours = withContours

    def segment(self):
        # Load the YOLO model from the specified path
        model = YOLO(self.model_path)

        # Iterate over the list of file paths
        for img_path in self.image_file_paths:
            self.img_path = img_path
            # Process the image with the YOLO model and get the results
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
            elif self.text_prompt is not None:
                results = format_results(results[0], 0)
                annotations = self.prompt(results, text=True)
            elif self.point_prompt[0] != [0, 0]:
                results = format_results(results[0], 0)
                annotations = self.prompt(results, point=True)
            else:
                annotations = results[0].masks.data

            annotations = np.array([annotations])
            self.fast_process(
                annotations=annotations,
                mask_random_color=self.randomcolor,
            )


    def prompt(self, results, box=None, text=None):
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
        elif text:
            mask, idx = text_prompt(results, self.text_prompt, self.img_path, self.device)
        else:
            return None
        return mask
    
    def fast_process(self, annotations, mask_random_color):
        output_path = self.output / Path(self.img_path).name
        fast_process(
            annotations=annotations,
            args=self,
            mask_random_color=mask_random_color,
            output_path=str(output_path),
        )

    def create_dataset(self):
        """
        Create a Huggingface dataset from the segmented images and save it to disk.
        This dataset includes the original image path, segmented image path, and optional text.
        """
        # List to store the dataset examples
        examples = []
        
        # Iterate over the output directory
        for file_name in os.listdir(self.output):
            if file_name.endswith('.jpg'):  # Assuming segmented images are in jpg format
                # Construct the original image path
                image_path = os.path.join(str(Path(self.img_path).parent), file_name)

                # Construct the segmented image path
                segmented_image_path = os.path.join(str(self.output), file_name)

                # Append the example to the list
                examples.append({
                    'image_path': image_path,
                    'segmented_image_path': segmented_image_path,
                    'text_prompt': self.text_prompt if self.text_prompt else None,
                })

        # Convert the list of examples into a pandas DataFrame
        df = pd.DataFrame(examples)

        # Convert the DataFrame into a Huggingface dataset
        dataset = Dataset.from_pandas(df)

        # Save the dataset to disk
        dataset.save_to_disk(os.path.join(str(self.output), 'segmented_dataset'))


if __name__ == "__main__":
    dataset_name="echarlaix/vqa" #dataset name
    image_file_paths = load_hf_dataset(dataset_name)
    img_seg = SAG_IMG(image_file_paths)
    img_seg.segment()
    img_seg.create_dataset()

from datasets import Dataset
import pandas as pd
from models.sag_img import SAG_IMG
from models.sag_video import SAG_VID

import os
from datasets import load_dataset

def load_hf_dataset(dataset_name):
    #custom logic
    pass



class SAG_MEDIA:
    """
    SAG_MEDIA: Segment Anything for Image and Video.

    This class handles the process of iterating over a dataset of images and videos, segmenting them, and outputting a structured
    dataset ready for pre-training a model.

    """

    def __init__(self, image_file_paths, video_file_paths, model_path='./weights/FastSAM.pt', imgsz=1024, iou=0.9, text_prompt=None, conf=0.4,
                 output='./output/', randomcolor=True, point_prompt="[[0,0]]", point_label="[0]", box_prompt="[0,0,0,0]", better_quality=False,
                 device=None, retina=True, withContours=False):

        # Initialize SAG_IMG and SAG_VID with the provided parameters
        self.sag_img = SAG_IMG(
            image_file_paths=image_file_paths,
            model_path=model_path,
            imgsz=imgsz,
            iou=iou,
            text_prompt=text_prompt,
            conf=conf,
            output=output,
            randomcolor=randomcolor,
            point_prompt=point_prompt,
            point_label=point_label,
            box_prompt=box_prompt,
            better_quality=better_quality,
            device=device,
            retina=retina,
            withContours=withContours
        )

        self.sag_vid = SAG_VID(
            video_file_paths=video_file_paths,
            model_path=model_path,
            imgsz=imgsz,
            iou=iou,
            text_prompt=text_prompt,
            conf=conf,
            output=output,
            randomcolor=randomcolor,
            point_prompt=point_prompt,
            point_label=point_label,
            box_prompt=box_prompt,
            better_quality=better_quality,
            device=device,
            retina=retina,
            withContours=withContours
        )

    def segment(self):
        # Perform segmentation for images and videos
        self.sag_img.segment()
        self.sag_vid.segment()

    def create_dataset(self):
        """
        Create a Huggingface dataset from the segmented images and videos and save it to disk.
        This dataset includes the original media file path, segmented media file path, and optional text.
        """

        # Create datasets for images and videos
        self.sag_img.create_dataset()
        self.sag_vid.create_dataset()

        # Merge the datasets
        img_dataset = Dataset.load_from_disk(os.path.join(str(self.sag_img.output), 'segmented_dataset'))
        vid_dataset = Dataset.load_from_disk(os.path.join(str(self.sag_vid.output), 'segmented_dataset'))
        merged_dataset = Dataset.concatenate_datasets([img_dataset, vid_dataset])

        # Save the merged dataset to disk
        merged_dataset.save_to_disk(os.path.join(str(self.sag_img.output), 'merged_segmented_dataset'))

if __name__ == "__main__":
    dataset_name = "echarlaix/vqa"  # dataset name
    image_file_paths = load_hf_dataset(dataset_name)
    video_file_paths = load_hf_dataset(dataset_name)

    media_seg = SAG_MEDIA(image_file_paths, video_file_paths)
    media_seg.segment()
    media_seg.create_dataset()
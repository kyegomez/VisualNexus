from datasets import load_dataset
from metaseq import SegAutoMaskPredictor
import os

from datasets import Dataset
import pandas as pd


class SAG_VID:
    
    def __init__(self, model_type='vit_1', points_per_side=16, points_per_batch=64, min_area=1000, output_dir='./output'):
        """
        Segment anything for video class

        Args:
            model_type(str): type of the model to use for segmentation. Option: 'vit_1', 'vit_h', 'vit_b'
            points_per_side (int): number of points per side
            points_per_batch(int): Number of points per batch
            min_area(int): minimum area
            output_dir: str: directory to save output files
        """

        self.model_type = model_type
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.min_area = min_area
        self.output_dir = output_dir
        self.predictor = SegAutoMaskPredictor()

    def segment_video(self, video_path):
        """
        Segment a video file

        Args:
            video_path (str): path to the video file
        """

        output_path = os.path.join(self.output_dir, os.path.basename(video_path))
        self.predictor.video_predict(
            source=video_path,
            model_type=self.model_type,
            points_per_side=self.points_per_batch,
            min_area = self.min_area,
            output_path = output_path,
        )

    def load_and_segment_datset(self, dataset_name):
        """
        Load a video dataset from hf datasets library and segment it

        Args:
            dataset_name (str): Name of the dataset in hf datasets
        """

        #load the train split of the dataset from huggingface datasets
        dataset = load_dataset(dataset_name, split='train')

        #iterate over the dataset
        for item in dataset:
            #assuming file path is the key for the video file paths
            video_path = item['file_path']

            #segmennt the video
            self.segment_video(video_path)

    def create_dataset(self):
        """
        Create a Huggingface dataset from the segmented videos and save it to disk.
        """
        # List to store the dataset examples
        examples = []
        
        # Iterate over the output directory
        for file_name in os.listdir(self.output_dir):
            if file_name.endswith('.mp4'):  # Assuming segmented videos are in mp4 format
                # Construct the original video path
                video_path = '...'  # Replace this with the logic to construct original video path from segmented video path

                # Construct the segmented video path
                segmented_video_path = os.path.join(self.output_dir, file_name)
                
                # Append the example to the list
                examples.append({
                    'video_path': video_path,
                    'segmented_video_path': segmented_video_path,
                })
        
        # Convert the list of examples into a pandas DataFrame
        df = pd.DataFrame(examples)
        
        # Convert the DataFrame into a Huggingface dataset
        dataset = Dataset.from_pandas(df)
        
        # Save the dataset to disk
        dataset.save_to_disk(os.path.join(self.output_dir, 'segmented_dataset'))

if __name__ == "__main__":
    vid_seg = SAG_VID(output_dir='./output_videos')
    vid_seg.load_and_segment_dataset('dataset_name')
    vid_seg.create_dataset()
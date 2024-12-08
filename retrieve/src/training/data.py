import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
import re

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import (
    Dataset,
    DataLoader,
    SubsetRandomSampler,
    IterableDataset,
    get_worker_info,
)
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    url_opener,
    tar_file_expander,
    valid_sample,
)

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):
    def __init__(
        self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None
    ):
        logging.debug(f"Loading csv data from {input_filename}.")
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class IUXrayDataset(Dataset):  # TODO
    def __init__(
        self,
        img_root,
        json_file,
        transforms,
        tokenizer=None,
        load_include_path=False,
        load_include_k=False,
        retrieval_k=None,
        test=False,
    ):
        """
        Initializes the dataset object.
        :param json_file: path to the JSON file containing the annotations.
        :param transform: optional transform to be applied on a sample.
        """
        logging.debug(f"Loading json data from {json_file}.")
        self.load_include_path = load_include_path
        # Load data from the JSON file
        
        self.retrieve_k = retrieval_k
        self.load_include_k = load_include_k
        
        with open(json_file, "r") as file:
            self.data = json.load(file)
        # if not test:
        #     with open(json_file, "r") as file:
        #         self.data = json.load(file)["train"]
        # else:
        #     with open(json_file, "r") as file:
        #         self.data = json.load(file)["test"]

        # Flatten the list of image paths and associate each with the corresponding report
        self.image_ids = []
        self.image_report_pairs = []
        for entry in self.data:
            self.image_ids.append(entry["id"])
            for img_path in entry["image_path"]:
                if "0.png" in img_path:
                    self.image_report_pairs.append(
                        (os.path.join(img_root, img_path), entry.get('caption', entry.get('report')))
                    )
                else:
                    continue
                # self.image_report_pairs.append((img_path, entry["report"]))

        self.transform = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        """
        Returns the total number of image-report pairs in the dataset.
        """
        return len(self.image_report_pairs)

    def __getitem__(self, idx):
        item=self.data[idx]
        item['image_path']=item['image_path'][0] if isinstance(item['image_path'],list) else item['image_path']
        img_path, report = self.image_report_pairs[idx]
        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformation
        if self.transform:
            image = self.transform(image)
        report_text = self.tokenize([report])[0]
        if self.load_include_path:
            if self.load_include_k and self.retrieve_k:
                return image, report_text, img_path, self.retrieve_k,item
            return image, report_text, img_path
        return image, report_text
class PubMedVisionDataset(Dataset):#TODO
    def __init__(self, img_root,json_file, transforms, tokenizer=None,load_include_path=False,test=False,
                 load_include_k=False,
                    retrieval_k=None,
                 ):
        """
        Initializes the dataset object.
        :param json_file: path to the JSON file containing the annotations.
        :param transform: optional transform to be applied on a sample.
        """
        logging.debug(f'Loading json data from {json_file}.')
        self.load_include_path=load_include_path
        # Load data from the JSON file
        if not test:
            with open(json_file, 'r') as file:
                self.data = json.load(file)
        else:
            with open(json_file, 'r') as file:
                self.data = json.load(file)
        self.load_include_k=load_include_k
        self.retrieval_k=retrieval_k
        # Flatten the list of image paths and associate each with the corresponding report
        self.image_ids = []
        self.image_report_pairs = []
        for entry in self.data:
            self.image_ids.append(entry['id'])
            self.image_report_pairs.append((os.path.join(img_root, entry["image_path"]), entry["report"]))
            # for img_path in entry["image_path"]:
            #     if "0.png" in img_path:
            #         self.image_report_pairs.append((os.path.join(img_root, img_path), entry["report"]))
            #     else:
            #         continue
                # self.image_report_pairs.append((img_path, entry["report"]))
        
        self.transform = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        """
        Returns the total number of image-report pairs in the dataset.
        """
        return len(self.image_report_pairs)

    def __getitem__(self, idx):
        data_info = self.data[idx]
        img_path, report = self.image_report_pairs[idx]
        # Load image
        image = Image.open(img_path).convert('RGB')  
        
        # Apply transformation
        if self.transform:
            image = self.transform(image)
        report_text= self.tokenize([report])[0]
        if self.load_include_path:
            if self.load_include_k==True and self.retrieval_k is not None:
                return image, report_text, img_path, self.retrieval_k, data_info
            return image, report_text, img_path
        return image, report_text
        

class RadiologyDataset(Dataset):  # TODO
    def __init__(
        self,
        img_root,
        json_file,
        transforms,
        tokenizer=None,
        load_include_path=False,
        test=False,
    ):
        """
        Initializes the dataset object.
        :param json_file: path to the JSON file containing the annotations.
        :param transform: optional transform to be applied on a sample.
        """
        logging.debug(f"Loading json data from {json_file}.")
        self.load_include_path = load_include_path
        # Load data from the JSON file
        
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        

        # Flatten the list of image paths and associate each with the corresponding report
        self.image_ids = []
        self.image_report_pairs = []
        for entry in self.data:
            self.image_ids.append(entry["id"])
            # for img_path in entry["image_path"]:
            
            self.image_report_pairs.append(
                    (os.path.join(entry["image_root"], entry["image_path"][0]), entry["report"])
                )
                
                # self.image_report_pairs.append((img_path, entry["report"]))

        self.transform = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        """
        Returns the total number of image-report pairs in the dataset.
        """
        return len(self.image_report_pairs)
    def __getitem__(self, idx):
        img_path, report = self.image_report_pairs[idx]
        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformation
        if self.transform:
            image = self.transform(image)
        report_text = self.tokenize([report])[0]
        if self.load_include_path:
            return image, report_text, img_path
        return image, report_text
class PathologyDataset(Dataset):  # TODO
    def __init__(
        self,
        img_root,
        json_file,
        transforms,
        tokenizer=None,
        load_include_path=False,
        test=False,
    ):
        """
        Initializes the dataset object.
        :param json_file: path to the JSON file containing the annotations.
        :param transform: optional transform to be applied on a sample.
        """
        logging.debug(f"Loading json data from {json_file}.")
        self.load_include_path = load_include_path
        # Load data from the JSON file
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        

        # Flatten the list of image paths and associate each with the corresponding report
        self.image_ids = []
        self.image_report_pairs = []
        for entry in self.data:
            self.image_ids.append(entry["id"])
            # print(entry)
            # for img_path in entry["image_path"]:
            self.image_report_pairs.append(
                    (os.path.join(entry["image_root"], entry.get('image', entry.get('image_path'))), entry.get('caption', entry.get('report')))
                )
                
                # self.image_report_pairs.append((img_path, entry["report"]))

        self.transform = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        """
        Returns the total number of image-report pairs in the dataset.
        """
        return len(self.image_report_pairs)
    def __getitem__(self, idx):
        img_path, report = self.image_report_pairs[idx]
        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformation
        if self.transform:
            image = self.transform(image)
        report_text = self.tokenize([report])[0]
        if self.load_include_path:
            
            return image, report_text, img_path
        return image, report_text
class MimicVQADataset(Dataset):  #
    def __init__(
        self,
        img_root,
        jsonl_file,
        transforms,
        tokenizer=None,
        fixed_K=1,
        
    ):
        """
        Initializes the dataset object.
        :param json_file: path to the JSON file containing the annotations.
        :param transform: optional transform to be applied on a sample.
        """
        logging.debug(f"Loading json data from {jsonl_file}.")
        self.img_root = img_root
        # Load data from the JSON file
        self.fixed_K=fixed_K

        with open(jsonl_file, "r") as file:
            self.data = [json.loads(line) for line in file]
        # Flatten the list of image paths and associate each with the corresponding report
        self.image_ids = []
        # self.image_report_pairs = []
        # for entry in self.data:
        #     self.image_ids.append(entry["id"])
        #     for img_path in entry["image_path"]:
        #         if "0.png" in img_path:
        #             self.image_report_pairs.append(
        #                 (os.path.join(img_root, img_path), entry["report"])
        #             )
        #         else:
        #             continue
                # self.image_report_pairs.append((img_path, entry["report"]))

        self.transform = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        """
        Returns the total number of image-report pairs in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        data_info = self.data[idx]
        img_path = os.path.join(self.img_root, data_info["image"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        retrival_k = self.fixed_K
        report = data_info["report"]
        report_text = self.tokenize([report])[0]

        return image, report_text, img_path, retrival_k, data_info
class IUXrayVQADataset(Dataset):  # TODO
    def __init__(
        self,
        img_root,
        jsonl_file,
        transforms,
        tokenizer=None,
        fixed_K=1,
        
    ):
        """
        Initializes the dataset object.
        :param json_file: path to the JSON file containing the annotations.
        :param transform: optional transform to be applied on a sample.
        """
        logging.debug(f"Loading json data from {jsonl_file}.")
        self.img_root = img_root
        # Load data from the JSON file
        self.fixed_K=fixed_K

        with open(jsonl_file, "r") as file:
            self.data = [json.loads(line) for line in file]
        # Flatten the list of image paths and associate each with the corresponding report
        self.image_ids = []
        # self.image_report_pairs = []
        # for entry in self.data:
        #     self.image_ids.append(entry["id"])
        #     for img_path in entry["image_path"]:
        #         if "0.png" in img_path:
        #             self.image_report_pairs.append(
        #                 (os.path.join(img_root, img_path), entry["report"])
        #             )
        #         else:
        #             continue
                # self.image_report_pairs.append((img_path, entry["report"]))

        self.transform = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        """
        Returns the total number of image-report pairs in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        data_info = self.data[idx]
        img_path = os.path.join(self.img_root, data_info["image"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        retrival_k = self.fixed_K
        report = data_info["report"]
        report_text = self.tokenize([report])[0]

        return image, report_text, img_path, retrival_k, data_info
    
class pmc_oa_VQADataset(Dataset):  # TODO
    def __init__(
        self,
        img_root,
        jsonl_file,
        transforms,
        tokenizer=None,
        test=False,
        fixed_K=1,
        
    ):
        """
        Initializes the dataset object.
        :param json_file: path to the JSON file containing the annotations.
        :param transform: optional transform to be applied on a sample.
        """
        logging.debug(f"Loading json data from {jsonl_file}.")
        self.img_root = img_root
        # Load data from the JSON file
        self.fixed_K=fixed_K

        if not test:
            self.image_folder=os.path.join(img_root,'train_images')
            with open(jsonl_file, "r") as file:
                self.data = [json.loads(line) for line in file]
        else:
            self.image_folder=os.path.join(img_root,'test_images')
            with open(jsonl_file, "r") as file:
                self.data = [json.loads(line) for line in file]
        # Flatten the list of image paths and associate each with the corresponding report
        self.image_ids = []
        # self.image_report_pairs = []
        # for entry in self.data:
        #     # self.image_ids.append(entry["id"])
        #     self.image_report_pairs.append(
        #                 (os.path.join(self.image_folder, entry["image"]), entry["report"])
        #             )

        self.transform = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        """
        Returns the total number of image-report pairs in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        data_info = self.data[idx]
        img_path = os.path.join(self.image_folder, data_info["image"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        retrival_k = self.fixed_K
        report = data_info["report"]
        report_text = self.tokenize([report])[0]

        return image, report_text, img_path, retrival_k, data_info
class HarvardVQADataset(Dataset):  # TODO
    def __init__(
        self,
        img_root,
        jsonl_file,
        transforms,
        tokenizer=None,
        fixed_K=1,
        
    ):
        """
        Initializes the dataset object.
        :param json_file: path to the JSON file containing the annotations.
        :param transform: optional transform to be applied on a sample.
        """
        logging.debug(f"Loading json data from {jsonl_file}.")
        self.img_root = img_root
        # Load data from the JSON file
        self.fixed_K=fixed_K

        with open(jsonl_file, "r") as file:
            self.data = [json.loads(line) for line in file]
        # Flatten the list of image paths and associate each with the corresponding report
        self.image_ids = []
        # self.image_report_pairs = []
        # for entry in self.data:
        #     self.image_ids.append(entry["id"])
        #     for img_path in entry["image_path"]:
        #         if "0.png" in img_path:
        #             self.image_report_pairs.append(
        #                 (os.path.join(img_root, img_path), entry["report"])
        #             )
        #         else:
        #             continue
                # self.image_report_pairs.append((img_path, entry["report"]))

        self.transform = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        """
        Returns the total number of image-report pairs in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        data_info = self.data[idx]
        img_path = os.path.join(self.img_root, data_info["image"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        retrival_k = self.fixed_K
        report = data_info["report"]
        report_text = self.tokenize([report])[0]

        return image, report_text, img_path, retrival_k, data_info
class quilt_1m_VQADataset(Dataset):  # TODO
    def __init__(
        self,
        img_root,
        jsonl_file,
        transforms,
        tokenizer=None,
        fixed_K=1,
        
    ):
        """
        Initializes the dataset object.
        :param json_file: path to the JSON file containing the annotations.
        :param transform: optional transform to be applied on a sample.
        """
        logging.debug(f"Loading json data from {jsonl_file}.")
        self.img_root = img_root
        # Load data from the JSON file
        self.fixed_K=fixed_K

        with open(jsonl_file, "r") as file:
            self.data = [json.loads(line) for line in file]
        # Flatten the list of image paths and associate each with the corresponding report
        self.image_ids = []
        # self.image_report_pairs = []
        # for entry in self.data:
        #     self.image_ids.append(entry["id"])
        #     for img_path in entry["image_path"]:
        #         if "0.png" in img_path:
        #             self.image_report_pairs.append(
        #                 (os.path.join(img_root, img_path), entry["report"])
        #             )
        #         else:
        #             continue
                # self.image_report_pairs.append((img_path, entry["report"]))

        self.transform = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        """
        Returns the total number of image-report pairs in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        data_info = self.data[idx]
        img_path = os.path.join(self.img_root, data_info["image"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        retrival_k = self.fixed_K
        report = data_info["report"]
        report_text = self.tokenize([report])[0]

        return image, report_text, img_path, retrival_k, data_info

class quilt_1m_Dataset(Dataset):  # TODO
    def __init__(
        self,
        img_root,
        jsonl_file,
        transforms,
        tokenizer=None,
        load_include_path=False,
        load_include_k=False,
        retrieval_k=None,
        test=False,
    ):
        """
        Initializes the dataset object.
        :param json_file: path to the JSON file containing the annotations.
        :param transform: optional transform to be applied on a sample.
        """
        self.retrieve_k = retrieval_k
        self.load_include_k = load_include_k
        self.load_include_path = load_include_path
        # Load data from the JSON file
        self.image_folder=img_root
        with open(jsonl_file, "r") as file:
            try:
                self.data = json.load(file)
            except:
                self.data = [json.loads(line) for line in file]
        
        # Flatten the list of image paths and associate each with the corresponding report
        self.image_ids = []
        self.image_report_pairs = []
        for entry in self.data:
            self.image_ids.append(entry["id"])
            img_name = entry.get('image_path', entry.get('image'))
            report = entry.get('caption', entry.get('report'))
            self.image_report_pairs.append([os.path.join(self.image_folder, img_name), report])
        
        self.transform = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        """
        Returns the total number of image-report pairs in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data[idx])
        item= self.data[idx]
        item_id=item['id']
        img_name = item.get('image_path', item.get('image'))
        report = item.get('caption', item.get('report'))
        
        # (id,img_name, report) = self.data[idx]
        # Load image
        image = Image.open(os.path.join(self.image_folder,img_name)).convert("RGB")

        # Apply transformation
        if self.transform:
            image = self.transform(image)
        report_text = self.tokenize([report])[0]
        if self.load_include_path:
            if self.load_include_k and self.retrieve_k:
                return image, report_text, img_name, self.retrieve_k,item
            return image, report_text, img_name
        return image, report_text
class pmc_oa_Dataset(Dataset):  # TODO
    def __init__(
        self,
        img_root,
        jsonl_file,
        transforms,
        tokenizer=None,
        load_include_path=False,
        load_include_k=False,
        retrieval_k=None,
        test=False,
    ):
        """
        Initializes the dataset object.
        :param json_file: path to the JSON file containing the annotations.
        :param transform: optional transform to be applied on a sample.
        """
        self.retrieve_k = retrieval_k
        self.load_include_k = load_include_k
        self.load_include_path = load_include_path
        # Load data from the JSON file
        
        if not test:
            self.image_folder=os.path.join(img_root,'train_images')
            with open(jsonl_file, "r") as file:
                self.data = json.load(file)
        else:
            self.image_folder=os.path.join(img_root,'test_images')
            with open(jsonl_file, "r") as file:
                self.data = json.load(file)
        
        # Flatten the list of image paths and associate each with the corresponding report
        self.image_ids = []
        self.image_report_pairs = []
        for entry in self.data:
            # self.image_ids.append(entry["id"])
            self.image_report_pairs.append(
                        (os.path.join(self.image_folder, entry["image"]), entry["report"])
                    )
        # for entry in self.data:
        #     self.image_ids.append(entry["id"])
        self.transform = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        """
        Returns the total number of image-report pairs in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data[idx])
        item= self.data[idx]
        item_id=item['id']
        img_name=item['image']
        report=item['report']
        
        # (id,img_name, report) = self.data[idx]
        # Load image
        image = Image.open(os.path.join(self.image_folder,img_name)).convert("RGB")

        # Apply transformation
        if self.transform:
            image = self.transform(image)
        report_text = self.tokenize([report])[0]
        if self.load_include_path:
            if self.load_include_k and self.retrieve_k:
                return image, report_text, img_name, self.retrieve_k,item
            return image, report_text, img_name
        return image, report_text

class IUXrayVQADataset_with_conf(Dataset):  # TODO
    def __init__(
        self,
        img_root,
        jsonl_file,
        transforms,
        tokenizer=None,
        config_type=None,
        k_max=20,
        k_min=0,
    ):
        """
        Initializes the dataset object.
        :param json_file: path to the JSON file containing the annotations.
        :param transform: optional transform to be applied on a sample.
        """
        logging.debug(f"Loading json data from {jsonl_file}.")
        self.img_root = img_root
        # Load data from the JSON file
        if config_type is None:
            raise ValueError("config_type cannot be None.")

        with open(jsonl_file, "r") as file:
            self.data = [json.loads(line) for line in file]
        
        if config_type == "tokenProb":
            confidence_list = [
            entry["confidence"]
            for entry in self.data
            if entry["confidence"] is not None
        ]
        elif config_type=='verbConf':
            confidence_list = [
            entry["confidence"]
            for entry in self.data
            if entry["confidence"] is not None
            ]
            confidence_list=[int(confidence.strip('%')) for confidence in confidence_list if confidence]
        avg_confidence = sum(confidence_list) / len(confidence_list)
        print(confidence_list)
        print(f'average confidence: {avg_confidence}')
        self.min_confidence = min(confidence_list)
        self.max_confidence = max(confidence_list)
        print(f'min confidence: {self.min_confidence}')
        print(f'max confidence: {self.max_confidence}')
        self.avg_confidence = avg_confidence
        self.confidence_list=confidence_list

        self.k_min = k_min
        self.k_max = k_max

        

        # Flatten the list of image paths and associate each with the corresponding report
        self.image_ids = []
        # self.image_report_pairs = []
        # for entry in self.data:
        #     self.image_ids.append(entry["id"])
        #     for img_path in entry["image_path"]:
        #         if "0.png" in img_path:
        #             self.image_report_pairs.append(
        #                 (os.path.join(img_root, img_path), entry["report"])
        #             )
        #         else:
        #             continue
                # self.image_report_pairs.append((img_path, entry["report"]))

        self.transform = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        """
        Returns the total number of image-report pairs in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        data_info = self.data[idx]
        img_path = os.path.join(self.img_root, data_info["image"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        retrival_k = self.map_confidence_to_k(self.confidence_list[idx])
        report = data_info["report"]
        report_text = self.tokenize([report])[0]

        return image, report_text, img_path, retrival_k, data_info

    def map_confidence_to_k(self, confidence):
        """
        Maps a confidence value to a k value inversely proportional to the confidence.

        Args:
        - min_conf (float): The minimum confidence value.
        - max_conf (float): The maximum confidence value.
        - confidence (float): The current confidence value.
        - k_min (int): The minimum value of k.
        - k_max (int): The maximum value of k.

        Returns:
        - k (int): The number of objects to retrieve.
        """

        if confidence < self.min_confidence or confidence > self.max_confidence:
            raise ValueError("Confidence value out of bounds.")

        # Calculate the inverse proportional value
        normalized_confidence = (confidence - self.min_confidence) / (
            self.max_confidence - self.min_confidence
        )
        k = self.k_max - normalized_confidence * (self.k_max - self.k_min)

        return round(k)


class MimicDataset(Dataset):  # TODO
    def __init__(
        self,
        img_root,
        json_file,
        transforms,
        tokenizer=None,
        load_include_path=False,
        load_include_k=False,
        retrieval_k=None,
        test=False,
    ):
        """
        Initializes the dataset object.
        :param json_file: path to the JSON file containing the annotations.
        :param transform: optional transform to be applied on a sample.
        """
        logging.debug(f"Loading json data from {json_file}.")
        self.load_include_path = load_include_path
        self.retrieve_k = retrieval_k
        self.load_include_k = load_include_k
        # Load data from the JSON file
        # if not test:
        #     with open(json_file, 'r') as file:
        #         self.data = json.load(file)["train"]
        # else:
        #     with open(json_file, 'r') as file:
        #         self.data = json.load(file)["test"]
        with open(json_file, "r") as file:
            self.data = json.load(file)

        # Flatten the list of image paths and associate each with the corresponding report
        self.image_ids = []
        self.image_report_pairs = []
        for entry in self.data:
            self.image_ids.append(entry["id"])
            for img_path in entry["image_path"]:
                self.image_report_pairs.append(
                    (os.path.join(img_root, img_path), entry["report"])
                )

                # self.image_report_pairs.append((img_path, entry["report"]))

        self.transform = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        """
        Returns the total number of image-report pairs in the dataset.
        """
        return len(self.image_report_pairs)

    def __getitem__(self, idx):
        item=self.data[idx]
        img_path, report = self.image_report_pairs[idx]
        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformation
        if self.transform:
            image = self.transform(image)
        report_text = self.tokenize([report])[0]
        if self.load_include_path:
            if self.load_include_k and self.retrieve_k:
                return image, report_text, img_path, self.retrieve_k,item
            return image, report_text, img_path
        return image, report_text


class HarvardDataset(Dataset):  # TODO
    def __init__(
        self,
        img_root,
        json_file,
        transforms,
        tokenizer=None,
        load_include_path=False,
        load_include_k=False,
        retrieval_k=None,
        test=False,
    ):
        """
        Initializes the dataset object.
        :param json_file: path to the JSON file containing the annotations.
        :param transform: optional transform to be applied on a sample.
        """
        logging.debug(f"Loading json data from {json_file}.")
        self.load_include_path = load_include_path
        self.retrieve_k = retrieval_k
        self.load_include_k = load_include_k
        # Load data from the JSON file
        # if not test:
        #     with open(json_file, 'r') as file:
        #         self.data = json.load(file)["train"]
        # else:
        #     with open(json_file, 'r') as file:
        #         self.data = json.load(file)["test"]
        
        with open(json_file, "r") as file:
            self.data = json.load(file)
        # Flatten the list of image paths and associate each with the corresponding report
        self.image_ids = []
        self.image_report_pairs = []

        for entry in self.data:
            filename = entry["filename"]
            # match = re.search(r'data_(\d+)\.npz', filename)

            self.image_ids.append(entry["id"])

            self.image_report_pairs.append(
                (os.path.join(img_root, entry["image_path"]), entry["gpt4_summary"])
            )

            # self.image_report_pairs.append((img_path, entry["report"]))

        self.transform = transforms
        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        """
        Returns the total number of image-report pairs in the dataset.
        """
        return len(self.image_report_pairs)

    def __getitem__(self, idx):
        item=self.data[idx]
        img_path, report = self.image_report_pairs[idx]
        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        report_text = self.tokenize([report])[0]
        if self.load_include_path:
            if self.load_include_k and self.retrieve_k:
                return image, report_text, img_path, self.retrieve_k,item
            return image, report_text, img_path
        return image, report_text


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split("::")
        assert (
            len(weights) == len(urllist)
        ), f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, "r").read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset

        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype("int")
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert (
                len(self.urls) == len(self.weights)
            ), f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(
                    url=self.rng.choices(self.urls, weights=self.weights, k=1)[0]
                )


def get_wds_dataset(
    args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None
):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    "Currently, the number of dataset samples must be specified for the training dataset. "
                    "Please specify it via `--train-num-samples` if no dataset length info is present."
                )
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(
        epoch=epoch
    )  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."

    if resampled:
        pipeline = [
            ResampledShards2(
                input_shards,
                weights=args.train_data_upsampling_factors,
                deterministic=True,
                epoch=shared_epoch,
            )
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend(
                [
                    detshuffle2(
                        bufsize=_SHARD_SHUFFLE_SIZE,
                        initial=_SHARD_SHUFFLE_INITIAL,
                        seed=args.seed,
                        epoch=shared_epoch,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                ]
            )
        pipeline.extend(
            [
                # at this point, we have an iterator over the shards assigned to each worker at each node
                tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                ),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(handler=log_and_continue),
            ]
        )
    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
            wds.to_tuple("image", "text"),
            wds.batched(args.batch_size, partial=not is_train),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert (
                num_shards >= args.workers * args.world_size
            ), "number of shards must be >= total workers"
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(
            num_batches / num_workers
        )  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(
            num_worker_batches
        )  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_pmc_oa_Dataset(
    args,
    preprocess_fn,
    is_train,
    epoch=0,
    tokenizer=None,
    no_shuffle=False,
    load_include_path=False,
    test=False,
):  # TODO
    json_filename = args.train_data if is_train else args.val_data
    assert json_filename
    dataset =pmc_oa_Dataset(
        args.img_root,
        json_filename,
        preprocess_fn,
        tokenizer=tokenizer,
        load_include_path=load_include_path,
        test=test,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    if no_shuffle:
        shuffle = False
        print("Training dataloader do not shuffle.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_quilt_1m_Dataset(
    args,
    preprocess_fn,
    is_train,
    epoch=0,
    tokenizer=None,
    no_shuffle=False,
    load_include_path=False,
    test=False,
):  # TODO
    json_filename = args.train_data if is_train else args.val_data
    assert json_filename
    dataset =quilt_1m_Dataset(
        args.img_root,
        json_filename,
        preprocess_fn,
        tokenizer=tokenizer,
        load_include_path=load_include_path,
        test=test,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    if no_shuffle:
        shuffle = False
        print("Training dataloader do not shuffle.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_HarvardDataset(
    args,
    preprocess_fn,
    is_train,
    epoch=0,
    tokenizer=None,
    no_shuffle=False,
    load_include_path=False,
    test=False,
):  # TODO
    json_filename = args.train_data if is_train else args.val_data
    assert json_filename
    dataset = HarvardDataset(
        args.img_root,
        json_filename,
        preprocess_fn,
        tokenizer=tokenizer,
        load_include_path=load_include_path,
        test=test,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    if no_shuffle:
        shuffle = False
        print("Training dataloader do not shuffle.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_IUXrayDataset(
    args,
    preprocess_fn,
    is_train,
    epoch=0,
    tokenizer=None,
    no_shuffle=False,
    load_include_path=False,
    test=False,
):  # TODO
    json_filename = args.train_data if is_train else args.val_data
    assert json_filename
    dataset = IUXrayDataset(
        args.img_root,
        json_filename,
        preprocess_fn,
        tokenizer=tokenizer,
        load_include_path=load_include_path,
        test=test,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    if no_shuffle:
        shuffle = False
        print("Training dataloader do not shuffle.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_MimicDataset(
    args,
    preprocess_fn,
    is_train,
    epoch=0,
    tokenizer=None,
    no_shuffle=False,
    load_include_path=False,
    test=False,
):  # TODO
    json_filename = args.train_data if is_train else args.val_data
    assert json_filename
    dataset = MimicDataset(
        args.img_root,
        json_filename,
        preprocess_fn,
        tokenizer=tokenizer,
        load_include_path=load_include_path,
        test=test,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    if no_shuffle:
        shuffle = False
        print("Training dataloader do not shuffle.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):
    def __init__(
        self,
        transform=None,
        image_size=(224, 224),
        caption="Dummy caption",
        dataset_size=100,
        tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new("RGB", image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn,
        image_size=image_size,
        dataset_size=args.train_num_samples,
        tokenizer=tokenizer,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_radiology_Dataset(

    args,
    preprocess_fn,
    is_train,
    epoch=0,
    tokenizer=None,
    no_shuffle=False,
    load_include_path=False,
    test=False,
):  # TODO
    json_filename = args.train_data if is_train else args.val_data
    assert json_filename
    dataset = RadiologyDataset(
        args.img_root,
        json_filename,
        preprocess_fn,
        tokenizer=tokenizer,
        load_include_path=load_include_path,
        test=test,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    if no_shuffle:
        shuffle = False
        print("Training dataloader do not shuffle.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)
def get_pathology_Dataset(

    args,
    preprocess_fn,
    is_train,
    epoch=0,
    tokenizer=None,
    no_shuffle=False,
    load_include_path=False,
    test=False,
):  # TODO
    json_filename = args.train_data if is_train else args.val_data
    assert json_filename
    dataset = PathologyDataset(
        args.img_root,
        json_filename,
        preprocess_fn,
        tokenizer=tokenizer,
        load_include_path=load_include_path,
        test=test,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    if no_shuffle:
        shuffle = False
        print("Training dataloader do not shuffle.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "IUXray":
        return get_IUXrayDataset
    elif dataset_type == "Harvard":
        return get_HarvardDataset
    elif dataset_type == "mimic":
        return get_MimicDataset
    elif dataset_type == "pmc_oa":
        return get_pmc_oa_Dataset
    elif dataset_type == "quilt_1m":
        return get_quilt_1m_Dataset
    elif dataset_type=="radiology":
        return get_radiology_Dataset
    elif dataset_type=="pathology":
        return get_pathology_Dataset
    elif dataset_type == "auto":
        ext = data_path.split(".")[-1]
        if ext in ["csv", "tsv"]:
            return get_csv_dataset
        elif ext in ["tar"]:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}."
            )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(
    args,
    preprocess_fns,
    epoch=0,
    tokenizer=None,
    no_shuffle=False,
    load_include_path=False,
):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args,
            preprocess_train,
            is_train=True,
            epoch=epoch,
            tokenizer=tokenizer,
            no_shuffle=no_shuffle,
            load_include_path=load_include_path,
        )

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args,
            preprocess_val,
            is_train=False,
            tokenizer=tokenizer,
            load_include_path=load_include_path,
            test=False,
        )

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data

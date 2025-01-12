import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import pandas


def VOT20Dataset():
    return VOTDatasetClass().get_sequence_list()


class VOTDatasetClass(BaseDataset):
    """VOT2020 dataset

    Publication:
        M. Kristan et al., “The Eighth Visual Object Tracking VOT2020 Challenge Results.” 2020.
        
        http://prints.vicos.si/publications/384

    Download the dataset from https://www.votchallenge.net/vot2020/dataset.html"""
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vot20_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        ext = 'jpg'
        start_frame = 1

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        gt_file = open(str(anno_path), 'r')
        gt_lines = gt_file.readlines()

        gt_list = []

        for i in range(len(gt_lines)):
            bbox = gt_lines[i].strip().split(',')
            if len(bbox) > 0 and bbox[0][0] == "m":
                # strip m from start of line
                bbox[0] = bbox[0][1:]
            if len(bbox) >= 4:
                gt_list.append(bbox[:4])

        ground_truth_rect = np.array(gt_list, dtype=np.float64)

        gt_file.close()

        end_frame = ground_truth_rect.shape[0]

        frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                  sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
                  for frame_num in range(start_frame, end_frame+1)]

        return Sequence(sequence_name, frames, ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        list_path = '{}/list.txt'.format(self.base_path)
        sequence_list = pandas.read_csv(list_path, header=None, squeeze=True).values.tolist()

        return sequence_list

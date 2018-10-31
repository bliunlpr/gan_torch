import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from data.audioparse import KaldiFeatParser


class SequentialDataset(Dataset, KaldiFeatParser):
    def __init__(self, data_scp, label_file, audio_conf):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        utt_id /path/to/audio.wav
        ...
        :param data_scp: Path to scp as describe above
        :param label_file : Dictionary containing the delta_order, context_width, normalize_type and max_num_utt_cmvn
        :param audio_conf: Dictionary containing the sample_rate, num_channel, window_size window_shift
        """
        self.data_scp = data_scp
        self.label_file = label_file
        self.audio_conf = audio_conf
        self.exp_path = self.audio_conf.get("exp_path", './')
        self.normalize_type = self.audio_conf.get("normalize_type", 0)
        self.cmvn = None
        super(SequentialDataset, self).__init__(audio_conf, label_file)

        with open(self.data_scp) as f:
            utt_ids = f.readlines()
        self.utt_ids = [x.strip().split(' ') for x in utt_ids]
        self.utt_size = len(utt_ids)

        if not os.path.isdir(self.exp_path):
            raise Exception(self.exp_path + ' isn.t a path!')
        cmvn_file = os.path.join(self.exp_path, 'cmvn.npy')
        if self.normalize_type == 1:
            self.cmvn = self.loading_cmvn(cmvn_file, self.utt_ids, audio_conf)

    def __getitem__(self, index):

        while True:
            sample = self.utt_ids[index]
            utt_id, audio_path = sample[0], sample[1]
            feature_mat = self.parse_audio(self.audio_conf, audio_path)
            if feature_mat is None:
                index = random.randint(0, self.utt_size - 1)
                continue

            if self.cmvn is not None and self.normalize_type == 1:
                feature_mat = (feature_mat + self.cmvn[0, :]) * self.cmvn[1, :]
            elif self.normalize_type == 2:
                mean = feature_mat.mean()
                std = feature_mat.std()
                feature_mat.add_(-mean)
                feature_mat.div_(std)
            else:
                feature_mat = feature_mat
            feature_mat = torch.FloatTensor(feature_mat)
            
            if self.label_file is not None:
                target_utt_id = utt_id
                target = self.parse_transcript(target_utt_id)
                if target is None:
                    print ('{} has no target {}'.format(utt_id, target_utt_id))
                    index = random.randint(0, self.utt_size - 1)
                    continue
                else:
                    encoded_target = list(filter(None, [int(x) for x in list(target)]))
                    break
            else:
                encoded_target = None
                break

        return utt_id, feature_mat, encoded_target

    def __len__(self):
        return self.utt_size

    def load_cmvn(self):
        return self.cmvn

    def compute_target_count(self):
        encoded_targets = np.concatenate(
            [self.encode(targets)
             for targets in self.target_dict.values()])
        # count the number of each target
        count = np.bincount(encoded_targets, minlength=self.label_num)
        return count


def _seq_collate_fn(batch):
    def func(p):
        return p[1].size(0)

    nseq = len(batch)

    longest_sample = max(batch, key=func)
    max_seqlength = longest_sample[1].size(0)
    data_size = longest_sample[1].size(1)
    seq_data = torch.zeros(nseq, max_seqlength, data_size)

    seq_percentages = torch.FloatTensor(nseq)
    target_sizes = torch.IntTensor(nseq)

    utt_ids = []
    targets = []
    for x in range(nseq):
        sample = batch[x]

        utt_id = sample[0]
        feature_mat = sample[1]
        target = sample[2]

        utt_ids.append(utt_id)

        seq_length = feature_mat.size(0)
        seq_data[x].narrow(0, 0, seq_length).copy_(feature_mat)
        seq_percentages[x] = seq_length / float(max_seqlength)
        
        if target is not None:
            target_sizes[x] = len(target)
            targets.extend(target)
            
    if target is not None:
        targets = torch.IntTensor(targets)
        return utt_ids, seq_data, seq_percentages, targets, target_sizes
    else:
        return utt_ids, seq_data, seq_percentages, None, None


class SequentialDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(SequentialDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _seq_collate_fn


class GanSequentialDataset(Dataset, KaldiFeatParser):
    def __init__(self, data_scp, org_data_scp, label_file, audio_conf):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        utt_id /path/to/audio.wav
        utt_id label(1 3 5)
        ...
        :param data_scp: Path to scp as describe above
        :param org_data_scp: Path to org scp as describe above
        :param label_file: Path to label file as describe above
        :param audio_conf : Dictionary containing the delta_order, context_width, normalize_type and max_num_utt_cmvn
        """
        super(GanSequentialDataset, self).__init__(audio_conf, label_file)

        self.data_scp = data_scp
        self.org_data_scp = org_data_scp
        self.label_file = label_file
        self.exp_path = audio_conf.get("exp_path", './')
        self.pair = audio_conf.get("pair", False)
        self.normalize_type = audio_conf.get("normalize_type", 0)
        self.org_normalize_type = audio_conf.get("org_normalize_type", 0)
        self.cmvn = None
        self.org_cmvn = None

        super(GanSequentialDataset, self).__init__(audio_conf, label_file)

        self.audio_conf = dict(delta_order=audio_conf['delta_order'],
                               context_width=audio_conf['context_width'],
                               num_features=audio_conf['num_features'])

        self.org_audio_conf = dict(delta_order=audio_conf['org_delta_order'],
                                   context_width=audio_conf['org_context_width'],
                                   num_features=audio_conf['num_features'])

        with open(self.data_scp) as f:
            utt_ids = f.readlines()
        self.utt_ids = [x.strip().split(' ') for x in utt_ids]
        self.utt_size = len(utt_ids)
        
        with open(self.org_data_scp) as f:
            org_utt_ids = f.readlines()
        self.org_utt_ids = [x.strip().split(' ') for x in org_utt_ids]
        self.org_utt_size = len(org_utt_ids)
        self.org_utt_ids_dict = {k: v for k, v in (x.strip().split(' ') for x in org_utt_ids)}

        if not os.path.isdir(self.exp_path):
            raise Exception(self.exp_path + ' isn.t a path!')
        cmvn_file = os.path.join(self.exp_path, 'cmvn.npy')
        org_cmvn_file = os.path.join(self.exp_path, 'org_cmvn.npy')
        self.cmvn = self.loading_cmvn(cmvn_file, self.utt_ids, self.audio_conf)
        self.org_cmvn = self.loading_cmvn(org_cmvn_file, self.org_utt_ids, self.org_audio_conf)

    def __getitem__(self, index):
        
        while True:
            sample = self.utt_ids[index]
            utt_id, audio_path = sample[0], sample[1]
            feature_mat = self.parse_audio(self.audio_conf, audio_path)
            if feature_mat is None:
                index = random.randint(0, self.utt_size-1)
                continue
            if self.cmvn is not None and self.normalize_type == 1:
                feature_mat = (feature_mat + self.cmvn[0, :]) * self.cmvn[1, :]
            elif self.normalize_type == 2:
                mean = feature_mat.mean()
                std = feature_mat.std()
                feature_mat.add_(-mean)
                feature_mat.div_(std)
            else:
                feature_mat = feature_mat
            feature_mat = torch.FloatTensor(feature_mat)

            target_utt_id = utt_id                		 		
            target = self.parse_transcript(target_utt_id)
            if target is None:
                print ('{} has no target {}'.format(utt_id, target_utt_id))
                index = random.randint(0, self.utt_size-1)
                continue
            else:
                encoded_target = list([int(x) for x in list(target)])

            if self.pair:
                org_utt_id = target_utt_id
                audio_path = self.org_utt_ids_dict[org_utt_id]
            else:
                if index >= self.org_utt_size:
                    index = random.randint(0, self.org_utt_size - 1)
                sample = self.org_utt_ids[index]
                org_utt_id, audio_path = sample[0], sample[1]

            org_feature_mat = self.parse_audio(self.org_audio_conf, audio_path)
            if org_feature_mat is None:
                print ('{} has no pair {}'.format(utt_id, org_utt_id))
                index = random.randint(0, self.utt_size - 1)
                continue
            if self.org_cmvn is not None and self.org_normalize_type == 1:
                org_feature_mat = (org_feature_mat + self.org_cmvn[0, :]) * self.org_cmvn[1, :]
            elif self.org_normalize_type == 2:
                mean = org_feature_mat.mean()
                std = org_feature_mat.std()
                org_feature_mat.add_(-mean)
                org_feature_mat.div_(std)
            else:
                org_feature_mat = org_feature_mat
            org_feature_mat = torch.FloatTensor(org_feature_mat)
            break
                    
        return utt_id, org_utt_id, feature_mat, org_feature_mat, encoded_target
    
    def __len__(self):
        return self.utt_size
    
    def re_disorder(self):
        random.shuffle(self.org_utt_ids)

    def load_cmvn(self):
        return self.cmvn, self.org_cmvn


def _gan_seq_collate_fn(batch):
    def func(p):
        return p[2].size(0)
        
    def org_func(p):
        return p[3].size(0)
        
    nseq = len(batch)

    longest_sample = max(batch, key=func)
    max_seq_length = longest_sample[2].size(0)
    data_size = longest_sample[2].size(1)
    seq_data = torch.zeros(nseq, max_seq_length, data_size)

    org_longest_sample = max(batch, key=org_func)
    org_max_seq_length = org_longest_sample[3].size(0)
    org_data_size = org_longest_sample[3].size(1)
    org_seq_data = torch.zeros(nseq, org_max_seq_length, org_data_size)
    
    seq_percentages = torch.FloatTensor(nseq)
    org_seq_percentages = torch.FloatTensor(nseq)
    target_sizes = torch.IntTensor(nseq)
    
    utt_ids = []
    org_utt_ids = []
    targets = []
    for x in range(nseq):
        sample = batch[x]
        
        utt_id = sample[0]
        org_utt_id = sample[1]
        feature_mat = sample[2]
        org_feature_mat = sample[3]
        target = sample[4]
        
        utt_ids.append(utt_id)
        org_utt_ids.append(org_utt_id)
        
        seq_length = feature_mat.size(0)
        seq_data[x].narrow(0, 0, seq_length).copy_(feature_mat)
        seq_percentages[x] = seq_length / float(max_seq_length)
        
        org_seq_length = org_feature_mat.size(0)
        org_seq_data[x].narrow(0, 0, org_seq_length).copy_(org_feature_mat)
        org_seq_percentages[x] = org_seq_length / float(org_max_seq_length)
        
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    
    return utt_ids, org_utt_ids, seq_data, seq_percentages, org_seq_data, org_seq_percentages, targets, target_sizes
    

class GanSequentialDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(GanSequentialDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _gan_seq_collate_fn


def _frame_collate_fn(batch):
    def func(p):
        return p[1].size(0)

    nseq = len(batch)

    longest_sample = max(batch, key=func)
    max_length = longest_sample[1].size(0)
    data_size = longest_sample[1].size(1)
    seq_data = torch.zeros(nseq*max_length, data_size)

    num_frames = 0
    targets = []
    for x in range(nseq):
        sample = batch[x]
        utt_id = sample[0]
        feature_mat = sample[1]
        target = sample[2]

        seq_len = feature_mat.size(0)
        seq_data.narrow(0, num_frames, seq_len).copy_(feature_mat)
        num_frames += seq_len

        target_size = len(target)
        targets.extend(target)
        assert seq_len == target_size, \
            '{} feature dim is {} and target {}'.format(utt_id, seq_len, target_size)

    indices = torch.arange(0, num_frames).long()
    input_data = seq_data.index_select(0, indices)
    targets = torch.LongTensor(targets)
    return num_frames, input_data, targets


class FrameDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(FrameDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _frame_collate_fn


def _gan_frame_collate_fn(batch):
    def func(p):
        return p[2].size(0)

    def org_func(p):
        return p[3].size(0)

    minibatch_size = len(batch)

    longest_sample = max(batch, key=func)
    max_length = longest_sample[2].size(0)
    data_size = longest_sample[2].size(1)
    seq_data = torch.zeros(minibatch_size*max_length, data_size)

    org_longest_sample = max(batch, key=org_func)
    org_max_length = org_longest_sample[3].size(0)
    org_data_size = org_longest_sample[3].size(1)
    org_seq_data = torch.zeros(minibatch_size*org_max_length, org_data_size)

    num_frames = 0
    org_num_frames = 0
    targets = []
    utt_ids = []
    for x in range(minibatch_size):
        sample = batch[x]
        utt_id = sample[0]
        feature_mat = sample[2]
        org_feature_mat = sample[3]
        target = sample[4]
        utt_ids.append(utt_id)

        seq_len = feature_mat.size(0)
        seq_data.narrow(0, num_frames, seq_len).copy_(feature_mat)
        num_frames += seq_len
        org_seq_len = org_feature_mat.size(0)
        org_seq_data.narrow(0, org_num_frames, org_seq_len).copy_(org_feature_mat)
        org_num_frames += org_seq_len

        target_size = len(target)
        targets.extend(target)
        assert seq_len == target_size, \
            '{} feature dim is {} and target {}'.format(utt_id, seq_len, target_size)

    indices = torch.arange(0, num_frames).long()
    input_data = seq_data.index_select(0, indices)
    indices = torch.arange(0, org_num_frames).long()
    org_input_data = org_seq_data.index_select(0, indices)

    targets = torch.LongTensor(targets)

    return num_frames, input_data, org_num_frames, org_input_data, targets


class GanFrameDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(GanFrameDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _gan_frame_collate_fn


class FrameBatchGenerator(object):
    def __init__(self, data):
        num_frames, input_data, targets = data
        self.num_read_frame = 0
        self.num_frames = num_frames
        self.frame_rand_idx = np.random.permutation(self.num_frames)
        self.inputs = input_data
        self.targets = targets

    def next_batch(self, batch_size):
        if self.inputs is None or self.targets is None:
            return None, None

        s = self.num_read_frame
        e = s + batch_size

        if s >= self.num_frames or e > self.num_frames:
            return None, None

        self.num_read_frame += e - s
        indices = torch.LongTensor(self.frame_rand_idx[s:e])

        return self.inputs.index_select(0, indices), self.targets.index_select(0, indices)


class GanFrameBatchGenerator(object):
    def __init__(self, data, gan_conf):
        num_frames, input_data, org_num_frames, org_input_data, targets = data
        self.pair = gan_conf.get("pair", False)
        self.num_read_frame = 0
        self.num_frames = num_frames
        self.org_num_frames = org_num_frames
        self.inputs = input_data
        self.org_inputs = org_input_data
        self.targets = targets
        self.frame_rand_idx = np.random.permutation(self.num_frames)
        self.org_frame_rand_idx = np.random.permutation(self.num_frames)
        if not self.pair:
            s = 0
            e = self.org_num_frames
            while True:
                tmp = np.random.permutation(self.org_num_frames)
                if e > self.num_frames:
                    self.org_frame_rand_idx[s:self.num_frames] = tmp[:(self.num_frames - s)]
                    break
                self.org_frame_rand_idx[s:e] = tmp
                s += self.org_num_frames
                e += self.org_num_frames
       
    def next_batch(self, batch_size):
        if self.inputs is None or self.targets is None:
            return None, None, None

        s = self.num_read_frame
        e = s + batch_size

        if s >= self.num_frames or e > self.num_frames:
            return None, None, None

        self.num_read_frame += (e - s)
        indices = torch.LongTensor(self.frame_rand_idx[s:e])
        org_index = torch.LongTensor(self.org_frame_rand_idx[s:e])

        return (self.inputs.index_select(0, indices),
                self.org_inputs.index_select(0, org_index),
                self.targets.index_select(0, indices))

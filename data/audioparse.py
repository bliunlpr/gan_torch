import os
import gzip
import numpy as np
import struct
import progressbar


class Targetcounter(object):
    def __init__(self, target_path, label_num):
        self.target_dict = read_target_file(target_path)
        self.label_num = label_num    
        ##self.target_normalizer = lambda x, y: x
        ##self.alphabet = [str(target) for target in range(self.label_num)]
        ##self.lookup = {character: index for index, character
        ##               in enumerate(self.alphabet)}
    def compute_target_count(self):
        encoded_targets = np.concatenate(
            [self.encode(targets)
             for targets in self.target_dict.values()])

        #  count the number of occurences of each target
        count = np.bincount(encoded_targets,
                        minlength=self.label_num)
        return count
            
    def encode(self, targets):
        """ encode a target sequence """
        ##normalized_targets = self.target_normalizer(targets, self.lookup.keys())
        ##encoded_targets = []

        ##for target in normalized_targets.split(' '):
        ##    encoded_targets.append(self.lookup[target])
        encoded_targets = list([int(x) for x in list(targets)])
        return np.array(encoded_targets, dtype=np.int)
        
                
def load_audio_feat_len(audio_path):
    path, pos = audio_path.split(':')
    ark_read_buffer = open(path, 'rb')
    ark_read_buffer.seek(int(pos), 0)
    header = struct.unpack('<xcccc', ark_read_buffer.read(5))
    if header[0].decode('utf-8') != "B":
        raise Exception("Input .ark file is not binary")
    if header[1].decode('utf-8') == "C":
        raise Exception("Input .ark file is compressed")

    _, rows = struct.unpack('<bi', ark_read_buffer.read(5))

    return rows
        
        
def read_target_file(target_path):
    """
    read the file containing the state alignments
    Args:
        target_path: path to the alignment file
    Returns:
        A dictionary containing
            - Key: Utterance ID
            - Value: The state alignments as a space seperated string
    """

    target_dict = {}

    with gzip.open(target_path, 'rb') as fid:
        for line in fid:
            split_line = line.decode('utf-8').strip().split(' ')
            target_dict[split_line[0]] = split_line[1:]
    return target_dict


def splice(utt, context_width):
    """
    splice the utterance
    Args:
        utt: numpy matrix containing the utterance features to be spliced
        context_width: how many frames to the left and right should
            be concatenated
    Returns:
        a numpy array containing the spliced features, if the features are
        too short to splice None will be returned
    """
    # return None if utterance is too short
    if utt.shape[0] < 1+2*context_width:
        return None

    #  create spliced utterance holder
    utt_spliced = np.zeros(
        shape=[utt.shape[0], utt.shape[1]*(1+2*context_width)],
        dtype=np.float32)

    #  middle part is just the utterance
    utt_spliced[:, context_width*utt.shape[1]:
                (context_width+1)*utt.shape[1]] = utt

    for i in range(context_width):

        #  add left context
        utt_spliced[i+1:utt_spliced.shape[0],
                    (context_width-i-1)*utt.shape[1]:
                    (context_width-i)*utt.shape[1]] = utt[0:utt.shape[0]-i-1, :]

        # add right context
        utt_spliced[0:utt_spliced.shape[0]-i-1,
                    (context_width+i+1)*utt.shape[1]:
                    (context_width+i+2)*utt.shape[1]] = utt[i+1:utt.shape[0], :]

    return utt_spliced


def add_delta(utt, delta_order):
    num_frames = utt.shape[0]
    feat_dim = utt.shape[1]

    utt_delta = np.zeros(
        shape=[num_frames, feat_dim * (1 + delta_order)],
        dtype=np.float32)

    #  first order part is just the utterance max_offset+1
    utt_delta[:, 0:feat_dim] = utt

    scales = [[1.0], [-0.2, -0.1, 0.0, 0.1, 0.2],
              [0.04, 0.04, 0.01, -0.04, -0.1, -0.04, 0.01, 0.04, 0.04]]

    delta_tmp = np.zeros(shape=[num_frames, feat_dim], dtype=np.float32)
    for i in range(1, delta_order + 1):
        max_offset = (len(scales[i]) - 1) / 2
        for j in range(-max_offset, 0):
            delta_tmp[-j:, :] = utt[0:(num_frames + j), :]
            for k in range(-j):
                delta_tmp[k, :] = utt[0, :]
            scale = scales[i][j + max_offset]
            if scale != 0.0:
                utt_delta[:, i * feat_dim:(i + 1) * feat_dim] += scale * delta_tmp

        scale = scales[i][max_offset]
        if scale != 0.0:
            utt_delta[:, i * feat_dim:(i + 1) * feat_dim] += scale * utt

        for j in range(1, max_offset + 1):
            delta_tmp[0:(num_frames - j), :] = utt[j:, :]
            for k in range(j):
                delta_tmp[-(k + 1), :] = utt[(num_frames - 1), :]
            scale = scales[i][j + max_offset]
            if scale != 0.0:
                utt_delta[:, i * feat_dim:(i + 1) * feat_dim] += scale * delta_tmp

    return utt_delta


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_conf, audio_path):
        """
        :param audio_conf: audio_conf
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class KaldiFeatParser(AudioParser):
    def __init__(self, audio_conf, target_path):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(KaldiFeatParser, self).__init__()
        self.exp_path = audio_conf['exp_path'] or './'
        self.delta_order = 0
        self.context_width = 0
        self.label_num = audio_conf['label_num'] or 0
        self.num_utt_cmvn = audio_conf['max_num_utt_cmvn'] or 1000
        if target_path is not None:
            self.target_dict = read_target_file(target_path)                        
        else:
            self.target_dict = None
            
    def compute_cmvn(self, audio_conf, utt_ids, num_utt_cmvn):
        cmvn_rand_idx = np.random.permutation(num_utt_cmvn)
        audio_path = utt_ids[0][1]
        mat = self.parse_audio(audio_conf, audio_path)
        nfeatures = mat.shape[1]
        sum_all = np.zeros(shape=[1, nfeatures], dtype=np.float32)
        sum_square_all = np.zeros(shape=[1, nfeatures], dtype=np.float32)
        cmvn = np.zeros(shape=[2, nfeatures], dtype=np.float32)

        frame_count = 0
        print(">> compute cmvn using {0} utterance ".format(num_utt_cmvn))
        p = progressbar.ProgressBar(num_utt_cmvn)
        p.start()
        for n in range(num_utt_cmvn):
            audio_path = utt_ids[cmvn_rand_idx[n]][1]
            feature_mat = self.parse_audio(audio_conf, audio_path)
            if feature_mat is None:
                continue
            sum_1utt = np.sum(feature_mat, axis=0)
            sum_all = np.add(sum_all, sum_1utt)
            feature_mat_square = np.square(feature_mat)
            sum_square_1utt = np.sum(feature_mat_square, axis=0)
            sum_square_all = np.add(sum_square_all, sum_square_1utt)
            frame_count += feature_mat.shape[0]
            p.update(n)
        p.finish()

        mean = sum_all / frame_count
        var = sum_square_all / frame_count - np.square(mean)
        print (frame_count)
        print (mean)
        print (var)
        cmvn[0, :] = -mean
        cmvn[1, :] = 1 / np.sqrt(var)
        return cmvn

    def loading_cmvn(self, cmvn_file, utt_ids, audio_conf):
        audio_path = utt_ids[0][1]
        mat = self.parse_audio(audio_conf, audio_path)
        if mat is None:
            raise Exception('Feat file is not exist!')
        in_size = mat.shape[1]  # count nfeatures
        num_utt_cmvn = min(self.num_utt_cmvn, len(utt_ids))
        if os.path.exists(cmvn_file):
            cmvn = np.load(cmvn_file)
            if cmvn.shape[1] == in_size:
                print ('load cmvn from {}'.format(cmvn_file))
                cmvn = cmvn
            else:
                cmvn = self.compute_cmvn(audio_conf, utt_ids, num_utt_cmvn)
                np.save(cmvn_file, cmvn)
                print ('original cmvn is wrong, so save new cmvn to {}'.
                        format(cmvn_file))
                cmvn = cmvn
        else:
            cmvn = self.compute_cmvn(audio_conf, utt_ids, num_utt_cmvn)
            np.save(cmvn_file, cmvn)
            print ('save cmvn to {}'.format(cmvn_file))
            cmvn = cmvn

        return cmvn

    def parse_audio(self, audio_conf, audio_path):
        self.num_features = audio_conf['num_features'] 
        self.delta_order = audio_conf['delta_order'] or 0
        self.context_width = audio_conf['context_width'] or 0
        
        path, pos = audio_path.split(':')
        ark_read_buffer = open(path, 'rb')
        ark_read_buffer.seek(int(pos), 0)
        header = struct.unpack('<xcccc', ark_read_buffer.read(5))
        if header[0].decode('utf-8') != "B":
            raise Exception("Input .ark file is not binary")
        if header[1].decode('utf-8') == "C":
            raise Exception("Input .ark file is compressed")

        _, rows = struct.unpack('<bi', ark_read_buffer.read(5))
        _, cols = struct.unpack('<bi', ark_read_buffer.read(5))

        if header[1].decode('utf-8') == "F":
            tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 4),
                                    dtype=np.float32)
        elif header[1].decode('utf-8') == "D":
            tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 8),
                                    dtype=np.float64)
        else:
            raise Exception("Input .ark is wrong")

        utt_mat = np.reshape(tmp_mat, (rows, cols))
        ark_read_buffer.close()
        
        if cols > self.num_features:
            utt_mat = utt_mat[:,0:self.num_features]  
              
        if self.delta_order > 0:
            utt_mat = add_delta(utt_mat, self.delta_order)

        if self.context_width > 0:
            utt_mat = splice(utt_mat, self.context_width)

        return utt_mat

    def parse_transcript(self, transcript_path):

        if transcript_path in self.target_dict:
            targets = self.target_dict[transcript_path]
            return targets
        else:
            return None

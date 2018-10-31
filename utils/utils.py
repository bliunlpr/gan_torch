import numpy as np
import data_input
import time
import progressbar
import os
import sys

def test_edit_distance():
    graph = tf.Graph()
    with graph.as_default():
        truth = tf.sparse_placeholder(tf.int32)
        hyp = tf.sparse_placeholder(tf.int32)
        editDist = tf.edit_distance(hyp, truth, normalize=False)

    with tf.Session(graph=graph) as session:
        truthTest = sparse_tensor_feed([[0,1,2], [0,1,2,3,4]])
        hypTest = sparse_tensor_feed([[3,4,5], [0,1,2,2]])
        feedDict = {truth: truthTest, hyp: hypTest}
        dist = session.run([editDist], feed_dict=feedDict)
        print(dist)

def target_list_to_sparse_tensor(targetList):
    '''make tensorflow SparseTensor from list of targets, with each element
    in the list being a list or array with the values of the target sequence
    (e.g., the integer values of a character map for an ASR target string)
    See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
    for example of SparseTensor format'''
    indices = []
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1]+1]
    return (np.array(indices), np.array(vals), np.array(shape))
        


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
	    sequence.
		   If maxlen is provided, any sequence longer than maxlen is truncated to
		  maxlen. Truncation happens off either the beginning or the end
		  (default) of the sequence. Supports post-padding (default) and
		  pre-padding.

		  Args:
			  sequences: list of lists where each element is a sequence
			  maxlen: int, maximum length
			  dtype: type to cast the resulting sequence.
			  padding: 'pre' or 'post', pad either before or after each sequence.
			  truncating: 'pre' or 'post', remove values from sequences larger
			  than maxlen either in the beginning or in the end of the sequence
			  value: float, value to pad the sequences to the desired value.
		  Returns
			  x: numpy array with dimensions (number_of_sequences, maxlen)
			  lengths: numpy array with the original sequence lengths
	  '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

	  # take the sample shape from the first non empty sequence
	  # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
        break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

		# check `trunc` has expected shape
    trunc = np.asarray(trunc, dtype=dtype)
    if trunc.shape[1:] != sample_shape:
        raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
						 (trunc.shape[1:], idx, sample_shape))

    if padding == 'post':
        x[idx, :len(trunc)] = trunc
    elif padding == 'pre':
        x[idx, -len(trunc):] = trunc
    else:
        raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths





def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape
            

class BatchGenerator(object):  

    def __init__(self, specPath, config):
        self.Path = specPath
        self.context_width = config.context_width 
        self.label_num = config.num_classes 
        self.Startpoint = config.Startpoint
        self.LoadSize = config.LoadSize
        self.batchSize = config.batchSize
        self.overlap_size = config.overlap_size
        self.target_is_char = config.target_is_char
        self.counter = -1        
        self.data = None
        self.batchRandIxs = None            
    
    def get_data(self):
        scp_reader = data_input.FeatureReader(self.Path + '/feats.scp', self.context_width)
        if self.target_is_char:
            target_coder = data_input.AlignmentCoder(self.Path + '/labels.char.gz', self.label_num, lambda x, y: x)
        else:
            target_coder = data_input.AlignmentCoder(self.Path + '/labels.phn.gz', self.label_num, lambda x, y: x)

        _, mat = scp_reader.get_utt(0)
        nFeatures = np.shape(mat)[1] ##count nFeatures
    
        randIxs = np.random.permutation(self.LoadSize)
        start, end = (0, self.batchSize)
        dataBatches = []
    
        while end < self.LoadSize:
            batchSeqLengths = np.zeros(self.batchSize)
            batchTargetList = []
            batchInputs = []
            for batchI, origI in enumerate(randIxs[start:end]):
                utt_id, feature_mat = scp_reader.get_utt(origI + self.Startpoint, self.context_width)
                if self.overlap_size > 1:
                    row = np.shape(feature_mat)[0]
                    row_over = int((row + self.overlap_size - 1)/self.overlap_size)
                    feature_mat_overlap = np.zeros(shape=[row_over,nFeatures],dtype=np.float32)
                    for overI,oriI in enumerate(range(0,row,self.overlap_size)): 
                        feature_mat_overlap[overI,:] = feature_mat[oriI,:]
                else:
                    feature_mat_overlap = feature_mat         
                targets = target_coder.get_target(utt_id)
                encoded_targets = target_coder.encode(targets)    
                batchTargetList.append(encoded_targets)  
                batchInputs.append(feature_mat_overlap)
            batch_train_inputs, batch_train_seq_len = pad_sequences(batchInputs)
            dataBatches.append((batch_train_inputs, sparse_tuple_from(batchTargetList),
                          batch_train_seq_len))
            start += self.batchSize
            end += self.batchSize
        self.data = dataBatches   
        self.batchRandIxs = np.random.permutation(len(self.data)) #randomize batch order
        return True
            
    
        
    def get_batch_num(self, gpu_num): 
    
        return (len(self.data)/gpu_num)
                         
    def next_batch(self):         
        self.counter += 1
        return self.data[self.batchRandIxs[self.counter]]


class FrameBatchGenerator(object):  

    def __init__(self, specPath, args):
        self.Path = specPath
        self.args = args
        self.context_width = args.context_width 
        self.delta_order = args.delta_order
        self.label_num = args.num_classes 
        self.batchSize = args.batchSize
        
        self.LoadSize_per_batch = args.LoadSize_per_batch
        self.Startpoint = 0
        self.LoadSize = 0
        self.nLoads = 0
        self.utt_num = 0
        self.cmvn_utt_num = 5000
        self.cmvn = None
        
        self.counter = -1        
        self.data = None
        self.batchRandIxs = None
        self.uttrandIxs = None

    def get_cmvn(self, scp_reader):
        _, mat = scp_reader.get_utt(0)
        nFeatures = np.shape(mat)[1]        
        sum_all = np.zeros(shape=[1, nFeatures],dtype=np.float32)
        sum_square_all = np.zeros(shape=[1, nFeatures],dtype=np.float32)
        cmvn = np.zeros(shape=[2, nFeatures],dtype=np.float32)
        frame_count = 0
        Index = 0
        
        if self.cmvn_utt_num > self.utt_num:
            self.cmvn_utt_num = self.utt_num  
            
        while Index < self.cmvn_utt_num:
            utt_id, feature_mat = scp_reader.get_utt(self.uttrandIxs[Index])
            sum_1utt = np.sum(feature_mat, axis=0)
            sum_all = np.add(sum_all, sum_1utt)
            feature_mat_square = np.square(feature_mat)
            sum_square_1utt = np.sum(feature_mat_square, axis=0) 
            sum_square_all = np.add(sum_square_all, sum_square_1utt)  
            frame_count += np.shape(feature_mat)[0]
            Index += 1             
        mean = sum_all/frame_count
        var = sum_square_all/frame_count - np.square(mean)
        cmvn[0,:] = -mean
        cmvn[1,:] = 1/np.sqrt(var)
        return cmvn
        
    def get_loads_num(self):        
        utt_num = 0
        for count, line in enumerate(open(self.Path + '/feats.scp', 'rU')):
            utt_num += 1      
        print ('totalN={0}'.format(utt_num))
        self.utt_num = utt_num
        self.uttrandIxs = np.random.permutation(utt_num)
        
        nLoads = int((utt_num + self.LoadSize_per_batch - 1)/self.LoadSize_per_batch) 
        self.nLoads = nLoads
        
        scp_reader = data_input.FeatureReader(self.Path + '/feats.scp', self.delta_order, self.context_width)
        _, mat = scp_reader.get_utt(0)
        nFeatures = np.shape(mat)[1] ##count nFeatures
        
        if os.path.isfile(self.Path + '/cmvn.npy'):
            cmvn = np.load(self.Path + '/cmvn.npy')
            if np.shape(cmvn)[1] == nFeatures:
                print ('load cmvn from {}'.format(self.Path))
                self.cmvn = cmvn
            else:
                cmvn = self.get_cmvn(scp_reader)
                np.save(self.Path + '/cmvn.npy', cmvn)
                print ('original cmvn is wrong, so save new cmvn to {}'.format(self.Path))
                self.cmvn = cmvn    
        else:         
            cmvn = self.get_cmvn(scp_reader)
            np.save(self.Path + '/cmvn.npy', cmvn)
            print ('save cmvn to {}'.format(self.Path))
            self.cmvn = cmvn 
                 
        return nLoads   
                    
    def get_data(self, load):
        self.counter = -1   
        scp_reader = data_input.FeatureReader(self.Path + '/feats.scp', self.delta_order, self.context_width)
        target_coder = data_input.AlignmentCoder(self.Path + '/labels.frame.gz', self.label_num, lambda x, y: x)
    
        _, mat = scp_reader.get_utt(0)
        nFeatures = np.shape(mat)[1] ##count nFeatures 
        window = 2 * self.context_width + 1  
        feat_dim = nFeatures/(window*(self.delta_order+1))        
                      		
        st = time.time()
        dataTmps = []
        total_frame = 0
        target_error = 0
        success = 0
        
        self.Startpoint = load * self.LoadSize_per_batch
        print('Loading train data Startpoint={0}'.format(self.Startpoint))            
        if load < (self.nLoads - 1):
            self.LoadSize = self.LoadSize_per_batch
        else: 
            self.LoadSize = self.utt_num % self.LoadSize_per_batch
                
        Index = self.Startpoint
        while Index < self.Startpoint + self.LoadSize:
            utt_id, feature_mat = scp_reader.get_utt(self.uttrandIxs[Index])
            feature_mat = (feature_mat + self.cmvn[0,:]) * (self.cmvn[1,:])
            row = np.shape(feature_mat)[0]
            utt_id_slpit = utt_id.split('_')
            target_utt_id = utt_id_slpit[0] + '_' + utt_id_slpit[1] + '_ORG'	                
            ##target_utt_id = utt_id 		
            targets = target_coder.get_target(target_utt_id)            
            if targets is None:
                print ('{} has no target {}').format(utt_id,target_utt_id)
                Index += 1
                target_error += 1
                continue
            else:
                encoded_targets = target_coder.encode(targets) 
            assert	row ==  np.shape(encoded_targets)[0],'{} target and feature have no match dim'.format(utt_id)

            for frameId in range(row):
                ##train_inputs = feature_mat[frameId,:].reshape((window,self.delta_order+1,-1)).transpose((1,2,0))
                train_inputs = feature_mat[frameId,:]
                Target = encoded_targets[frameId]
                dataTmps.append((train_inputs, Target))
            total_frame += row
            success += 1
            Index += 1
        assert 	total_frame == len(dataTmps),'len(dataTmps) is {0} and len(dataTmps) is {1}'.format(total_frame,len(dataTmps))
        print ('target_error={} success={}'.format(target_error,success))			
        dataBatches = []
        total_batch = 0
        batchSize = self.batchSize
        frameRandIxs = np.random.permutation(total_frame)
        start, end = (0, batchSize)
        		
        bar = progressbar.ProgressBar()
        for i in bar(range(total_frame/batchSize)):
            ##batchInputs = np.zeros(shape=[batchSize, self.delta_order+1, feat_dim, window],dtype=np.float32)
            batchInputs = np.zeros(shape=[batchSize, nFeatures],dtype=np.float32)
            batchTarget = np.zeros((batchSize),dtype=np.int64)
            for batchI, origI in enumerate(frameRandIxs[start:end]):
                train_inputs, Target = dataTmps[origI]
                batchInputs[batchI,:] = train_inputs
                batchTarget[batchI] = Target
            dataBatches.append((batchInputs, batchTarget))	
            total_batch += 1
            start += batchSize
            end += batchSize
        print ('batchSize is {0}, total_batch is {1}, took {2:.2f} seconds'.format(batchSize, total_batch, time.time() - st))
        self.data = dataBatches   
        self.batchRandIxs = np.random.permutation(len(self.data)) #randomize batch order
        del dataTmps
                
        return (len(self.data))
                         
    def next_batch(self):         
        self.counter += 1
        return self.data[self.batchRandIxs[self.counter]]

def compute_cmvn(scp_reader, max_num_utt):
        
    cmvn_rand_idx = np.random.permutation(max_num_utt)
    _, mat = scp_reader.get_utt(0)
    nFeatures = np.shape(mat)[1]        
    sum_all = np.zeros(shape=[1, nFeatures],dtype=np.float32)
    sum_square_all = np.zeros(shape=[1, nFeatures],dtype=np.float32)
    cmvn = np.zeros(shape=[2, nFeatures],dtype=np.float32)

    frame_count = 0
    print(">> compute cmvn using {0} utterance ".format(max_num_utt))
    p = progressbar.ProgressBar(max_num_utt)
    p.start()
    for n in xrange(max_num_utt):
        utt_id, feature_mat = scp_reader.get_utt(cmvn_rand_idx[n])
        if feature_mat is None:
            continue
        sum_1utt = np.sum(feature_mat, axis=0)
        sum_all = np.add(sum_all, sum_1utt)
        feature_mat_square = np.square(feature_mat)
        sum_square_1utt = np.sum(feature_mat_square, axis=0) 
        sum_square_all = np.add(sum_square_all, sum_square_1utt)  
        frame_count += np.shape(feature_mat)[0]
        p.update(n)
    p.finish()

    mean = sum_all / frame_count
    var = sum_square_all / frame_count - np.square(mean)
    print frame_count
    print mean
    print var
    cmvn[0, :] = -mean
    cmvn[1, :] = 1 / np.sqrt(var)
    return cmvn
                    
class Gan_FrameBatchGenerator(object):
    def __init__(self, specpath, args):
        self.args = args
        self.log_dir = args.log_dir
        self.path = specpath
        self.label_num = args.num_classes 
        self.cmvn_file = os.path.join(self.log_dir, 'cmvn.npy')
        self.org_cmvn_file = os.path.join(self.log_dir, 'org_cmvn.npy')
        if not os.path.isdir(self.log_dir):
            raise Exception(self.log_dir + ' isn.t a path!')

        utt_num = 0
        self.data_scp = self.path + '/feats.scp'
        for count, line in enumerate(open(self.data_scp, 'rU')):
            utt_num += 1      
        print ('totalN={0}'.format(utt_num))
        self.num_utt = utt_num
        self.num_loaded_utt = 0
        self.utt_rand_idx = np.random.permutation(utt_num)

        self.context_width = args.context_width 
        self.delta_order = args.delta_order
        self.org_delta_order = args.org_delta_order
        self.org_context_width = args.org_context_width 
        
        scp_reader = data_input.FeatureReader(self.data_scp, self.delta_order, self.context_width)
        _, mat = scp_reader.get_utt(0)
        if mat is None:
            raise Exception('Feat file is not exist!')
        self.feat_size = np.shape(mat)[1] ##count nFeatures
        # compute or load cmvn
        self.num_utt_cmvn = min(args.max_num_utt_cmvn, self.num_utt)
        if os.path.exists(self.cmvn_file):
            cmvn = np.load(self.cmvn_file)
            if np.shape(cmvn)[1] == self.feat_size:
                print ('load cmvn from {}'.format(self.cmvn_file))
                self.cmvn = cmvn
            else:
                cmvn = compute_cmvn(scp_reader, self.num_utt_cmvn)
                np.save(self.cmvn_file, cmvn)
                print ('original cmvn is wrong, so save new cmvn to {}'.format(self.cmvn_file))
                self.cmvn = cmvn
        else:
            cmvn = compute_cmvn(scp_reader, self.num_utt_cmvn)
            np.save(self.cmvn_file, cmvn)
            print ('save cmvn to {}'.format(self.cmvn_file))
            self.cmvn = cmvn
        
        utt_num = 0
        self.org_data_scp = self.path + '/org_feats.scp'
        for count, line in enumerate(open(self.org_data_scp, 'rU')):
            utt_num += 1      
        print ('totalN={0}'.format(utt_num))
        self.org_num_utt = utt_num        
        org_scp_reader = data_input.FeatureReader(self.org_data_scp, self.org_delta_order, self.org_context_width)
        _, mat = scp_reader.get_utt(0)
        if mat is None:
            raise Exception('Feat file is not exist!')
        self.org_feat_size = np.shape(mat)[1] ##count nFeatures       
        self.num_utt_cmvn = min(args.max_num_utt_cmvn, self.org_num_utt)
        if os.path.isfile(self.org_cmvn_file):
            cmvn = np.load(self.org_cmvn_file)
            if np.shape(cmvn)[1] == self.org_feat_size:
                print ('load org_cmvn from {}'.format(self.org_cmvn_file))
                self.org_cmvn = cmvn
            else:
                cmvn = compute_cmvn(org_scp_reader, self.num_utt_cmvn)
                np.save(self.org_cmvn_file, cmvn)
                print ('original org_cmvn is wrong, so save new org_cmvn to {}'.format(self.org_cmvn_file))
                self.org_cmvn = cmvn      
        else:    
            cmvn = compute_cmvn(org_scp_reader, self.num_utt_cmvn)
            np.save(self.org_cmvn_file, cmvn)
            print ('save org_cmvn to {}'.format(self.org_cmvn_file))
            self.org_cmvn = cmvn 
            
        # parameter for batching
        self.frame_rand_idx = None
        self.org_frame_rand_idx = None
        self.num_frames = 0
        self.org_num_frames = 0
        self.num_readed_frame = 0

        self.noise_inputs = None
        self.org_inputs = None
        self.outputs = None
        self.num_utt_per_loading = args.num_utter_per_loading

    def loadcmvn(self):
        if self.org_cmvn is not None:
            return self.org_cmvn

    def re_disorder(self):
        self.num_loaded_utt = 0
        self.utt_rand_idx = np.random.permutation(self.num_utt)
        self.noise_inputs = None
        self.org_inputs = None
        self.outputs = None
        self.frame_rand_idx = None
        self.org_frame_rand_idx = None
        self.num_frames = 0
        self.org_num_frames = 0
        self.num_readed_frame = 0

    def next_load(self, num_loading_utt):
        num_cur_load = 0
        num_frames = 0
        org_num_frames = 0
        org_error = 0
        target_error = 0
        success = 0
        noise_input_list = []
        org_input_list = []
        output_list = []
        bar = progressbar.ProgressBar(num_loading_utt)
        bar.start()
              
        scp_reader = data_input.FeatureReader(self.data_scp, self.delta_order, self.context_width)
        org_scp_reader = data_input.FeatureReader(self.org_data_scp, self.org_delta_order, self.org_context_width)
        target_coder = data_input.AlignmentCoder(self.path + '/labels.frame.gz', self.label_num, lambda x, y: x)
        
        st = time.time()
        
        while num_cur_load < num_loading_utt and self.num_loaded_utt < self.num_utt:        
            bar.update(num_cur_load)
            
            utt_id, feature_mat = scp_reader.get_utt(self.utt_rand_idx[self.num_loaded_utt])
            num_cur_load += 1
            self.num_loaded_utt += 1
            feature_mat = (feature_mat + self.cmvn[0,:]) * (self.cmvn[1,:])
            if np.isnan(feature_mat).sum() > 0:
                continue    
            row = np.shape(feature_mat)[0]
            path_slpit = self.path.split('/')
            path_last = path_slpit[-1] or path_slpit[-2]
            if path_last == 'train' or 'dev':
                utt_id_slpit = utt_id.split('_')
                wsj_name = utt_id_slpit[1]
                speaker = wsj_name[0:3]
                org_utt_id = speaker + '_' + wsj_name + '_ORG'	
            else:
                utt_id_slpit = utt_id.split('_')
                wsj_name = utt_id_slpit[1]
                speaker = utt_id_slpit[0]
                channel = utt_id_slpit[2].split('.')[1]                
                org_utt_id = speaker + '_' + wsj_name + '_BTH.' + channel	
            org_feature_mat = org_scp_reader.get_ids_utt(org_utt_id)
            if org_feature_mat is None:
                print ('{} has no org_feature {}').format(utt_id, org_utt_id)
                org_error += 1
                continue
            else:
                org_feature_mat = (org_feature_mat + self.org_cmvn[0,:]) * (self.org_cmvn[1,:])
                                            
            target_utt_id = org_utt_id 		 		
            targets = target_coder.get_target(target_utt_id)            
            if targets is None:
                print ('{} has no target {}').format(utt_id,target_utt_id)
                target_error += 1
                continue
            else:
                encoded_targets = target_coder.encode(targets) 
            assert	row ==  np.shape(encoded_targets)[0],'{} feature dim is {} and target {}'.format(utt_id, row, np.shape(encoded_targets)[0]) 
            noise_input_list.append(feature_mat)
            org_input_list.append(org_feature_mat)
            output_list.append(encoded_targets)
            num_frames += row
            org_num_frames += np.shape(org_feature_mat)[0]
            success += 1
        bar.finish()
        print ('org_error={} target_error={} success={}'.format(org_error,target_error,success))	
        print ('num_frames={}, org_num_frames={}').format(num_frames, org_num_frames)
        if num_frames <= 0:
            return num_frames

        self.num_frames = num_frames
        self.org_num_frames = num_frames
        self.num_readed_frame = 0
        self.frame_rand_idx = np.random.permutation(self.num_frames)
        self.org_frame_rand_idx = np.random.permutation(self.num_frames)
        org_frame_rand_idx = np.random.permutation(self.org_num_frames)
        if self.num_frames > self.org_num_frames:
            tmp = np.random.permutation(self.org_num_frames)
            self.org_frame_rand_idx[:self.org_num_frames] = org_frame_rand_idx
            self.org_frame_rand_idx[self.org_num_frames:] = tmp[0:(self.num_frames - self.org_num_frames)]  
        else:
            self.org_frame_rand_idx = org_frame_rand_idx[:self.num_frames]  
            
        assert np.shape(self.frame_rand_idx)[0] == np.shape(self.org_frame_rand_idx)[0]
        
        self.noise_inputs = np.zeros(shape=[self.num_frames, self.feat_size],dtype=np.float32)
        self.org_inputs = np.zeros(shape=[self.num_frames, self.org_feat_size],dtype=np.float32)
        self.outputs = np.zeros((self.num_frames), dtype=np.int64)

        s = 0
        for feat in noise_input_list:
            e = s + np.shape(feat)[0]
            self.noise_inputs[s:e,:] = feat
            s = e
            
        s = 0
        for feat in org_input_list:
            e = s + np.shape(feat)[0]
            self.org_inputs[s:e,:] = feat
            s = e
            
        s = 0
        for label in output_list:
            e = s + np.shape(label)[0]
            self.outputs[s:e] = label
            s = e
            
        et = time.time()
        print ('Loading {0} frame of audio data, Using {1} seconds'.format(self.num_frames, et - st))
        return self.num_frames
        
    def next_batch(self, batch_size):
        if self.noise_inputs is None or self.outputs is None:
            if self.next_load(self.num_utt_per_loading) <= 0:
                return (None, None, None)

        s = self.num_readed_frame
        e = s + batch_size

        while s >= self.num_frames or e > self.num_frames:
            if self.next_load(self.num_utt_per_loading) <= 0:
                return (None, None, None)
            else:
                s = self.num_readed_frame
                e = s + batch_size

        self.num_readed_frame += e - s
        return (self.noise_inputs[self.frame_rand_idx[s:e],:],self.org_inputs[self.org_frame_rand_idx[s:e],:],self.outputs[self.frame_rand_idx[s:e]])

class UttGenerator(object):  

    def __init__(self, specpath, args):
        self.path = specpath
        self.args = args
        self.log_dir = args.log_dir
        self.cmvn_file = os.path.join(self.log_dir, 'cmvn.npy')
        self.context_width = args.context_width 
        self.delta_order = args.delta_order
        self.label_num = args.num_classes 
        
        self.num_utt_per_loading = args.num_utter_per_loading
        self.num_loaded_utt = 0
        self.num_readed_utt = 0
        self.num_cur_load = 0
        self.input_list = []
        self.utt_list = []            
        
        self.data_scp = self.path + '/feats.scp'
        utt_num = 0
        for count, line in enumerate(open(self.data_scp, 'rU')):
            utt_num += 1      
        print ('totalN={0}'.format(utt_num))
        self.num_utt = utt_num
        
        scp_reader = data_input.FeatureReader(self.data_scp, self.delta_order, self.context_width)
        _, mat = scp_reader.get_utt(0)
        self.feat_size = np.shape(mat)[1]
        if os.path.exists(self.cmvn_file):
            cmvn = np.load(self.cmvn_file)
            if np.shape(cmvn)[1] == self.feat_size:
                print ('load cmvn from {}'.format(self.cmvn_file))
                self.cmvn = cmvn
            else:
                raise Exception("the loaded cmvn is wrong, please provide the right cmvn file")
        else:
            raise Exception("%s is not exists, please check the cmvn file")
     
     
    def next_load(self, num_loading_utt):
        num_cur_load = 0  
        num_frames = 0       
        self.input_list = []         
        self.utt_list = []              
        scp_reader = data_input.FeatureReader(self.data_scp, self.delta_order, self.context_width)
        
        st = time.time()
        
        while num_cur_load < num_loading_utt and self.num_loaded_utt < self.num_utt: 
        
            utt_id, feature_mat = scp_reader.get_utt(self.num_loaded_utt)
            num_cur_load += 1
            self.num_loaded_utt += 1
            feature_mat = (feature_mat + self.cmvn[0,:]) * (self.cmvn[1,:])
            if np.isnan(feature_mat).sum() > 0:
                continue    
            num_frames += np.shape(feature_mat)[0]
            self.input_list.append(feature_mat)
            self.utt_list.append(utt_id)
        
        if num_frames <= 0:
            return num_frames
        self.num_cur_load = num_cur_load
        self.num_readed_utt = 0
    
        return num_frames
    def next_utt(self):         
        if self.input_list is []:
            if self.next_load(self.num_utt_per_loading) <= 0:
                return (None, None)
        
        s = self.num_readed_utt

        if s >= self.num_cur_load:
            if self.next_load(self.num_utt_per_loading) <= 0:
                return (None, None)
            else:
                s = self.num_readed_utt

        self.num_readed_utt += 1
        return (self.utt_list[s], self.input_list[s])


    def compute_target_count(self):
        
        target_coder = data_input.AlignmentCoder(os.path.join(self.args.train_scp, 'labels.frame.gz'), self.label_num, lambda x, y: x)
        encoded_targets = np.concatenate(
            [target_coder.encode(targets)
             for targets in target_coder.target_dict.values()])

        #count the number of occurences of each target
        count = np.bincount(encoded_targets,
                            minlength=self.label_num)

        return count
    
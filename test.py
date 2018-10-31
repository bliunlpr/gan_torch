from __future__ import print_function
import argparse
import os
import numpy as np
import random
import shutil
import time 
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from tqdm import tqdm
from model import AeGenerator, CnnLabelNet, DnnLabelNet
from data.data_loader import SequentialDataset, GanSequentialDataset
from data.data_loader import SequentialDataLoader, GanSequentialDataLoader
from data.audioparse import Targetcounter
from utils.kaldi_io import ArkWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--works_dir', type=str, help='the path to dataset', default="/home/bliu/SRC/workspace/pytf/gan_torch/")
parser.add_argument('--train_scp', type=str, help='the scp file for training', default="/home/bliu/SRC/workspace/pytf/data/chime4/train/multi_6ch")
parser.add_argument('--val_scp', type=str, help='the scp file for dev', default="/home/bliu/SRC/workspace/pytf/data/chime4/dev/multi_1ch")
parser.add_argument('--eval_scp', type=str, help='the scp file for evaluation', default="/home/bliu/SRC/workspace/pytf/data/chime4/test/multi_1ch")

parser.add_argument('--decoder_data', type=str, choices=["eval", "val"], help='the scp file for evaluation', default="eval")
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--batch_size', type=int, default=384, help='the batch size for once training')

parser.add_argument('--log_dir', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

                
def main():
    global args
    args = parser.parse_args()
    print(args)
    
    supported_decoder_data = {
        'eval': args.eval_scp,
        'val': args.val_scp
    }
    
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)    
    cudnn.benchmark = True
    
    model_path = os.path.join(args.works_dir, args.resume)
    if not os.path.isfile(model_path):
        raise Exception("no checkpoint found at {}".format(model_path))
    log_dir, _ = os.path.split(model_path)
    args.log_dir = log_dir
    decode_dir = os.path.join(args.log_dir, 'decode_' + args.decoder_data)    
    if os.path.exists(decode_dir):
        print('Directory {} already exists and remove it.'.format(decode_dir))
        shutil.rmtree(decode_dir)
        os.makedirs(decode_dir)
    else:
        os.makedirs(decode_dir)
    
    data_dir = supported_decoder_data[args.decoder_data]
    print ('data_dir is {}'.format(data_dir))
    
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    args = package.get('args', args)
    audio_conf = package.get('audio_conf')
    
    generator = AeGenerator.load_model(model_path, 'gen_state_dict')
    generator = torch.nn.DataParallel(generator).cuda()
    if args.label_net_type == 'DNN':
        labelnet = DnnLabelNet.load_model(model_path, 'label_state_dict')
    elif args.label_net_type == 'CNN':
        labelnet = CnnLabelNet.load_model(model_path, 'label_state_dict')
    else:
        raise Exception("wrong label_net_type {}".format(args.label_net_type))    
    labelnet = torch.nn.DataParallel(labelnet).cuda()
    generator.eval()
    labelnet.eval()
    
    with open(os.path.join(decode_dir, 'resume_info'), 'a+') as fwrite:
        fwrite.write(model_path + '\n')
    
    dataset = SequentialDataset(data_scp=os.path.join(data_dir, 'feats.scp'),
                                label_file=None,
                                audio_conf=audio_conf)
    dataloader = SequentialDataLoader(dataset, batch_size=1,
                                      num_workers=args.num_workers)    
    shutil.copy(os.path.join(data_dir, 'text'), decode_dir)
    shutil.copy(os.path.join(data_dir, 'utt2spk'), decode_dir)    
    writer = ArkWriter(decode_dir + '/feats.scp_tmp', decode_dir + '/likelihoods.ark')
    
    target_counter = Targetcounter(os.path.join(args.train_scp, 'labels.frame.gz'), audio_conf['label_num'])                                       
    prior = target_counter.compute_target_count().astype(np.float32)
    prior = prior/prior.sum()
    np.save(os.path.join(args.log_dir, 'prior.npy'), prior)
                
    for i, (data) in tqdm(enumerate(dataloader, start=0), total=len(dataloader)):
        utt_ids, inputs, _, _, _ = data
        utt_id = utt_ids[0]   
        inputs = Variable(inputs, volatile=True).cuda()
        
        _, acoustic_input = generator(inputs)  # compute output
        output = labelnet(acoustic_input)
        out = output.cpu().data.numpy()
        # floor the values to avoid problems with log
        np.where(prior == 0, np.finfo(float).eps, prior)
        prior_log = np.log(prior)
        out = out - prior_log
        # write the pseudo-likelihoods in kaldi feature format
        writer.write_next_utt(utt_id, out)
        
    writer.close()
                            
if __name__ == '__main__':
    main()
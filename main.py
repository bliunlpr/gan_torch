from __future__ import print_function
import argparse
import os
import random
import shutil
import time 
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from model import AeGenerator
from model import CnnDiscriminator
from model import DnnDiscriminator
from model import CnnLabelNet
from model import DnnLabelNet
from trainer import CeFrameTrainer
from trainer import MseFrameTrainer
from trainer import GanFrameTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--works_dir', type=str, help='the path to dataset', default="/home/bliu/SRC/workspace/pytf/gan_torch/")
parser.add_argument('--train_scp', type=str, help='the scp file for training', default="/home/bliu/SRC/workspace/pytf/data/chime4/train/multi_1ch")
parser.add_argument('--val_scp', type=str, help='the scp file for dev', default="/home/bliu/SRC/workspace/pytf/data/chime4/dev/multi_1ch")
parser.add_argument('--eval_scp', type=str, help='the scp file for evaluation', default="/home/bliu/SRC/workspace/pytf/data/chime4/test/multi_1ch")

parser.add_argument('--num_utt_per_loading', type=int, help='the number of utterances for each loading', default=10)
parser.add_argument('--max_num_utt_cmvn', type=int, help='the max number of utterances for cmvn', default=20000)
parser.add_argument('--num_workers', type=int, help='the num_workers', default=4)
parser.add_argument('--batch_size', type=int, default=384, help='the batch size for once training')
parser.add_argument('--context_width', type=int, default=9, help='input context-width')
parser.add_argument('--org_context_width', type=int, default=9, help='input org-context-width')
parser.add_argument('--delta_order', type=int, default=0, help='input delta-order')
parser.add_argument('--org_delta_order', type=int, default=0, help='input delta-order')
parser.add_argument('--num_features', type=int, default=39, help='num-features')
parser.add_argument('--num_classes', type=int, default=0, help='num of label classes')

parser.add_argument('--g_skip_out', dest='g_skip_out', action='store_true', help='skip_out to generator')
parser.add_argument('--g_bn', dest='g_bn', action='store_true', help='bn to generator')
parser.add_argument('--g_drop', default=0.0, type=float, help='g_drop')
parser.add_argument('--g_act_type', type=str, default='relu', help="Type of nonlinearity in G: leaky or prelu. (Def: leaky).")
parser.add_argument('--g_dnn_num_layer', type=int, default=2, help='layer num of g')
parser.add_argument('--g_dnn_hidden_dim', type=int, default=1024, help='dim of hidden') 
parser.add_argument('--g_dnn_bottleneck_dim', type=int, default=1024, help='dim of hidden')  

parser.add_argument('--d_net_type', type=str, default='DNN', help="Type of d net: cnn or dnn. (Def: dnn).")
parser.add_argument('--d_act_type', type=str, default='relu', help="Type of nonlinearity in D: leakyrelu or relu. (Def: leakyrelu).")
parser.add_argument('--d_bn', dest='d_bn', action='store_true', help='if norm for D')
parser.add_argument('--d_drop', default=0.0, type=float, help='d_drop')
parser.add_argument('--d_dnn_hidden_dim', type=int, default=1024, help='dim of hidden')  
parser.add_argument('--d_dnn_num_layer', type=int, default=2, help='layer num of d')

parser.add_argument('--label_net_type', type=str, default='DNN', help="Type of label net: cnn or dnn. (Def: dnn).")
parser.add_argument('--label_act_type', type=str, default='relu', help="Type of nonlinearity in D: leakyrelu or relu. (Def: leakyrelu).")
parser.add_argument('--label_drop', default=0.0, type=float, help='label_drop')
parser.add_argument('--label_bn', dest='label_bn', action='store_true', help='if norm for label')
parser.add_argument('--label_dnn_in_dim', type=int, default=1280, help='dim of bn')  
parser.add_argument('--label_dnn_hidden_dim', type=int, default=1024, help='dim of hidden')  
parser.add_argument('--label_dnn_num_layer', type=int, default=2, help='layer num of label')

parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--log_dir', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--print_freq', '-p', default=500, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--validate_freq', type=int, default=500, help='how many batches to validate the trained model')   
parser.add_argument('--pair', dest='pair', action='store_true', help='pair to train model')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--debug', dest='debug', action='store_true', help='debug to model')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--num_val', default=0, type=int, help='num_val') 
parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning_anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--best_prec', default=0., type=float, help='best_prec')
parser.add_argument('--dev_acc', default=0., type=float, help='best_wer')
parser.add_argument('--dev_loss', default=99999., type=float, help='dev_loss_prev')

parser.add_argument('--alpha', type=float, default=0.0, help='alpha, default=0.2')
parser.add_argument('--dealpha_epoch', type=int, default=8, help='Epoch where noise in disc is removed (Def: 5).')                             
parser.add_argument('--dealpha_bound', type=float, default=0.05, help='Min noise std to be still alive (Def: 0.001)')
parser.add_argument('--dealpha_decay', type=float, default=1.0, help='Decay rate of noise std (Def: 0.7)')
parser.add_argument('--g_steps', type=int, default=1, help='how many steps to train G when gan training')   
parser.add_argument('--d_steps', type=int, default=1, help='how many steps to train D when gan training')   

                
def main():
    global args
    args = parser.parse_args()
    print(args)
    
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)    
    cudnn.benchmark = True
    
    if args.resume:
        model_path = os.path.join(args.works_dir, args.resume)
        if not os.path.isfile(model_path):
            raise Exception("no checkpoint found at {}".format(model_path))
        log_dir, _ = os.path.split(model_path)
        args.log_dir = log_dir
        
        package = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = package.get('args', args)
        args.lr = package.get('learning_rate', args.lr)
        args.best_prec = package.get('best_prec', args.best_prec)
        args.dev_acc = package.get('dev_acc', args.dev_acc)
        args.dev_loss = package.get('dev_loss', args.dev_loss)
        args.alpha = package.get('alpha', args.alpha)
        args.start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training        
        args.num_val = package.get('num_val', args.num_val)
        audio_conf = package.get('audio_conf')
        
        generator = AeGenerator.load_model(model_path, 'gen_state_dict')
        gen_parameters = generator.parameters()
        gen_optimizer = optim.Adam(gen_parameters, lr=args.lr, betas=(args.beta1, 0.999)) 
        ##gen_optimizer = optim.SGD(gen_parameters, lr=args.lr, momentum=args.momentum, nesterov=True)
        ##gen_optimizer.load_state_dict(package['gen_optim_dict'])
        
        if args.d_net_type == 'DNN':
            discriminator = DnnDiscriminator.load_model(model_path, 'dis_state_dict')
        elif args.d_net_type == 'CNN':
            discriminator = CnnDiscriminator.load_model(model_path, 'dis_state_dict')
        else:
            raise Exception("wrong d_net_type {}".format(args.d_net_type))  
        dis_parameters = discriminator.parameters()
        dis_optimizer = optim.Adam(dis_parameters, lr=args.lr, betas=(args.beta1, 0.999))   
        ##dis_optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, nesterov=True)
        ##dis_optimizer.load_state_dict(package['dis_optim_dict'])
        
        if args.label_net_type == 'DNN':
            labelnet = DnnLabelNet.load_model(model_path, 'label_state_dict')
        elif args.label_net_type == 'CNN':
            labelnet = CnnLabelNet.load_model(model_path, 'label_state_dict')
        else:
            raise Exception("wrong label_net_type {}".format(args.label_net_type))    
        label_parameters = labelnet.parameters()
        label_optimizer = optim.Adam(label_parameters, lr=args.lr, betas=(args.beta1, 0.999))   
        ##label_optimizer = optim.SGD(label_parameters, lr=args.lr, momentum=args.momentum, nesterov=True)
        ##label_optimizer.load_state_dict(package['label_optim_dict'])
        
        print("=> loaded checkpoint '{}' (epoch {})".format(model_path, args.start_epoch))      
    else:
        with open(args.train_scp + '/num_pdfs', 'r') as fid:
            dict_num = int(fid.read()) 
        print('dict_num is {0}'.format(dict_num))
        args.num_classes = dict_num
    
        record_name = 'record_lr{}_gan_alpha{}_dealpha{}'.format(args.lr, args.alpha, args.dealpha_decay) 
        if args.g_skip_out:
            record_name += '_skip'
        else:
            record_name += '_noskip'          
        if args.g_bn:
            record_name += '_bn'
        else:
            record_name += '_nobn'      
        if args.pair:
            record_name += '_pair'
        
        exp_path = os.path.join(args.works_dir, 'exp')
        args.log_dir = os.path.join(exp_path, record_name)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        
        audio_conf = dict(exp_path=args.log_dir,
                          num_features=args.num_features,
                          delta_order=args.delta_order,
                          context_width=args.context_width,
                          pair=args.pair,
                          org_delta_order=args.org_delta_order,
                          org_context_width=args.org_context_width,
                          label_num=args.num_classes,
                          normalize_type=1,
                          org_normalize_type=0,
                          max_num_utt_cmvn=args.max_num_utt_cmvn)
                               
        generator = AeGenerator(args)
        gen_parameters = generator.parameters()
        gen_optimizer = optim.Adam(gen_parameters, lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay) 
        ##gen_optimizer = optim.RMSprop(gen_parameters, lr=args.lr)
        ##gen_optimizer = optim.SGD(gen_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)       
 
        if args.d_net_type == 'DNN':
            discriminator = DnnDiscriminator(args)
        elif args.d_net_type == 'CNN':
            discriminator = CnnDiscriminator(args)
        else:
            raise Exception("wrong d_net_type {}".format(args.d_net_type))     
        dis_parameters = discriminator.parameters()
        dis_optimizer = optim.Adam(dis_parameters, lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay) 
        ##dis_optimizer = optim.RMSprop(dis_parameters, lr=args.lr)
        ##dis_optimizer = optim.SGD(dis_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)     
    
        if args.label_net_type == 'DNN':
            labelnet = DnnLabelNet(args)
        elif args.label_net_type == 'CNN':
            labelnet = CnnLabelNet(args)
        else:
            raise Exception("wrong label_net_type {}".format(args.label_net_type))    
        label_parameters = labelnet.parameters()   
        label_optimizer = optim.Adam(label_parameters, lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay) 
        ##label_dis_optimizer = optim.RMSprop(label_dis_parameters, lr=args.lr) 
        ##label_dis_optimizer = optim.SGD(label_dis_parameters, lr=args.lr, momentum=args.momentum)  
    
    criterion_label = nn.CrossEntropyLoss().cuda()
    criterion_mse = nn.MSELoss().cuda()
    
    model_list = [generator, discriminator, labelnet]
    criterion_list = [criterion_label, criterion_mse]
    optimizer_list = [gen_optimizer, dis_optimizer, label_optimizer]
    
    for model in model_list:
        print (model)
        print (model.state_dict().keys())
        print("Number of parameters: {}".format(model.get_param_size(model)))
        
    trainer = GanFrameTrainer(args, audio_conf, model_list, criterion_list, optimizer_list)    
    trainer.train()                       

    
if __name__ == '__main__':
    main()

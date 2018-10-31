import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import math
import numpy as np
import torch.utils.model_zoo as model_zoo

supported_acts = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(), 'tanh': nn.Tanh(), 
                  'leakyrelu': nn.LeakyReLU(), 'prelu': nn.PReLU()}

class ModelBase(nn.Module):
    """
    ModelBase class for sharing code among various model.
    """
    def forward(self, x):
        raise NotImplementedError
    
    @classmethod
    def load_model(cls, path, state_dict):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(args=package['args'])
        if package[state_dict] is not None:
            model.load_state_dict(package[state_dict])    
        return model
        
    @staticmethod
    def serialize(model, state_dict, optimizer=None, optim_dict=None):
        model_is_cuda = next(model.parameters()).is_cuda
        model = model.module if model_is_cuda else model
        package = {
            'args': model.args,
            state_dict: model.state_dict()            
        }
        if optimizer is not None:
            package[optim_dict] = optimizer.state_dict()
        return package
        
    @staticmethod    
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params
        
    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features  
    

class VGG(ModelBase):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


class GramMatrix(nn.Module):
    def forward(self,input):
        b, c, h, w = input.size()
        f = input.view(b,c,h*w) # bxcx(hxw)
        # torch.bmm(batch1, batch2, out=None)   #
        # batch1: bxmxp, batch2: bxpxn -> bxmxn #
        G = torch.bmm(f,f.transpose(1,2)) # f: bxcx(hxw), f.transpose: bx(hxw)xc -> bxcxc
        return G.div_(h*w)

class styleLoss(nn.Module):
    def forward(self,input,target):
        GramInput = GramMatrix()(input)
        return nn.MSELoss()(GramInput,target)

class BNMatching(nn.Module):
    # A style loss by aligning the BN statistics (mean and standard deviation)
    # of two feature maps between two images. Details can be found in
    # https://arxiv.org/abs/1701.01036
    def FeatureMean(self,input):
        b,c,h,w = input.size()
        f = input.view(b,c,h*w) # bxcx(hxw)
        return torch.mean(f,dim=2)
    def FeatureStd(self,input):
        b,c,h,w = input.size()
        f = input.view(b,c,h*w) # bxcx(hxw)
        return torch.std(f, dim=2)
    def forward(self,input,target):
        # input: 1 x c x H x W
        mu_input = self.FeatureMean(input)
        mu_target = self.FeatureMean(target)
        std_input = self.FeatureStd(input)
        std_target = self.FeatureStd(target)
        return nn.MSELoss()(mu_input,mu_target) + nn.MSELoss()(std_input,std_target)
        
                            
class FCLayer(nn.Module):
    def __init__(self, input_size, output_size, act_func=nn.ReLU, batch_norm=True, dropout = 0.0, bias=True):
        super(FCLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(input_size, output_size, bias=bias)
        self.act_func = act_func
        self.batch_norm = nn.BatchNorm1d(input_size) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.bias = bias

    def forward(self, x):
        if len(x.size()) > 2:
            t, n, z = x.size(0), x.size(1), x.size(2)   # (seq_len, batch, input_size)
            if not x.is_contiguous():
                x = x.contiguous()
            x = x.view(t * n, -1)          # (seq_len * batch, input_size)

        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.fc(x)
        if self.act_func is not None:
            x = self.act_func(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

        
class Con2dLayer(nn.Module):
    def __init__(self, nn_type,in_channels,out_channels,kernel_size,stride=1,padding=0,act_func=nn.ReLU,batch_norm=False,dropout=0.0,bias=True):
        super(Con2dLayer, self).__init__()
        if nn_type == 'Con2d':
            self.con2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
                                   padding=padding, bias=bias)
        elif nn_type == 'Con2dTrans':
            self.con2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, 
                                            padding=padding, bias=bias)
        else:
            raise Exception("wrong nn_type {}".format(nn_type))
        self.act_func = act_func
        self.batch_norm = nn.BatchNorm2d(in_channels) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):

        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.con2d(x)
        if self.act_func is not None:
            x = self.act_func(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
        
        
class AeGenerator(ModelBase):

    def __init__(self, args):
        super(AeGenerator, self).__init__()
        self.args = args
        self.debug = args.debug
        self.skip_out = args.g_skip_out
        self.act_type = supported_acts[args.g_act_type]
        self.batch_norm = args.g_bn
        self.drop = args.g_drop
        
        self.context_width = 2 * args.context_width + 1
        self.channels = args.delta_order + 1  
        self.num_features = args.num_features
        self.features = self.channels * self.context_width * self.num_features
        
        self.kernel_size = 3
        self.enc_channels = [16, 32, 64, 128]
        self.dec_channels = [128, 64, 32, 16]         
        self.enc_strides = [(2,1), (2,1), (2,2), (1,1)]
        self.dec_strides = [(1,1), (2,2), (2,1), (2,1)]
        self.enc_paddings = [(0,0), (0,0), (0,0), (0,0)]
        self.dec_paddings = [(0,0), (0,0), (0,0), (0,0)]
        
        enc_layers = []    
        enc_layer = Con2dLayer('Con2d', self.channels, self.enc_channels[0],
                               kernel_size=self.kernel_size, stride=self.enc_strides[0], 
                               padding=self.enc_paddings[0], act_func=self.act_type, batch_norm=self.batch_norm)
        enc_layers.append(enc_layer)        
        for i in range(1, len(self.enc_channels)): 
            enc_layer = Con2dLayer('Con2d', self.enc_channels[i-1], self.enc_channels[i],
                                   kernel_size=self.kernel_size, stride=self.enc_strides[i], 
                                   padding=self.enc_paddings[i], act_func=self.act_type, batch_norm=self.batch_norm)
            enc_layers.append(enc_layer)
        self.encoder = nn.ModuleList(enc_layers)
        
        dec_layers = []        
        dec_layer = Con2dLayer('Con2dTrans',self.dec_channels[0], self.dec_channels[1],
                               kernel_size=self.kernel_size, stride=self.dec_strides[0], 
                               padding=self.dec_paddings[0], act_func=self.act_type, batch_norm=self.batch_norm)   
        dec_layers.append(dec_layer)     
        for i in range(1, len(self.dec_channels)-1): 
            in_channel = self.dec_channels[i]*2 if self.skip_out else self.dec_channels[i]
            dec_layer = Con2dLayer('Con2dTrans', in_channel, self.dec_channels[i+1],
                                   kernel_size=self.kernel_size, stride=self.dec_strides[i], 
                                   padding=self.dec_paddings[i], act_func=self.act_type, batch_norm=self.batch_norm)
            dec_layers.append(dec_layer)
        in_channel = self.dec_channels[-1]*2 if self.skip_out else self.dec_channels[-1]
        dec_layer = Con2dLayer('Con2dTrans', in_channel, self.channels,
                               kernel_size=self.kernel_size, stride=self.dec_strides[-1], 
                               padding=self.dec_paddings[-1], act_func=self.act_type, batch_norm=self.batch_norm)  
        dec_layers.append(dec_layer)                       
        self.decoder = nn.ModuleList(dec_layers)        
        self.gen_dis1 = nn.Linear(self.features, self.features) 
            
    def forward(self, x):
        skips = []  
        x = x.view(-1, self.context_width, self.channels, self.num_features).transpose(1, 2).transpose(2, 3)
        if self.debug:
            print ('input size is {}'.format(x.size()))
        
        for i in range(0, len(self.enc_channels)-1):  
            x = self.encoder[i](x)
            if self.debug:
                print ('---Downconv layer {} size is {}, activation {} ---'.format(i, x.size(), self.act_type))
            if self.skip_out:
                skips.append(x) 
                if self.debug:
                    print ('Adding skip connection layer {}, size is {}'.format(i, x.size()))
                
        x = self.encoder[len(self.enc_channels)-1](x)
        if self.debug:
            print ('---Downconv layer {} size is {}, activation {} ---'.format(len(self.enc_channels)-1, x.size(), self.act_type))
            
        acoustic_input = x
        if self.debug:
            print ('--- acoustic_input size is {} ---'.format(x.size())) 
        
        for i in range(0, len(self.dec_channels)-1): 
            x = self.decoder[i](x)
            if self.debug:
                print ('Deconv layer {} size is {}, activation {}'.format(i, x.size(), self.act_type))
            if self.skip_out:
                skip_ = skips[-i-1]
                if self.debug:
                    print ('Fusing skip connection of shape is {}'.format(skip_.size()))
                x = torch.cat((x,skip_), 1)
            
        x = self.decoder[len(self.dec_channels)-1](x)
        if self.debug:
            print ('---Deconv layer {} size is {}, activation {}'.format(len(self.dec_channels)-1, x.size(), self.act_type))
            
        x = x.view(-1, self.num_flat_features(x))        
        x = self.gen_dis1(x)
        
        return x, acoustic_input      

                
class DnnGenerator(ModelBase):

    def __init__(self, args):
        super(DnnGenerator, self).__init__()
        self.args = args    
        self.act_type = supported_acts[args.g_act_type]
        self.dropout = args.g_drop
        self.num_layer = args.g_dnn_num_layer
        self.batch_norm = args.g_bn
        self.hidden_dim = args.g_dnn_hidden_dim
        self.bottleneck_dim = args.g_dnn_bottleneck_dim
        self.channels = args.delta_order + 1  
        self.features = self.channels * (2 * args.context_width + 1) * args.num_features
               
        enc_layers = []
        fc = FCLayer(self.features, self.hidden_dim, self.act_type, batch_norm=False, dropout=0.0, bias=True)
        enc_layers.append(fc)
        for i in range(1, self.num_layer - 1): 
            fc = FCLayer(self.hidden_dim, self.hidden_dim, self.act_type, batch_norm=False, dropout=self.dropout, bias=True)       
            enc_layers.append(fc)

        fc = nn.Linear(self.hidden_dim, self.bottleneck_dim, bias=True)
        enc_layers.append(fc)
        self.encoder = nn.ModuleList(enc_layers)
        
        dec_layers = []
        fc = FCLayer(self.bottleneck_dim, self.hidden_dim, self.act_type, batch_norm=False, dropout=self.dropout, bias=True)      
        dec_layers.append(fc)
        in_size = self.hidden_dim * 2 if args.skip_out else self.hidden_dim
        for i in range(1, self.num_layer - 2):            
            fc = FCLayer(in_size, self.hidden_dim, self.act_type, batch_norm=False, dropout=self.dropout, bias=True)  
            dec_layers.append(fc)
        fc = FCLayer(in_size, self.features, None, batch_norm=False, dropout=self.dropout, bias=True)
        dec_layers.append(fc)
        self.decoder = nn.ModuleList(dec_layers)

    def forward(self, x):
        if len(x.size()) == 4:
            x = x.view(-1, self.num_flat_features(x))
        skips = []
         
        for i in xrange(0, self.num_layer - 1):  
            x = self.encoder[i](x)
            if self.skip_out:
                skips.append(x) 
                
        x = self.encoder[self.num_layer - 1](x)
        
        acoustic_input = x
                
        for i in xrange(0, self.num_layer - 1): 
            x = self.decoder[i](x)
            if self.skip_out:
                skip_ = skips[-i]
                x = torch.cat((x,skip_), 1)
            
        x = self.decoder[self.num_layer - 1](x)
        
        return x, acoustic_input
      

class DnnDiscriminator(ModelBase):
    def __init__(self, args):
        super(DnnDiscriminator, self).__init__()
        self.args = args
        self.channels = args.delta_order + 1 
        self.features = self.channels * (2 * args.context_width + 1) * args.num_features
        self.hidden_dim = args.d_dnn_hidden_dim
        self.act_type = supported_acts[args.d_act_type]
        self.num_layer = args.d_dnn_num_layer
        self.batch_norm = args.d_bn
        self.dropout = args.d_drop
        
        layers = []
        fc = FCLayer(self.features, self.hidden_dim, self.act_type, batch_norm=False, dropout=self.dropout, bias=True)
        layers.append(('d_0', fc))
        for i in range(1, self.num_layer - 1):
            fc = FCLayer(self.hidden_dim, self.hidden_dim, self.act_type, batch_norm=self.batch_norm, dropout=self.dropout, bias=True)
            layers.append(('d_%d' % i, fc))
        fc = FCLayer(self.hidden_dim, 1, act_func=None, batch_norm=False, dropout=0.0, bias=True)
        layers.append(('d_%d' % (self.num_layer - 1), fc))
        
        self.dis = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        if len(x.size()) == 4:
            x = x.view(-1, self.num_flat_features(x))
        x = self.dis(x)
        return x       
                

class DnnLabelNet(ModelBase):
    def __init__(self, args):
        super(DnnLabelNet, self).__init__()
        self.args = args
        self.in_dim = args.label_dnn_in_dim
        self.hidden_dim = args.label_dnn_hidden_dim
        self.out_dim = args.num_classes
        self.act_type = supported_acts[args.label_act_type]
        self.num_layer = args.label_dnn_num_layer
        self.batch_norm = args.label_bn
        self.dropout = args.label_drop
        
        layers = []
        fc = FCLayer(self.in_dim, self.hidden_dim, self.act_type, batch_norm=False, dropout=self.dropout, bias=True)
        layers.append(('Label_0', fc))
        for i in range(1, self.num_layer - 1):
            fc = FCLayer(self.hidden_dim, self.hidden_dim, self.act_type, batch_norm=self.batch_norm, dropout=self.dropout, bias=True)
            layers.append(('Label_%d' % i, fc))
        fc = FCLayer(self.hidden_dim, self.out_dim, act_func=None, batch_norm=False, dropout=0.0, bias=True)
        layers.append(('Label_%d' % (self.num_layer - 1), fc))
        
        self.LabelNet = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        if len(x.size()) == 4:
            x = x.view(-1, self.num_flat_features(x))
        x = self.LabelNet(x)
        return x       
        
        
class CnnDiscriminator(ModelBase):

    def __init__(self, args):
        super(CnnDiscriminator, self).__init__()
        self.args = args
        self.debug = args.debug
        self.act_type = supported_acts[args.d_act_type]
        self.batch_norm = args.d_bn
        self.drop = args.d_drop
        
        self.context_width = 2 * args.context_width + 1
        self.channels = args.delta_order + 1  
        self.num_features = args.num_features
        self.features = self.channels * self.context_width * self.num_features

        self.kernel_size = 3
        self.num_fmaps = [16, 32, 64, 128]                       
        self.strides = [(2,1), (2,1), (2,1), (2,2)]        
        self.paddings = [(0,0), (0,0), (0,0), (0,0)]
        
        layers = []
        for i in range(len(self.num_fmaps)):
            in_channel = self.channels if i == 0 else self.num_fmaps[i-1]
            layer = Con2dLayer('Con2d', in_channel, self.num_fmaps[i],
                               kernel_size=self.kernel_size, stride=self.strides[i], 
                               padding=self.paddings[i], act_func=self.act_type, batch_norm=self.batch_norm)
            layers.append(('cnn_d_%d' % i, layer))
            
        self.dis = nn.Sequential(OrderedDict(layers))          
        self.dense = nn.Linear(3072, 1)
        
    def forward(self, x):
        if len(x.size()) == 2:
            x = x.view(-1, self.context_width, self.channels, self.num_features).transpose(1, 2).transpose(2, 3) 
        if self.debug:   
            print ('CnnDiscriminator input size is {}'.format(x.size()))
                
        x = self.dis(x)
        if self.debug:   
            print ('after cnn size is {}'.format(x.size()))                
        x = x.view(-1, self.num_flat_features(x))      
        d_logit_out = self.dense(x)
                    
        return d_logit_out

                
class CnnLabelNet(ModelBase):

    def __init__(self, args):
        super(CnnLabelNet, self).__init__()
        self.debug = args.debug
        self.act_type = supported_acts[args.label_act_type]
        self.batch_norm = args.label_bn
        self.drop = args.label_drop
        
        self.context_width = 2 * args.context_width + 1
        self.channels = args.delta_order + 1  
        self.num_features = args.num_features
        self.features = self.channels * self.context_width * self.num_features

        self.kernel_size = 3
        self.num_fmaps = [16, 32, 64, 128]                       
        self.strides = [(2,1), (2,1), (2,1), (2,2)]        
        self.paddings = [(0,0), (0,0), (0,0), (0,0)]
        
        layers = []
        for i in range(len(self.num_fmaps)):
            in_channel = self.channels if i == 0 else self.num_fmaps[i-1]
            layer = Con2dLayer('Con2d', in_channel, self.num_fmaps[i],
                               kernel_size=self.kernel_size, stride=self.strides[i], 
                               padding=self.paddings[i], act_func=self.act_type, batch_norm=self.batch_norm)
            layers.append(('cnn_label_%d' % i, layer))
            
        self.cnn_layer = nn.Sequential(OrderedDict(layers))                   
        self.label_dense = nn.Linear(3072, args.num_classes)
        
    def forward(self, disc_noise_std, x):
        if len(x.size()) == 2:
            x = x.view(-1, self.context_width, self.channels, self.num_features).transpose(1, 2).transpose(2, 3) 
        if self.debug:   
            print ('CnnLabelNet input size is {}'.format(x.size()))
                
        x = self.dis(x)
        if self.debug:   
            print ('after cnn size is {}'.format(x.size()))          
        x = x.view(-1, self.num_flat_features(x))
        x = self.label_dense(x)
        
        return x
    
import time
import os
import shutil
import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

from data.data_loader import GanFrameDataLoader, GanFrameBatchGenerator, GanSequentialDataset
from data.data_loader import FrameDataLoader, FrameBatchGenerator, SequentialDataset
from data.data_loader import SequentialDataLoader
from utils.file_logger import FileLogger

def save_checkpoint(state, save_path, is_best=False, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename),
                        os.path.join(save_path, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    pred = output.data.max(1)[1]  # get the index of the max log-probability
    batch_size = target.size(0)
    correct = pred.eq(target.data).cpu().sum()
    correct *= (100.0 / batch_size)
    return correct         


class Trainer(object):

    def __init__(self, args, name='BaseModel'):
        self.args = args
        self.name = name
        
        self.epochs = args.epochs
        self.curr_epoch = 0
        self.start_epoch = 0
        self.log_dir = args.log_dir               
        self.batch_size = args.batch_size
        self.num_batch_train = 0
        self.num_val = args.num_val
        self.print_freq = args.print_freq
        self.validate_freq = args.validate_freq
              
        self.best_prec = args.best_prec
        self.dev_acc_prev = args.dev_acc
        self.dev_loss_prev = args.dev_loss
        self.learning_rate = args.lr 
        self.start_halving_inc = 0.5
        self.end_halving_inc = 0.1
        self.start_halving_impr = 0.01
        self.end_halving_impr = 0.001  
        self.lr_reduce_factor = 0.9
        self.min_acc_impr = 0.5
        self.perform_diff_rate_thresh = 0.08
        
    def half_learning_rate(self):
        """Half the learning rate"""
        for optimizer in self.optimizer_list:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate * 0.5
        return self.learning_rate * 0.5
    
    def adjust_learning_rate_by_factor(self, factor):
        """Adjusts the learning rate according to the given factor"""
        self.learning_rate = self.learning_rate * factor
        for optimizer in self.optimizer_list:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        return self.learning_rate

    def set_learning_rate(self, learning_rate):
        """Sets the learning rate to the given learning_rate"""
        for optimizer in self.optimizer_list:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            
    def train(self):
        raise NotImplementedError

    def validate(self):

        losses = AverageMeter()
        acc = AverageMeter()
    
        # switch to evaluate mode
        self.generator_para.eval()
        self.discriminator_para.eval()
        self.labelnet_para.eval()

        for i, (data) in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader)):
            data_generator = FrameBatchGenerator(data, self.audio_conf)
            while True:
                noise_inputs, targets = data_generator.next_batch(self.batch_size)
                if noise_inputs is None or targets is None:
                    break
            
                input_var = Variable(noise_inputs).cuda()
                org_var = Variable(org_inputs).cuda()
                target_var = Variable(targets).cuda()
                                
                # compute output                
                _, acoustic_input = self.generator_para(input_var)
                acoustic_output = self.labelnet_para(acoustic_input)
                label_loss = self.criterion_label(acoustic_output, target_var) / noise_inputs.size(0)

                correct_rate = accuracy(acoustic_output, target_var)  # measure accuracy
                losses.update(label_loss.data[0], noise_inputs.size(0))
                acc.update(correct_rate, noise_inputs.size(0))

        print('>> Validate: avg_accuracy = {0:.4f}, avg_loss = {1:.4f}'.format(acc.avg, losses.avg))
        return losses.avg, acc.avg
            
    
class GanFrameTrainer(Trainer):
    """ Speech Enhancement Generative Adversarial Network """
    def __init__(self, args, audio_conf, model_list, criterion_list, optimizer_list):
        
        self.generator = model_list[0] 
        self.generator_para = torch.nn.DataParallel(self.generator).cuda() 
        self.discriminator = model_list[1]
        self.discriminator_para = torch.nn.DataParallel(self.discriminator).cuda() 
        self.labelnet = model_list[2]
        self.labelnet_para = torch.nn.DataParallel(self.labelnet).cuda() 
                          
        self.criterion_label = criterion_list[0]
        self.criterion_mse = criterion_list[1]
        
        self.optimizer_list = optimizer_list
        self.gen_optimizer = optimizer_list[0]
        self.dis_optimizer = optimizer_list[1]
        self.label_optimizer = optimizer_list[2]

        super(GanFrameTrainer, self).__init__(args)
        
        self.audio_conf = audio_conf
        self.pair = self.audio_conf.get("pair", False)
        
        self.train_dataset = GanSequentialDataset(data_scp=os.path.join(args.train_scp, 'feats.scp'),
                                                  org_data_scp=os.path.join(args.train_scp, 'org_feats.scp'),
                                                  label_file=os.path.join(args.train_scp, 'labels.frame.gz'),
                                                  audio_conf=self.audio_conf)
        self.train_dataloader = GanFrameDataLoader(self.train_dataset,
                                                   batch_size=args.num_utt_per_loading,
                                                   num_workers=args.num_workers,
                                                   shuffle=True, pin_memory=False)
        self.val_dataset = SequentialDataset(data_scp=os.path.join(args.val_scp, 'feats.scp'),
                                             label_file=os.path.join(args.val_scp, 'labels.frame.gz'),
                                             audio_conf=self.audio_conf)
        self.val_dataloader = FrameDataLoader(self.val_dataset,
                                              batch_size=args.num_utt_per_loading,
                                              num_workers=args.num_workers,
                                              shuffle=False, pin_memory=False)

        self.curr_alpha = args.alpha
        self.deactivated_alpha = False
        self.g_steps = args.g_steps
        self.d_steps = args.d_steps
    
    def train(self):              
        
        header = ['curr_epoch', 'curr_val', 'train_cost', 'train_acc', 'val_cost',
                  'val_acc', 'learning_rate', 'alpha']
        file_logger = FileLogger(os.path.join(self.log_dir, "out.tsv"), header) 

        epoch_train_losses = AverageMeter()
        epoch_train_acc = AverageMeter()
        epoch_dev_losses = AverageMeter()
        epoch_dev_acc = AverageMeter()
        
        label_losses = AverageMeter()
        g_adv_losses = AverageMeter()
        d_losses = AverageMeter()
        acc = AverageMeter()
        
        noise_label_value = 0.
        org_label_value = 1.    
    
        noise_label = torch.FloatTensor(self.batch_size).cuda()
        org_label = torch.FloatTensor(self.batch_size).cuda()

        for epoch in range(self.start_epoch, self.epochs):
            self.curr_epoch = epoch
            num_batch = 0
            if self.curr_epoch >= self.args.dealpha_epoch and not self.deactivated_alpha:
                decay = self.args.dealpha_decay
                new_alpha = decay * self.curr_alpha
                if new_alpha < self.args.dealpha_bound:
                    print('New alpha {} < lbound {}, setting 0.'.format(new_alpha,
                                                                        self.args.dealpha_bound))
                    # it it's lower than a lower bound, cancel out completely
                    new_alpha = self.args.dealpha_bound
                    self.deactivated_alpha = True
                else:
                    print('Applying decay {} to alpha {}: {}'.format(decay, self.curr_alpha,
                                                                     new_alpha))
                self.curr_alpha = new_alpha

            epoch_train_losses.reset()
            epoch_train_acc.reset()
            epoch_dev_losses.reset()
            epoch_dev_acc.reset()
            
            label_losses.reset()
            g_adv_losses.reset()
            d_losses.reset()
            acc.reset()

            one_epoch_start = time.time()
            self.generator_para.train()
            self.discriminator_para.train()
            self.labelnet_para.train()
            if not self.pair:
                self.train_dataset.re_disorder()
                self.train_dataloader = GanFrameDataLoader(self.train_dataset,
                                                           batch_size=self.args.num_utt_per_loading,
                                                           num_workers=self.args.num_workers,
                                                           shuffle=True, pin_memory=False)

            for i, (data) in enumerate(self.train_dataloader, start=0):
                data_generator = GanFrameBatchGenerator(data, self.audio_conf)
                while True:
                    noise_inputs, org_inputs, targets = data_generator.next_batch(self.args.batch_size)
                    if noise_inputs is None or targets is None:
                        break

                    self.num_batch_train += 1
                    num_batch += 1

                    input_var = Variable(noise_inputs).cuda()
                    org_var = Variable(org_inputs).cuda()
                    target_var = Variable(targets).cuda()
                
                    # Update D network #
                    self.gen_optimizer.zero_grad()
                    self.dis_optimizer.zero_grad()
                    self.label_optimizer.zero_grad()
                    gen_output, _ = self.generator_para(input_var)
                    ##gen_output = (gen_output + self.mean) * self.var

                    d_rl_logic = self.discriminator_para(org_var)
                    d_fk_logic = self.discriminator_para(gen_output)
                                                  
                    noise_label.resize_as_(d_fk_logic.data).fill_(noise_label_value)
                    noise_label_variable = Variable(noise_label)
                    d_fk_loss = self.criterion_mse(d_fk_logic, noise_label_variable)
                    d_fk_loss.backward()
                
                    org_label.resize_as_(d_rl_logic.data).fill_(org_label_value)
                    org_label_variable = Variable(org_label)
                    d_rl_loss = self.criterion_mse(d_rl_logic, org_label_variable)
                    d_rl_loss.backward()
                
                    d_loss = (d_rl_loss + d_fk_loss) / noise_inputs.size(0)
                    loss_sum = d_loss.data.sum()
                    inf = float("inf")
                    if loss_sum == inf or loss_sum == -inf:
                        print("WARNING: received an inf loss, setting loss value to 0")
                        continue
                    else:
                        loss_value = d_loss.data[0]
                    d_losses.update(loss_value, noise_inputs.size(0))
                    if loss_value > (1.0 - self.perform_diff_rate_thresh) * d_losses.avg:
                        self.perform_diff_rate_thresh = max(
                                  self.perform_diff_rate_thresh * 0.8, 0.0001)
                        self.set_learning_rate(0.5 * self.learning_rate)
                        self.dis_optimizer.step()
                        self.set_learning_rate(self.learning_rate)
                    else:
                        self.dis_optimizer.step()
                
                    # Update G network ###
                    self.gen_optimizer.zero_grad()
                    self.dis_optimizer.zero_grad()
                    self.label_optimizer.zero_grad()
                    gen_output, acoustic_input = self.generator_para(input_var)
                    ##gen_output = (gen_output + self.mean) * self.var

                    d_fk_logic = self.discriminator_para(gen_output)
                    g_adv_loss = self.criterion_mse(d_fk_logic, org_label_variable) / noise_inputs.size(0)
                    g_adv_loss *= self.curr_alpha

                    acoustic_output = self.labelnet_para(acoustic_input)
                    label_loss = self.criterion_label(acoustic_output, target_var) / noise_inputs.size(0)
                    g_loss = g_adv_loss + label_loss
                    g_loss.backward()
                    loss_sum = g_adv_loss.data.sum()
                    inf = float("inf")
                    if loss_sum == inf or loss_sum == -inf:
                        print("WARNING: received an inf loss, setting loss value to 0")
                        continue
                    else:
                        loss_value = g_adv_loss.data[0]
                    g_adv_losses.update(loss_value, noise_inputs.size(0))
                    if loss_value > (1.0 - self.perform_diff_rate_thresh) * g_adv_losses.avg:
                        self.perform_diff_rate_thresh = max(
                                        self.perform_diff_rate_thresh * 0.8, 0.0001)
                        self.set_learning_rate(0.5 * self.learning_rate)
                        self.gen_optimizer.step()
                        self.set_learning_rate(self.learning_rate)
                    else:
                        self.gen_optimizer.step()

                    # Update C network ###
                    self.gen_optimizer.zero_grad()
                    self.dis_optimizer.zero_grad()
                    self.label_optimizer.zero_grad()
                    _, acoustic_input = self.generator_para(input_var)
                    acoustic_output = self.labelnet_para(acoustic_input)
                    label_loss = self.criterion_label(acoustic_output, target_var) / noise_inputs.size(0)
                    label_loss.backward()
                    loss_sum = label_loss.data.sum()
                    inf = float("inf")
                    if loss_sum == inf or loss_sum == -inf:
                        print("WARNING: received an inf loss, setting loss value to 0")
                        continue
                    else:
                        loss_value = label_loss.data[0]
                    if loss_value > (1.0 - self.perform_diff_rate_thresh) * label_losses.avg:
                        self.perform_diff_rate_thresh = max(
                                        self.perform_diff_rate_thresh * 0.8, 0.0001)
                        self.set_learning_rate(0.5 * self.learning_rate)
                        self.label_optimizer.step()
                        self.set_learning_rate(self.learning_rate)
                    else:
                        self.label_optimizer.step()

                    correct_rate = accuracy(acoustic_output, target_var)
                    acc.update(correct_rate, noise_inputs.size(0))
                    label_losses.update(loss_value, noise_inputs.size(0))

                    if num_batch % self.print_freq == 0:
                        print('Train: Epoch[{0}]: (lr = {1:.6f}) Batch: [{2}]\t'
                              'Label_Loss {label_losses.val:.4f} ({label_losses.avg:.4f})\t'
                              'G_adv_Loss {g_adv_losses.val:.4f} ({g_adv_losses.avg:.4f})\t'
                              'D_Loss {d_losses.val:.4f} ({d_losses.avg:.4f})\t'
                              'Prec@1 {acc.val:.3f} ({acc.avg:.3f})'
                              .format(epoch, self.learning_rate, num_batch,
                                      label_losses=label_losses, g_adv_losses=g_adv_losses,
                                      d_losses=d_losses, acc=acc))
            
                    if self.num_batch_train % self.validate_freq == 0:
                        self.num_val += 1

                        train_loss, train_acc = (label_losses.avg, acc.avg)
                        dev_loss, dev_acc = self.validate()
                        
                        self.generator_para.train()
                        self.discriminator_para.train()
                        self.labelnet_para.train()
            
                        epoch_dev_acc.update(dev_acc)
                        epoch_dev_losses.update(dev_loss)
                        epoch_train_acc.update(train_acc)
                        epoch_train_losses.update(train_loss)
                    
                        file_logger.write([epoch, self.num_val, train_loss, train_acc, dev_loss,
                                           dev_acc, self.learning_rate, self.curr_alpha])
                
                        if dev_acc < self.dev_acc_prev + self.min_acc_impr:
                            self.min_acc_impr = max(0.5 * self.min_acc_impr, 0.001)
                            self.learning_rate = self.adjust_learning_rate_by_factor(self.lr_reduce_factor)
                            self.learning_rate = max(self.learning_rate, 0.000005)
                            print ('learning_rate has been set to {}'.format(self.learning_rate))

                        self.dev_acc_prev = dev_acc

                        # remember best prec@1 and save checkpoint
                        is_best = dev_acc > self.best_prec
                        self.best_prec = max(dev_acc, self.best_prec)
                        
                        gen_state = self.generator.serialize(self.generator_para, 'gen_state_dict', self.gen_optimizer, 'gen_optim_dict')
                        state = gen_state
                        dis_state = self.discriminator.serialize(self.discriminator_para, 'dis_state_dict', 
                                                                 self.dis_optimizer, 'dis_optim_dict')
                        state.update(dis_state)
                        label_state = self.labelnet.serialize(self.labelnet_para, 'label_state_dict', 
                                                              self.label_optimizer, 'label_optim_dict')
                        state.update(label_state)                        
                        other_state = {'epoch': epoch + 1,
                                       'num_val': self.num_val,
                                       'dev_acc': dev_acc,
                                       'dev_loss': dev_loss,
                                       'alpha': self.curr_alpha,
                                       'best_prec': self.best_prec,
                                       'learning_rate': self.learning_rate,
                                       'audio_conf': self.audio_conf}
                        state.update(other_state) 
                        save_checkpoint(state, self.log_dir, filename='epoch_{}_{}.pth'.format(epoch, self.num_val))
            
            one_epoch_end = time.time()
            print('######################################################')
            print('Train: Epoch: [{0}]\t'
                  'Time {1}\t'
                  'train_Loss {2:.4f}\t'
                  'train_acc {3:.4f}\t'
                  'dev_Loss {4:.4f}\t'
                  'dev_acc {5:.4f}\t'.format(epoch, one_epoch_end - one_epoch_start,
                                         epoch_train_losses.avg, epoch_train_acc.avg,
                                         epoch_dev_losses.avg, epoch_dev_acc.avg))
        
        print('Finished training with {0} epochs'.format(self.curr_epoch))
        
                              
class CeFrameTrainer(Trainer):
    """ Speech Enhancement Generative Adversarial Network """
    def __init__(self, args, audio_conf, model_list, criterion_list, optimizer_list, name='CETrainer'):
        self.generator = model_list[0] 
        self.generator_para = torch.nn.DataParallel(self.generator).cuda() 
        self.labelnet = model_list[2]
        self.labelnet_para = torch.nn.DataParallel(self.labelnet).cuda() 
                          
        self.criterion_label = criterion_list[0]
        
        self.optimizer_list = optimizer_list
        self.gen_optimizer = optimizer_list[0]
        self.label_optimizer = optimizer_list[2]

        super(CeFrameTrainer, self).__init__(args)
        
        self.audio_conf = audio_conf
        

        self.train_dataset = SequentialDataset(data_scp=os.path.join(args.train_scp, 'feats.scp'),
                                               label_file=os.path.join(args.train_scp, 'labels.frame.gz'),
                                               audio_conf=self.audio_conf)
        self.train_dataloader = FrameDataLoader(self.train_dataset,
                                                batch_size=args.num_utt_per_loading,
                                                num_workers=args.num_workers,
                                                shuffle=True, pin_memory=False)
        self.val_dataset = SequentialDataset(data_scp=os.path.join(args.val_scp, 'feats.scp'),
                                             label_file=os.path.join(args.val_scp, 'labels.frame.gz'),
                                             audio_conf=self.audio_conf)
        self.val_dataloader = FrameDataLoader(self.val_dataset,
                                              batch_size=args.num_utt_per_loading,
                                              num_workers=args.num_workers,
                                              shuffle=False, pin_memory=False)

    def train(self):              
       
        header = ['curr_epoch', 'curr_val', 'train_cost', 'train_acc', 'val_cost', 'val_acc']
        file_logger = FileLogger(os.path.join(self.log_dir, "out.tsv"), header) 
        
        epoch_train_losses = AverageMeter()
        epoch_train_acc = AverageMeter()
        epoch_dev_losses = AverageMeter()
        epoch_dev_acc = AverageMeter()
        
        label_losses = AverageMeter()
        acc = AverageMeter()
                             
        for epoch in range(self.start_epoch, self.epochs):
            self.curr_epoch = epoch
            
            epoch_train_losses.reset()
            epoch_train_acc.reset()
            epoch_dev_losses.reset()
            epoch_dev_acc.reset()
            label_losses.reset()
            acc.reset()

            one_epoch_start = time.time()

            self.generator_para.train()
            self.labelnet_para.train()        
            num_batch = 0
            for i, (data) in enumerate(self.train_dataloader, start=0):
                data_generator = FrameBatchGenerator(data)
                while True:
                    inputs, targets = data_generator.next_batch(self.args.batch_size)
                    if inputs is None or targets is None:
                        break

                    self.num_batch_train += 1
                    num_batch += 1

                    input_var = Variable(inputs).cuda()
                    target_var = Variable(targets).cuda(async=True)
                    
                    self.label_dis_optimizer.zero_grad()
                    self.gen_optimizer.zero_grad()
                    _, acoustic_input = self.generator_para(input_var)
                    acoustic_output = self.labelnet_para(acoustic_input)
                    label_loss = self.criterion_label(acoustic_output, target_var) / inputs.size(0)
                    label_loss.backward()
                    self.label_optimizer.step()
                    self.gen_optimizer.step()
                    
                    # measure accuracy and record loss
                    accuracy_rate = accuracy(acoustic_output, target_var)
                    acc.update(accuracy_rate, inputs.size(0))
                    label_losses.update(label_loss.data[0], inputs.size(0))
                                        
                    if num_batch % self.args.print_freq == 0:
                        print('Train: Epoch[{0}]: (lr = {1:.4f}) Batch: [{2}]\t'
                              'Label_Loss {label_losses.val:.4f} ({label_losses.avg:.4f})\t'
                              'Prec@1 {acc.val:.3f} ({acc.avg:.3f})'
                              .format(epoch, self.learning_rate, num_batch,
                                      label_losses=label_losses, acc=acc))
            
                    if self.num_batch_train % self.validate_freq == 0:
                        self.num_val += 1

                        train_loss, train_acc = (label_losses.avg, acc.avg)
                        dev_loss, dev_acc = self.validate()
                        
                        self.generator_para.train()
                        self.labelnet_para.train()       
            
                        epoch_dev_acc.update(dev_acc)
                        epoch_dev_losses.update(dev_loss)
                        epoch_train_acc.update(train_acc)
                        epoch_train_losses.update(train_loss)
                    
                        file_logger.write([epoch, self.num_val, train_loss, train_acc,
                                           dev_loss, dev_acc])
                
                        if dev_acc < self.dev_acc_prev + self.min_acc_impr:
                            self.min_acc_impr = max(0.5 * self.min_acc_impr, 0.001)
                            self.learning_rate = self.adjust_learning_rate_by_factor(self.lr_reduce_factor)
                            self.learning_rate = max(self.learning_rate, 0.000005)
                            print ('learning_rate has been set to {}'.format(self.learning_rate))

                        self.dev_acc_prev = dev_acc

                        # remember best prec@1 and save checkpoint
                        is_best = dev_acc > self.best_prec
                        self.best_prec = max(dev_acc, self.best_prec)

                        gen_state = self.generator.serialize(self.generator_para, 'gen_state_dict', self.gen_optimizer, 'gen_optim_dict')
                        state = gen_state
                        label_state = self.labelnet.serialize(self.labelnet_para, 'label_state_dict', 
                                                              self.label_optimizer, 'label_optim_dict')
                        state.update(label_state)                        
                        other_state = {'epoch': epoch + 1,
                                       'num_val': self.num_val,
                                       'dev_acc': dev_acc,
                                       'dev_loss': dev_loss,
                                       'best_prec': self.best_prec,
                                       'learning_rate': self.learning_rate,
                                       'audio_conf': self.audio_conf}
                        state.update(other_state) 
                        save_checkpoint(state, self.log_dir, filename='epoch_{}_{}.pth'.format(epoch, self.num_val))
            
                one_epoch_end = time.time()
                print('######################################################')
                print('Train: Epoch: [{0}]\t'
                      'Time {1}\t'
                      'train_Loss {2:.4f}\t'
                      'train_acc {3:.4f}\t'
                      'dev_Loss {4:.4f}\t'
                      'dev_acc {5:.4f}\t'.format(epoch, one_epoch_end - one_epoch_start,
                                                 epoch_train_losses.avg, epoch_train_acc.avg,
                                                 epoch_dev_losses.avg, epoch_dev_acc.avg))
        
        print('Finished training with {0} epochs'.format(self.curr_epoch))
        

class MseFrameTrainer(Trainer):
    """ Speech Enhancement Generative Adversarial Network """
    def __init__(self, args, audio_conf, model_list, criterion_list, optimizer_list):
        super(MseFrameTrainer, self).__init__(args, model_list, criterion_list, optimizer_list)
        self.audio_conf = audio_conf
        self.audio_conf['pair'] = True
        
        self.generator = model_list[0] 
        self.generator_para = torch.nn.DataParallel(self.generator).cuda()                                   
        self.criterion_mse = criterion_list[0]        
        self.optimizer_list = optimizer_list
        self.gen_optimizer = optimizer_list[0]
        
        self.train_dataset = GanSequentialDataset(data_scp=os.path.join(args.train_scp, 'feats.scp'),
                                                  org_data_scp=os.path.join(args.train_scp, 'org_feats.scp'),
                                                  label_file=os.path.join(args.train_scp, 'labels.frame.gz'),
                                                  audio_conf=self.audio_conf)
        self.train_dataloader = GanFrameDataLoader(self.train_dataset,
                                                   batch_size=args.num_utt_per_loading,
                                                   num_workers=args.num_workers,
                                                   shuffle=True, pin_memory=False)
        self.val_dataset = GanSequentialDataset(data_scp=os.path.join(args.val_scp, 'feats.scp'),
                                                org_data_scp=os.path.join(args.val_scp, 'org_feats.scp'),
                                                label_file=os.path.join(args.val_scp, 'labels.frame.gz'),
                                                audio_conf=self.audio_conf)
        self.val_dataloader = GanFrameDataLoader(self.val_dataset,
                                                 batch_size=args.num_utt_per_loading,
                                                 num_workers=args.num_workers,
                                                 shuffle=False, pin_memory=False)

    def train(self):
        
        header = ['curr_epoch', 'train_cost', 'val_cost', 'time']
        file_logger = FileLogger(os.path.join(self.log_dir, "out.tsv"), header)

        mse_losses = AverageMeter()

        epoch_train_losses = AverageMeter()
        epoch_dev_losses = AverageMeter()

        for epoch in range(self.start_epoch, self.epochs):
            one_epoch_start = time.time()
            self.curr_epoch = epoch
            num_batch = 0
            mse_losses.reset()
            epoch_train_losses.reset()
            epoch_dev_losses.reset()
            
            self.generator_para.train()
            
            for i, (data) in enumerate(self.train_dataloader, start=0):
                data_generator = GanFrameBatchGenerator(data, self.audio_conf)
                while True:
                    noise_inputs, org_inputs, targets = data_generator.next_batch(self.args.batch_size)
                    if noise_inputs is None or targets is None:
                        break

                    self.num_batch_train += 1
                    num_batch += 1

                    input_var = Variable(noise_inputs).cuda()
                    org_var = Variable(org_inputs).cuda()

                    self.gen_optimizer.zero_grad()
                    gen_output, _ = self.generator_para(input_var)
                    gen_output_2dim = gen_output.view(self.batch_size, -1)
                    mse_loss = self.criterion_mse(gen_output_2dim, org_var) / noise_inputs.size(0)
                    mse_losses.update(mse_loss)
                    mse_loss.backward()
                    self.gen_optimizer.step()

                    if num_batch % self.args.print_freq == 0:
                        print('Train: Epoch[{0}]: (lr = {1:.6f}) Batch: [{2}]\t'
                              'Mse_Loss {mse_losses.val:.4f} ({mse_losses.avg:.4f})\t'
                              .format(epoch, self.learning_rate, num_batch,
                                      mse_losses=mse_losses))

                    if self.num_batch_train % self.args.validate_freq == 0:
                        self.num_val += 1

                        train_loss = mse_losses.avg
                        dev_loss = self.validate()

                        epoch_dev_losses.update(dev_loss)
                        epoch_train_losses.update(train_loss)

                        file_logger.write([epoch, self.num_val, train_loss, dev_loss])
                        
                        rel_impr = (self.dev_loss_prev - dev_loss) / self.dev_loss_prev
                        self.dev_loss_prev = dev_loss
                        if rel_impr < self.start_halving_impr:
                            self.learning_rate = self.adjust_learning_rate_by_factor(self.lr_reduce_factor)
                            self.learning_rate = max(self.learning_rate, 0.000005)
                            self.set_learning_rate(self.learning_rate)
                            print ('learning_rate has been set to {}'.format(self.learning_rate))
                        
                        self.dev_loss_prev = dev_loss

                        gen_state = self.generator.serialize(self.generator_para, 'gen_state_dict', self.gen_optimizer, 'gen_optim_dict')
                        state = gen_state                                   
                        other_state = {'epoch': epoch + 1,
                                       'num_val': self.num_val,
                                       'dev_loss': dev_loss,
                                       'learning_rate': self.learning_rate,
                                       'audio_conf': self.audio_conf}
                        state.update(other_state) 
                        save_checkpoint(state, self.log_dir, filename='epoch_{}_{}.pth'.format(epoch, self.num_val))
            
                one_epoch_end = time.time()
                print('######################################################')
                print('Train: Epoch: [{0}]\t'
                      'Time {1}\t'
                      'train_Loss {2:.4f}\t'
                      'dev_Loss {3:.4f}\t'.format(epoch, one_epoch_end - one_epoch_start,
                                              epoch_train_losses.avg, epoch_dev_losses.avg))

            print('Finished training with {0} epochs'.format(self.curr_epoch))
            
    def validate(self):

        losses = AverageMeter()    
        # switch to evaluate mode
        self.generator_para.eval()
        
        for i, (data) in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader)):
            data_generator = GanFrameBatchGenerator(data, self.audio_conf)
            while True:
                noise_inputs, org_inputs, targets = data_generator.next_batch(self.args.batch_size)
                if noise_inputs is None or targets is None:
                    break

                input_var = Variable(noise_inputs).cuda()
                org_var = Variable(org_inputs).cuda()

                gen_output, _ = self.generator_para(input_var)
                gen_output_2dim = gen_output.view(self.batch_size, -1)
                mse_loss = self.criterion_mse(gen_output_2dim, org_var) / noise_inputs.size(0)
                losses.update(mse_loss)   
        print('>> Validate: avg_loss = {1:.4f}'.format(losses.avg))
        return losses.avg

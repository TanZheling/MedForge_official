import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_loops, get_testset, get_synset, get_dataset, get_network, get_eval_pool, evaluate_synset,  match_loss, get_time, \
    TensorDataset, epoch, DiffAugment, ParamDiffAug

import wandb
import copy
import random
from reparam_module import ReparamModule
import torch.utils.data
import warnings
import wandb
# import gc
import time
from glad_utils import *


from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, AutoConfig, Swinv2Config, Swinv2Model, CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPTokenizerFast, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection,AutoProcessor,AutoTokenizer,CLIPVisionModel
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
warnings.filterwarnings("ignore") 
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    # for param in model.named_parameters():
    #     # param.requires_grad = False
    #     print(param[0])
    for _, param in model.named_parameters():
        # print(_)
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    seed = args.seed
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run = wandb.init(sync_tensorboard=False,
                project="baseline_save_{}_seed".format(args.detail),
                name='{}_{}_epoch_eval_train={}lr{}bs{}sche_{}_seed{}'.format(args.dataset,args.model, args.Epoch, args.learningrate,args.batch_train,args.schedule,args.seed),
                    # name='test',
                config=args,
                )

    # testloader, channel, im_size, num_classes = get_testset(args.dataset, args.data_path, args=args)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv, dst_val, valloader = get_dataset(args.dataset, args.data_path, args.batch_real, args.res, args=args)
    # images, labels, lr = get_synset(args.syn_path)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_real, shuffle=True, num_workers=18)

    net, processor = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width, gene_dim=0,class_map=class_map, class_map_inv=class_map_inv)
    net = net.to(args.device)
    if args.head:
        for name,param in net.named_parameters():
            if name.startswith('classifier'):
                param.requires_grad = True
            else:
                param.requires_grad = False
        # for name,param in net.named_parameters():    
        #     print(name,param.requires_grad)
    print_trainable_parameters(net)

    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=0.0005)
    if args.schedule == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    if args.schedule == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.cos_T)
    loss_avg, acc_avg, num_exp = 0, 0, 0

    run_dir = "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), run.name)
    args.save_path = os.path.join(args.save_path, "baseline", run_dir)


    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    
    loss_test, acc_test, auc_test, _, _, _ = epoch('test', testloader, net, optimizer, criterion, args, aug=False,it=None, log=False,it_eval=None,model_eva=None,baseline=True,class_names=class_names,processor = processor)
    wandb.log({"Scratch Loss": loss_test})
    wandb.log({"Scratch ACC": acc_test})
    wandb.log({"Scratch AUC": auc_test})
    print("Scratch/loss", loss_test, "Scratch/acc", acc_test,"Scratch/auc", auc_test)
    
    for ep in range(args.Epoch):
        net.train()
        gtlabel_list=[]
        output_list=[]
        predict_list=[]
        check=[]
        # img_l=[]
        for i_batch, datum in enumerate(trainloader):
            img = datum[0].to(args.device)
            # img_l.append(img)
            lab = datum[1].to(args.device)
            # print("2:{}".format(torch.cuda.memory_allocated(0)))
            for x in lab.tolist():
                gtlabel_list.append(x)
            n_b = lab.shape[0]

            inputs = processor(text=class_names,padding=True, return_tensors="pt").to(args.device)
            output_ori = net(pixel_values=img,**inputs)
            logits_per_image = output_ori.logits_per_image # this is the image-text similarity score
            output = logits_per_image.softmax(dim=1)

            # output = net(img)   # whether need to add 'with torch.no_grad()' for test mode?
            if type(output)!=torch.Tensor:
                # print(type(output),output)
                output = output.logits

            loss = criterion(output, lab)

            predicted = torch.argmax(output.data, 1)
            for x in predicted.tolist():
                predict_list.append(x)
            for x in output.tolist():

                check.append(torch.tensor(x))

            correct = (predicted == lab).sum()
            # print(loss.item())
            loss_avg += loss.cpu().detach().numpy().item()*n_b
            acc_avg += correct.item()
            num_exp += n_b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if args.schedule != 'no':
            scheduler.step()

        
        loss_avg /= num_exp
        acc_avg /= num_exp
       
        print("loss", loss_avg, "acc", acc_avg)
        wandb.log({"LR":optimizer.param_groups[0]['lr']})
        # print("第%d个epoch的学习率：%f" % (ep, optimizer.param_groups[0]['lr']))

        if ep % 5 == 0:
            net.eval()
            with torch.no_grad():
                loss_test, acc_test, auc_test, _, _, _ = epoch('test', testloader, net, optimizer, criterion, args, aug=False,it=None, log=False,it_eval=None,model_eva=None,baseline=True,class_names=class_names,processor = processor)
                wandb.log({"Test Loss": loss_test})
                wandb.log({"Test ACC": acc_test})
                wandb.log({"Test AUC": auc_test})
                print("Test/loss", loss_test, "Test/acc", acc_test,"Test/auc", auc_test)
            save_state = {}
            print("Model's state_dict:")

            for param_tensor in net.state_dict():
                if 'lora' in param_tensor:
                    save_state.update({param_tensor:net.state_dict()[param_tensor]})

            torch.save(save_state,os.path.join(args.save_path, 'res_%s_%s_it%dipc_lora.pt'%(args.dataset, args.model, ep)))
            # net.save_pretrained(args.save_path_pro)
            # net.vision_model.save_pretrained(args.save_path_vis)
            
            # peft_model_id = args.save_path_pro
            # config = PeftConfig.from_pretrained(peft_model_id)
            # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            # model = PeftModel.from_pretrained(model, peft_model_id)
            # model = PeftModel.from_pretrained(model, args.save_path_vis)
            # print(model)
            
            print('save at '+os.path.join(args.save_path, 'res_%s_%s_it%dipc_lora.pt'%(args.dataset, args.model, ep)))
        # if ep in lr_schedule:
        #     optimizer = torch.optim.SGD(net.parameters(), lr=lr*0.1, momentum=0.9, weight_decay=0.0005)
            
    wandb.finish()

if __name__ == '__main__':
    import shared_args
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='M',
                        help='eval_mode')  # S: the same to training model, M: multi architectures
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--save_it', type=int, default=None, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')

    parser.add_argument('--mom_img', type=float, default=0.5, help='momentum for updating synthetic images')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_test', type=int, default=128, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='noise', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')

    parser.add_argument('--save_path', type=str, default='result', help='path to save results')

    parser.add_argument('--space', type=str, default='p', choices=['p', 'wp'])
    parser.add_argument('--res', type=int, default=128, choices=[128,224, 256, 512], help='resolution')
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--avg_w', action='store_true')

    parser.add_argument('--eval_all', action='store_true')

    parser.add_argument('--min_it', type=bool, default=False)
    parser.add_argument('--no_aug', type=bool, default=False)

    parser.add_argument('--force_save', action='store_true')

    parser.add_argument('--sg_batch', type=int, default=10)

    parser.add_argument('--rand_f', action='store_true')

    parser.add_argument('--logdir', type=str, default='./logged_files')

    parser.add_argument('--wait_eval', action='store_true')

    parser.add_argument('--idc_factor', type=int, default=1)

    parser.add_argument('--rand_gan_un', action='store_true')
    parser.add_argument('--rand_gan_con', action='store_true')

    parser.add_argument('--learn_g', action='store_true')

    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=5)


    parser.add_argument('--special_gan', default=None)

    parser.add_argument('--train_ann', type=str, default=None)
    parser.add_argument('--val_ann', type=str, default=None)
    parser.add_argument('--test_ann', type=str, default=None)

    parser.add_argument('--method', type=str, default='DC', help='Distillation Method')
    parser.add_argument('--valphase', action='store_true')

    parser.add_argument('--lr_img', type=float, default=1, help='learning rate for pixels or f_latents')
    parser.add_argument('--lr_w', type=float, default=0.001, help='learning rate for updating synthetic latent w')
    parser.add_argument('--lr_g', type=float, default=0.0001, help='learning rate for gan weights')

    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--inner_loop', type=float, default=1, help='inner loop')
    parser.add_argument('--outer_loop', type=float, default=1, help='outer loop')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--detail', type=str, default='', help='probable comments')
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--gene_csv', type=str, default='')
    parser.add_argument('--gene', action='store_true')
    parser.add_argument('--lr_gene',type=float, default=0.1)
    parser.add_argument('--logvalperformance', action='store_true')
    parser.add_argument('--noimg', action='store_true')
    parser.add_argument('--lora',action='store_true')
    parser.add_argument('--loglogits', action='store_true')
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--Epoch', type=int, default=100, help='training iterations')
    parser.add_argument('--head', action='store_true')
    parser.add_argument('--clip', action='store_true')
    parser.add_argument('--text_short', action='store_true')
    parser.add_argument('--learningrate', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--schedule', type=str, default='no', help='learning rate schedule')
    parser.add_argument('--cos_T', type=int, default=40, help='cosine schedule tem')
    parser.add_argument('--step_size', type=int, default=5, help='step schedule size')
    parser.add_argument('--gamma', type=float, default=0.1, help='step schedule gamma')
    parser.add_argument('--seed',type=int, default=0)
    args = parser.parse_args()

    main(args)

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
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, AutoConfig, Swinv2Config, Swinv2Model, CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPTokenizerFast, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection,AutoProcessor,AutoTokenizer,CLIPVisionModel
from peft import LoraConfig, get_peft_model,PeftModel
import wandb
import copy
import random
from reparam_module import ReparamModule
import torch.utils.data
import warnings
import gc
import collections
from ema_pytorch import EMA

from glad_utils import *
import nevergrad as ng

from functools import partial

warnings.filterwarnings("ignore", category=DeprecationWarning)

def softmax_entropy(output, label):
    """Entropy of softmax distribution from logits."""
    # print(output)
    loss = -(output.softmax(1) * output.log_softmax(1)).sum(1).mean()
    # print(loss)
    return loss
    

def ftwithdistill(net, images_train, labels_train, testloader, args, model_eva = None, class_names=None,processor = None):
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    sched1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0000001, end_factor=1.0, total_iters=Epoch//2)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epoch//2)
    sched = sched1
    criterion = nn.CrossEntropyLoss().to(args.device)
    ema = EMA(net, beta=0.995, power=1, update_after_step=0, update_every=1)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    loss_avg, acc_avg, num_exp = 0, 0, 0

    for ep in tqdm(range(Epoch+1)):
        img = images_train.to(args.device)
        lab = labels_train.to(args.device)
        
        n_b = lab.shape[0]

        inputs = processor(text=class_names,padding=True, return_tensors="pt").to(args.device)
        output_ori = net(pixel_values=img,**inputs)
        logits_per_image = output_ori.logits_per_image # this is the image-text similarity score
        output = logits_per_image.softmax(dim=1)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_avg /= num_exp
        acc_avg /= num_exp

        wandb.log({"loss": loss_avg}, step=ep)
        wandb.log({"acc": acc_avg}, step=ep)
        ema.update()
        sched.step()
        if ep == Epoch // 2:
            sched = sched2

        if ep == Epoch:
            with torch.no_grad():
                loss_test, acc_test, auc_test = test_task(ema, testloader, args, class_names=class_names,processor = processor)
                print("Test ACC: ", acc_test,"Test AUC: ", auc_test)
                # wandb.log({"Test Loss": loss_test})
                # wandb.log({"Test ACC": acc_test})
    return net
        # if ep in lr_schedule:
        #     optimizer = torch.optim.SGD(net.parameters(), lr=lr*0.1, momentum=0.9, weight_decay=0.0005)

def test_task(net, testloader, args, class_names=None,processor = None):
    criterion = nn.CrossEntropyLoss().to(args.device)
    lr = float(args.lr_net)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    loss_avg, acc_avg, num_exp = 0, 0, 0
    with torch.no_grad():
        # loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
        loss_test, acc_test, auc_test, f1_test, precision_test, recall_test = epoch('test', testloader, net, optimizer, criterion, args=args, aug=False,class_names=class_names,processor = processor)
        # wandb.log({"Test Loss": loss_test})
        wandb.log({"Test ACC": acc_test})
        wandb.log({"Test AUC": auc_test})
        print("Test ACC: ", acc_test,"Test AUC: ", auc_test)
        return loss_test,acc_test,auc_test

def initmodel(model, main_model_path, channel, num_classes, im_size=(32, 32), dist=True):
    if model == 'CLIP_Vit_b':
        visionmodel = CLIPVisionModel.from_pretrained(main_model_path)
        model = CLIPModel.from_pretrained(main_model_path)
        processor = CLIPProcessor.from_pretrained(main_model_path)
        processor.image_processor=CLIPImageProcessor(do_resize=False,do_center_crop=False,do_rescale=False,do_normalize=False,do_convert_rgb =False)
        tokenizer = AutoTokenizer.from_pretrained(main_model_path)
        for n, param in model.named_parameters():
            if 'text_model' in n:
                param.requires_grad = False
            # if 'projection' in n:
            #     param.requires_grad = False
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['q_proj','v_proj'],
            lora_dropout=0.1,
            bias="none",
        )
        config_proj = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['visual_projection'],
            lora_dropout=0.1,
            bias="none",
            # modules_to_save=["classifier"], #recheck this
        )
        
        visionmodel = get_peft_model(visionmodel, config)
        # lora_module = torch.load(lora_path)
        model.vision_model = visionmodel
        model = get_peft_model(model, config_proj)
        # model_dict = model.state_dict()
        # model_dict.update(lora_module)
        # model.load_state_dict(model_dict)
        
        # visionmodel = get_peft_model(visionmodel, config)
        # model.vision_model = visionmodel
        net = model
        # net = model
        return net, tokenizer
    elif model == 'CLIP_Vit_L':
        visionmodel = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        # textmodel = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
        # print(model)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        processor.image_processor=CLIPImageProcessor(do_resize=False,do_center_crop=False,do_rescale=False,do_normalize=False,do_convert_rgb =False)
        # print(processor)
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        for n, param in model.named_parameters():
            if 'text' in n:
                param.requires_grad = False
            # if 'projection' in n:
            #     param.requires_grad = False
                # modules_names.append(n)
                # net_parameters.append(param)
        config_encoder = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['q_proj','v_proj'],
            lora_dropout=0.1,
            bias="none",
            # modules_to_save=["classifier"], #recheck this
        )
        config_proj = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['visual_projection'],
            lora_dropout=0.1,
            bias="none",
            # modules_to_save=["classifier"], #recheck this
        )
        visionmodel = get_peft_model(visionmodel, config_encoder)
        model.vision_model = visionmodel
        model = get_peft_model(model, config_proj)
        # print(model)
        net = model

        # print(type(processor),type(net))
        return net, tokenizer
    elif model == 'CLIP_Vit_b_p32':
        visionmodel = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        processor.image_processor=CLIPImageProcessor(do_resize=False,do_center_crop=False,do_rescale=False,do_normalize=False,do_convert_rgb =False)
        # print(processor)
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        for n, param in model.named_parameters():
            if 'text' in n:
                param.requires_grad = False
        config_encoder = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['q_proj','v_proj'],
            lora_dropout=0.1,
            bias="none",
        )
        config_proj = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['visual_projection'],
            lora_dropout=0.1,
            bias="none",
        )
        visionmodel = get_peft_model(visionmodel, config_encoder)
        model.vision_model = visionmodel
        model = get_peft_model(model, config_proj)
        net = model
        return net, tokenizer
    elif model == 'CLIP_Vit_b_DoRA':
        visionmodel = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        processor.image_processor=CLIPImageProcessor(do_resize=False,do_center_crop=False,do_rescale=False,do_normalize=False,do_convert_rgb =False)
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")

        for n, param in model.named_parameters():
            if 'text' in n:
                param.requires_grad = False

        config_encoder = LoraConfig(
            use_dora=True,
            r=16,
            lora_alpha=16,
            target_modules=['q_proj','v_proj'],
            lora_dropout=0.1,
            bias="none",
        )
        config_proj = LoraConfig(
            use_dora=True,
            r=16,
            lora_alpha=16,
            target_modules=['visual_projection'],
            lora_dropout=0.1,
            bias="none",
        )
        visionmodel = get_peft_model(visionmodel, config_encoder)
        model.vision_model = visionmodel
        model = get_peft_model(model, config_proj)
        net = model
        return net, tokenizer


def get_loss(syndatalist, synlabellist, class_namelist, model,tokenizer,loss_type):
    """
    Get the loss of the model on the example dataset. Usually the example dataset only contains a few examples.
    """
    task_num = len(syndatalist)
    train_loss = 0
    if loss_type == 'CE':
        criterion = nn.CrossEntropyLoss().to(args.device)
    elif loss_type == 'entropy':
        criterion = softmax_entropy
    with torch.no_grad():
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # args.device = 'cpu'
        for i in range(task_num):
            # print(i)
            img = syndatalist[i].to(args.device)
            label = synlabellist[i].to(args.device)
            
            class_names = class_namelist[i]
            # print(img,label,class_names)
            inputs = tokenizer(text=class_names,padding=True, return_tensors="pt").to(args.device)
            with torch.no_grad():
                output_ori = model(pixel_values=img,**inputs)
            logits_per_image = output_ori.logits_per_image # this is the image-text similarity score
            output = logits_per_image.softmax(dim=1)
            loss = criterion(output, label)

            # loss = output_ori.loss
            # print(loss)
            train_loss += loss.detach().float()
        loss = train_loss.float()
    # average loss over the number of examples
    return float(loss) / task_num

def get_mixture_loss(syndatalist, synlabellist, class_namelist, model,cache,tokenizer,loss_type, weights):
    task_num = len(syndatalist)
    train_loss = 0
    if loss_type == 'CE':
        criterion = nn.CrossEntropyLoss().to(args.device)
    with torch.no_grad():
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # args.device = 'cpu'
        output = 0
        for j in range(len(cache)):
            for i in range(task_num):
                # print(i)
                model.load_state_dict(cache[j], strict=False)
                img = syndatalist[i].to(args.device)
                label = synlabellist[i].to(args.device)
                
                class_names = class_namelist[i]
                # print(img,label,class_names)
                inputs = tokenizer(text=class_names,padding=True, return_tensors="pt").to(args.device)
                with torch.no_grad():
                    output_ori = model(pixel_values=img,**inputs)
                logits_per_image = output_ori.logits_per_image # this is the image-text similarity score
                output_sub = weights[j]*logits_per_image.softmax(dim=1)
                output += output_sub
            
        loss = criterion(output, label).detach().float()

        #     # loss = output_ori.loss
        #     # print(loss)
        #     train_loss += loss.detach().float()
        # loss = train_loss.float()
    # average loss over the number of examples
    return float(loss) / task_num
    
def get_regular(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares
    
def get_score(weights, model, tokenizer, cache, syndatalist, synlabellist, class_namelist, get_loss, get_regular, loss_type='CE',level='whole',mode = 'merge'):
    # the composed lora state dict
    final_state_dict = {}
    # module list is the list
    lora_module_num = len(cache)
    # all keys are the same
    keys = list(cache[0].keys())
    for i in range(lora_module_num):
        lora_state_dict = cache[i]
        layer_len = len(lora_state_dict)
        if i == 0:
            if level == 'whole':
                for key in keys:
                    final_state_dict[key] = weights[i] * lora_state_dict[key]
            elif level == 'layer':
                for j in range(layer_len):
                    final_state_dict[keys[j]] = weights[i*layer_len+j] * lora_state_dict[keys[j]]
                    
        else:
            if level == 'whole':
                for key in keys:
                    final_state_dict[key] = (
                        final_state_dict[key] + weights[i] * lora_state_dict[key]
                    )
            elif level == 'layer':
                for j in range(layer_len):
                    final_state_dict[keys[j]] = (
                        final_state_dict[keys[j]] + weights[i*layer_len+j] * lora_state_dict[keys[j]]
                    )
    # reload the model with the new adapter config
    model.load_state_dict(final_state_dict, strict=False)
        
    # minimize the metric
    loss = get_loss(syndatalist, synlabellist, class_namelist, model,tokenizer,loss_type)
            
    # L1 regularization term
    metric_val = loss + get_regular(weights)
    
    return metric_val
    
def get_final_weights(weights, cache):
    final_state_dict = {}
    keys = cache[0].keys()
    lora_module_num = len(cache)
    for i in range(lora_module_num):
        lora_state_dict = cache[i]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    return final_state_dict

    
def lorahubmerge(lora_paths, syndatalist, synlabellist, class_namelist, net, tokenizer, max_inference_step, mode = 'merge', loss_type='CE', level='whole', balance = False):
    number_of_loras = len(lora_paths)
    if number_of_loras == 0:
        print("> No LoRA modules are provided. Please provide at least one LoRA module.")
        return None, None

    # load model
    model = net
    merge_time = number_of_loras-1
    recommendation_list = []
    
    for t in range(merge_time):
        cache =list()
        if t==0:
            lora_A = torch.load(lora_paths[0])
            loradict_A = {k: lora_A[k] for k in lora_A if "lora_" in k}
        lora_B = torch.load(lora_paths[t+1])
        loradict_B = {k: lora_B[k] for k in lora_B if "lora_" in k}
        layer_num = len(loradict_B)
        cache.append(loradict_A)
        cache.append(loradict_B)
        print(len(cache))
        # if t==0:
        get_score_partial = partial(get_score, 
                                model=model, 
                                tokenizer=tokenizer,
                                cache=cache,
                                syndatalist=syndatalist[:t+2],
                                synlabellist=synlabellist[:t+2],
                                class_namelist=class_namelist[:t+2],
                                # syndatalist=syndatalist,
                                # synlabellist=synlabellist,
                                # class_namelist=class_namelist,
                                get_loss=get_loss, 
                                get_regular=get_regular,
                                mode=mode,
                                loss_type = loss_type,
                                level=level
                                )
        # set up the limit of the weights
        if level=='whole':
            instrum = ng.p.Array(
                init=[0.5] * len(cache),
                upper=[1.5] * len(cache),
                lower=[-1.5] * len(cache),
            )
        elif level == 'layer':
            instrum = ng.p.Array(
                init=[0.5] * layer_num*len(cache),
                upper=[2] * layer_num*len(cache),
                lower=[-2] * layer_num*len(cache),
            )
        optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
        print(optimizer. _select_optimizer_cls())
        print("> Begin to perform gradient-free optimization ...")
        recommendation = optimizer.minimize(get_score_partial, verbosity=1)
        loradict_A = get_final_weights(recommendation.value, cache)
        print(recommendation.value)
        
    model.load_state_dict(loradict_A, strict=False)
    model = model.merge_and_unload()
    return recommendation.value, model, tokenizer
    
    
    
def weightedmerge(model, net, lora_path, syndatalist, synlabellist, class_namelist, mode = 'alldata', processor=None):
    Epoch = int(args.epoch_eval_train)
    w1=list()
    w2=list()
    if model == 'CLIP_Vit_b':
        A_module = net.state_dict()
        B_module = torch.load(lora_path)
        model_dict = net.state_dict()
        net_fin = net
        if mode == 'alldata':
            key_list = list(B_module.keys())
            w1 = [0.5 for _ in range(len(B_module.keys()))]
            w2 = [0.5 for _ in range(len(B_module.keys()))]
            w1 = torch.tensor(w1, dtype=torch.float32)
            w2 = torch.tensor(w2, dtype=torch.float32)
            w1.requires_grad = True
            w2.requires_grad = True
            optimizer = torch.optim.SGD([w1,w2], lr=0.01, momentum=0.9, weight_decay=0.0005)
            criterion = nn.CrossEntropyLoss().to(args.device)
            #train with task A
            for i in range(2):
                loss_avg, acc_avg, num_exp = 0, 0, 0
                for ep in tqdm(range(Epoch+1)):
                    img = syndatalist[i].to(args.device)
                    lab = synlabellist[i].to(args.device)
                    n_b = lab.shape[0]

                    inputs = processor(text=class_namelist[i],padding=True, return_tensors="pt").to(args.device)
                    new_dict = collections.OrderedDict()
                    for key, param in B_module.items():
                        w_init = A_module[key].to('cpu')
                        ind = key_list.index(key)
                        new_dict[key] = w1[ind]*w_init+w2[ind]*param.to('cpu')
                    model_dict.update(new_dict)
                    net_fin.load_state_dict(model_dict)
                        
                    output_ori = net_fin(pixel_values=img,**inputs)
                    logits_per_image = output_ori.logits_per_image # this is the image-text similarity score
                    output = logits_per_image.softmax(dim=1)
                    loss = criterion(output, lab)
                    # print(np.argmax(output.cpu().data.numpy(), axis=-1),lab.cpu().data.numpy())
                    acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

                    loss_avg += loss.item()*n_b
                    acc_avg += acc
                    num_exp += n_b
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    loss_avg /= num_exp
                    acc_avg /= num_exp
                    print(w1,w2)
    return net_fin

    

def main(args):

    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wandb.init(sync_tensorboard=False,
                project="MedForge_merge_Seed",
                name= args.detail,
                config=args,
                )

    testloader, channel, im_size, num_classes, class_names = get_testset(args.dataset, args.data_path, args=args)
    images_A, labels_A ,lr_A= get_synset(args.syn_path_A, args.label_A)
    images_B, labels_B, lr_B = get_synset(args.syn_path_B,args.label_B)
    images_C, labels_C, lr_C = get_synset(args.syn_path_C,args.label_C)
    tmpdatalist = list()
    tmpdatalist.append(images_A)
    tmpdatalist.append(images_B)
    tmpdatalist.append(images_C)
    tmplabellist = list()
    tmplabellist.append(labels_A)
    tmplabellist.append(labels_B)
    tmplabellist.append(labels_C)
    tmpclassnamelist = list()
    # tmpclassnamelist.append(["highly differentiated","poorly differentiated"])
    tmpclassnamelist.append(['benign breast tissues','malignant breast tumors'])
    tmpclassnamelist.append(['lung squamous cell carcinomas','lung adenocarcinomas','benign lung tissues'])
    tmpclassnamelist.append(['negtive colon tumor', 'positive colon tumor'])
    tmplora_paths = list()
    tmplora_paths.append(args.lora_path_A)
    tmplora_paths.append(args.lora_path_B)
    tmplora_paths.append(args.lora_path_C)
    order = args.order
    syndatalist = list()
    synlabellist = list()
    class_namelist = list()
    lora_paths = list()
    for i in range(len(order)):
        syndatalist.append(tmpdatalist[int(order[i])])
        synlabellist.append(tmplabellist[int(order[i])])
        class_namelist.append(tmpclassnamelist[int(order[i])])
        lora_paths.append(tmplora_paths[int(order[i])])

    # lora_paths = [args.lora_path_A, args.lora_path_C, args.lora_path_B]
    # net = get_network(args.model, channel, num_classes, im_size, width=args.width, depth=args.depth, dist=False).to(args.device)
    net, tokenizer = initmodel(args.model, args.main_model_path, channel, num_classes, im_size, dist=False)
    # net = net.to('cpu')
    net = net.to(args.device)
    mode = args.mode
    # ftwithdistill(net, images_A, labels_A, testloader, args, model_eva = None, class_names=class_name_A,processor = tokenizer)
    recom_weight, net, tokenizer = lorahubmerge(lora_paths, syndatalist, synlabellist, class_namelist, net, tokenizer, 
                                                args.max_inference_step, mode = mode, loss_type=args.loss_type, level=args.level, balance = args.distill_balance)
    
    # print(tokenizer)
    print(recom_weight)
    net = net.to(args.device)
    test_task(net, testloader, args, class_names=class_names,processor = tokenizer)
            
    wandb.finish()

if __name__ == '__main__':
    import shared_args

    parser = shared_args.add_shared_args()
    parser.add_argument('--syn_path_A', type=str, default=None, help='the path of the synthetic dataset, label(A)')
    parser.add_argument('--label_A', type=str, default=None, help='the path of the synthetic label, label(A)')
    parser.add_argument('--syn_path_B', type=str, default=None, help='the path of the synthetic dataset, label(B)')
    parser.add_argument('--label_B', type=str, default=None, help='the path of the synthetic label, label(B)')
    parser.add_argument('--syn_path_C', type=str, default=None, help='the path of the synthetic dataset, label(C)')
    parser.add_argument('--label_C', type=str, default=None, help='the path of the synthetic label, label(C)')
    parser.add_argument('--lora_path_A',type=str, default=None, help='the path of the lora module A to be merged')
    parser.add_argument('--lora_path_B',type=str, default=None, help='the path of the lora module B to be merged')
    parser.add_argument('--lora_path_C',type=str, default=None, help='the path of the lora module C to be merged')
    parser.add_argument('--main_model_path',type=str, default=None, help='the path of the main model')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_syn', type=int, default=None, help='batch size for syn data')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--load_all', action='store_true')
    parser.add_argument('--max_start_epoch', type=int, default=5)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--max_experts', type=int, default=None)
    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--lr_img', type=float, default=10000 , help='learning rate for pixels or f_latents')
    parser.add_argument('--lr_w', type=float, default=10, help='learning rate for updating synthetic latent w')
    parser.add_argument('--lr_lr', type=float, default=1e-06, help='learning rate learning rate')
    parser.add_argument('--lr_g', type=float, default=0.1, help='learning rate for gan weights')
    parser.add_argument('--detail', type=str, default='', help='probable comments')
    parser.add_argument('--client', type=int, default=-1, help='the index of client')
    parser.add_argument('--clip', action='store_true')
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--ori', action='store_true')
    parser.add_argument('--max_inference_step', type=int, default=40, help='the max optim steps')
    parser.add_argument('--order',type=str, default=None, help='the order of the tasks')
    parser.add_argument('--loss_type',type=str, default='CE', help='the type of loss')
    parser.add_argument('--level',type=str, default='whole', help='the level of weights')   
    parser.add_argument('--distill_balance', action='store_true')
    parser.add_argument('--mode',type=str, default='merge', help='the type of merging strategy')
    args = parser.parse_args()

    main(args)

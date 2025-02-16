import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, match_loss, get_time, \
    TensorDataset, epoch, DiffAugment, ParamDiffAug
import wandb
from tqdm import tqdm
import torchvision
import random
import gc
import warnings 
warnings.filterwarnings("ignore")
from glad_utils import *

import tracemalloc

import sys
# os.environ["WANDB_DISABLE"] = "keras"
# print(sys.modules)
# gc.enable()
# def show_memory():
#     print("*" * 60)
#     objects_list = []
#     for obj in gc.get_objects():
#         size = sys.getsizeof(obj)
#         objects_list.append((obj, size))
#     for obj, size in sorted(objects_list, key=lambda x: x[1], reverse=True)[:10]:
#         print(f"OBJ: {id(obj)}, TYPE: {type(obj)} SIZE: {size/1024/1024:.2f}MB {str(obj)[:100]}")

from collections import Counter
import linecache

def main(args):
    # tracemalloc.start()
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    args.dc_aug_param = None 

    seed = args.seed
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run = wandb.init(
        project = 'DM'+args.dataset + str(args.ipc) + args.model +args.detail+'dsa'+str(args.dsa),
        job_type="DM",
        name=time.strftime("%Y%m%d-%H%M%S")+str(seed),
        config=args
    )

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    run_dir = "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), run.name)

    args.save_path = os.path.join(args.save_path, "dm", run_dir)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv, dst_val, valloader = get_dataset(
            args.dataset, args.data_path, args.batch_real, args.res, args=args)
    gene_len = 10000
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []
    args.distributed = torch.cuda.device_count() > 1

    if args.space == 'p':
        G, zdim = None, None
    elif args.space == 'wp':
        G, zdim, w_dim, num_ws = load_sgxl(args.res, args)

    # images_all, labels_all, indices_class = build_dataset(dst_train, class_map, num_classes,args.gene,args.green)
    _, _, indices_class = build_dataset(dst_train, class_map, num_classes,args.gene,args.green)

    real_train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True,
                                                    num_workers=16)

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        res_img = []
        for idx in idx_shuffle:
            images_sel = torch.unsqueeze(dst_train[idx][0], dim=0)
            res_img.append(images_sel)
        images_res = torch.cat(res_img, dim=0).to(args.device)
        return images_res

    latents, f_latents, label_syn = prepare_latents(channel=channel, num_classes=num_classes, im_size=im_size,
                                                    zdim=zdim, G=G, class_map_inv=class_map_inv, get_images=get_images,
                                                    args=args)

    optimizer_img = get_optimizer_img(latents=latents, f_latents=f_latents, G=G, args=args)

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    metrics=['accs','aucs','f1','p','r']
    best_acc=dict()
    best_std=dict()
    best_acc_t=dict()
    best_std_t=dict()
    for m in model_eval_pool:
        for n in metrics:
            best_acc["{}_{}".format(m,n)] = 0
            best_std["{}_{}".format(m,n)] = 0
    for m in model_eval_pool:
        for n in metrics:
            best_acc_t["{}_{}".format(m,n)] = 0
            best_std_t["{}_{}".format(m,n)] = 0
    ''' initialize the synthetic data '''
    image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    print('%s training begins'%get_time())

    # best_acc = {"{}".format(m): 0 for m in model_eval_pool}

    # best_std = {m: 0 for m in model_eval_pool}
    geneseq_syn = None
    save_this_it = False
    
    # tracemalloc.start(10)
    # time_1 = tracemalloc.take_snapshot()
    # snapshot = tracemalloc.take_snapshot()
    # display_top(snapshot)
    
    if args.clip:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net, processor = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width, gene_dim=0,class_map=class_map, class_map_inv=class_map_inv)
            net = nn.DataParallel(net)
            net = net.cuda()
        else:
            net, processor = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width, gene_dim=0,class_map=class_map, class_map_inv=class_map_inv)
            net = net.to(args.device)
    else:
        net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width).to(args.device) # get a random model
    net.train()
    for param in list(net.parameters()):
        param.requires_grad = False
    
    for it in range(args.Iteration+1):
        torch.cuda.empty_cache()
        gc.collect()
        # show_memory()
        # tracemalloc.start()
        if it in eval_it_pool:
            save_this_it, testflag= eval_loop(latents=latents, f_latents=f_latents, label_syn=label_syn, gene_syn=geneseq_syn, G=G, best_acc=best_acc,
                                    best_std=best_std, testloader=testloader,
                                    model_eval_pool=model_eval_pool, channel=channel, num_classes=num_classes,
                                    im_size=im_size, it=it, args=args, mode='test',gene_dim=gene_len,log=True, class_names=class_names)

        if it > 0 and ((it in eval_it_pool and (save_this_it or it % 1000 == 0)) or (
                args.save_it is not None and it % args.save_it == 0)):
            image_logging(latents=latents, f_latents=f_latents, label_syn=label_syn, G=G, it=it, save_this_it=save_this_it, args=args)

        ''' Train synthetic data '''

        # embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

        loss_avg = 0

        if args.space == "wp":
            with torch.no_grad():
                image_syn_w_grad = torch.cat([latent_to_im(G, (syn_image_split, f_latents_split), args) for
                                              syn_image_split, f_latents_split, label_syn_split in
                                              zip(torch.split(latents, args.sg_batch),
                                                  torch.split(f_latents, args.sg_batch),
                                                  torch.split(label_syn, args.sg_batch))])
        else:
            image_syn_w_grad = latents

        if args.space == "wp":
            image_syn = image_syn_w_grad.detach()
            image_syn.requires_grad_(True)
        else:
            image_syn = image_syn_w_grad

        # snapshot = tracemalloc.take_snapshot()
        # display_top(snapshot)
        
        ''' update synthetic data '''
        if 'BN' not in args.model: # for ConvNet
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                img_real = get_images(c, args.batch_real).to(args.device)
                img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                inputs = processor(text=class_names,padding=True, return_tensors="pt").to(args.device)
                del inputs['input_ids']
                output_real = net.get_image_features(pixel_values=img_real)
                # print(output_real.shape)
                output_syn = net.get_image_features(pixel_values=img_syn)
                # output_real = net.(img_real).detach()
                # output_syn = embed(img_syn)

                loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

        else: # for ConvNetBN
            images_real_all = []
            images_syn_all = []
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                img_real = get_images(c, args.batch_real)
                img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                images_real_all.append(img_real)
                images_syn_all.append(img_syn)

            images_real_all = torch.cat(images_real_all, dim=0)
            images_syn_all = torch.cat(images_syn_all, dim=0)

            output_real = embed(images_real_all).detach()
            output_syn = embed(images_syn_all)

            loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)



        optimizer_img.zero_grad()
        loss.backward()

        if args.space == "wp":
            # this method works in-line and back-props gradients to latents and f_latents
            gan_backward(latents=latents, f_latents=f_latents, image_syn=image_syn, G=G, args=args)

        else:
            latents.grad = image_syn.grad.detach().clone()

        optimizer_img.step()
        loss_avg += loss.item()


        loss_avg /= (num_classes)

        wandb.log({
            "Loss": loss_avg
        }, step=it)

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

        if it == args.Iteration: # only record the final results
            data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
            torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.subset, args.model, args.ipc)))
        # snapshot = tracemalloc.take_snapshot()
        # display_top(snapshot)
        # gc.collect()
        # display_top(snapshot)
        # show_memory()
        # tracker.print_diff()
        # time_2 = tracemalloc.take_snapshot()
        # stats = time_2.compare_to(time_1, 'traceback')

        # top = stats[0]
        # print('\n'.join(top.traceback.format()))

if __name__ == '__main__':
    if __name__ == '__main__':
        import shared_args

        parser = shared_args.add_shared_args()

        parser.add_argument('--lr_img', type=float, default=10, help='learning rate for pixels or f_latents')
        parser.add_argument('--lr_w', type=float, default=.01, help='learning rate for updating synthetic latent w')
        parser.add_argument('--lr_g', type=float, default=0.0001, help='learning rate for gan weights')
        parser.add_argument('--lora',action='store_true')
        parser.add_argument('--loglogits', action='store_true')
        parser.add_argument('--hard', action='store_true')
        parser.add_argument('--clip', action='store_true')
        parser.add_argument('--text_short', action='store_true')
        parser.add_argument('--green', action='store_true')
        parser.add_argument('--detail', type=str, default='', help='probable comments')
        parser.add_argument('--gene', action='store_true')
        parser.add_argument('--lr_gene',type=float, default=0.1)
        parser.add_argument('--logvalperformance', action='store_true')
        parser.add_argument('--noimg', action='store_true')
        parser.add_argument('--balance', action='store_true')
        parser.add_argument('--gene_csv', type=str, default='')
        parser.add_argument('--seed',type=int, default=0)
        args = parser.parse_args()

        main(args)



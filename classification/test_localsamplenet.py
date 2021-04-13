import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
# import utils.provider
import importlib
import shutil
from models import pointnet_cls
from models.threeD_plot import threeD_plot
import src.samplenet 
from data.ModelNetDataLoader import ModelNetDataLoader
from src import sputils
from src.samplenet import SampleNet
import matplotlib.pyplot as plt
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from models.local_samplenet import Local_samplenet
import src.sputils
from knn_cuda import KNN

import time
import matplotlib.pyplot as plt
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--num_out_points', type=int, default=32, help='out Point Number [default: 32]')
    parser.add_argument('--log_dir', type=str, default='pointnet_cls_0908', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores with voting [default: 1]')
    parser.add_argument("--modelnet", type=float, default=30 ,help="chosie data base for training [default: 40")
    parser.add_argument("--bottleneck-size", type=int, default=128, help="bottleneck size [default: 128]")
    
    
    parser.add_argument("--seed_maker", type=str, default='FPS', help="FPS/samplenet")


    parser.add_argument("-npatches", "--num-patchs", type=int, default=32, help="Number of patches [default: 4]")
    parser.add_argument("-n_sper_patch", "--nsample-per-patch", type=int, default=32, help="Number of sample for each patch [default: 256]")
    parser.add_argument('--seeds_choice', default='FPS', help='FPS/Random/ Sampleseed- TBD')
    parser.add_argument("--trans_norm", type=bool, default=True, help="shift to center each patch")
    parser.add_argument("--scale_norm", type=bool, default=True, help="normelized scale of each patch")
    parser.add_argument("--concat_global_fetures", type=bool, default=True, help="concat global seeds to each patch")
    parser.add_argument("--one_feture_vec", type=bool, default=False, help="use one feture vector")
    parser.add_argument("--reduce_to_8", type=bool, default=True, help="reduce 32 points to 8")


    return parser.parse_args()

def test(model_task,model_sampler, loader, num_class=40, vote_num=1,sample_seed=None, one_feture_vec=False, reduce_to_8=False):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        
        # simp_pc, proj_pc = classifier.sampler(points)
        # pred, trans_feat = classifier(proj_pc)
        
        classifier = model_task.eval()
        classifier.sampler = model_sampler.eval()
        # sample_seed = model_sample_seed
        vote_pool = torch.zeros(target.size()[0],num_class).cuda()
        
        for _ in range(vote_num):
            #change to fps          proj_pc = fps(points)

            # if seed_maker =='samplenet':
            #     sample_seed = sample_seed.eval()
            #     _,seeds,seed_idx = sample_seed(points)

            # else:
            #     seed_idx = []
            
            

            print_3d = False
            if print_3d:
                plt.figure()

                numpy_seeds = seeds.cpu().detach().numpy()[:,:3]
                numpy_point = points.cpu().detach().numpy()[:,:3]
        
                ax = plt.axes(projection='3d')
            
                ax.scatter(numpy_point[:,0], numpy_point[:,1], numpy_point[:,2], s=0.01,color = 'blue')
                ax.scatter(numpy_seeds[:,0], numpy_seeds[:,1], numpy_seeds[:,2], s=0.01,color = 'red')
            
                plt.show()

            projected_points, simpc , end_points_sampler = classifier.sampler(points)
           
            #####
            
            if one_feture_vec or reduce_to_8:
                simpc = simpc
            else:
                simpc = end_points_sampler['simplified_points']
                #enter_points= simpc.permute(0,2,1)
                simpc = simpc.permute(0,2,1)
            
            x= points
            ####
            
            _, idx = KNN(1, transpose_mode=False)(x.contiguous(), simpc.contiguous())

            """Notice that we detach the tensors and do computations in numpy,
            and then convert back to Tensors.
            This should have no effect as the network is in eval() mode
            and should require no gradients.
            """
            # Convert to numpy arrays in B x N x 3 format. we assume 'bcn' format.
            x = x.permute(0, 2, 1).cpu().detach().numpy()
            simpc = simpc.permute(0, 2, 1).cpu().detach().numpy()

            idx = idx.cpu().detach().numpy()
            idx = np.squeeze(idx, axis=1)
            # idx= np.random.randint(1023, size=(1, 32))

            if reduce_to_8: 
                z = sputils.nn_matching(
                x, idx, 8, complete_fps=True)
            else:
                z = sputils.nn_matching(
                    x, idx, args.num_out_points, complete_fps=True
                )

            # Matched points are in B x N x 3 format.
            match = torch.tensor(z, dtype=torch.float32).cuda()
            
            match = match.permute(0,2,1)

            pred, _ = classifier(match)
            
            vote_pool += pred
        pred = vote_pool/vote_num
        pred_choice = pred.data.max(1)[1]
       
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    path_ev='/home/amit/Documents/sampleNet_stand_alone/SampleNet/registration/log/pointnet_cls_0908'
    file_handler = logging.FileHandler('%s/eval.txt' %  path_ev)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = 'data/modelnet40_normal_resampled/'
    
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, modelnet=args.modelnet,  split='test',
                                                    normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4,drop_last=False)

        ### model load names###
    clas_tesk_dir= 'log/pointnet_cls_task/'
    localsample_net_dir= 'log/LocalSamplenet/2021-04-11_21-31/'
   
   
   
    #sample net for seeds
    #seed_net_dir= 'log/classification/2020-10-26_19-29/'


    ### load classifier ####
    model_name = os.listdir(clas_tesk_dir+'/model')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(40,normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()
   
    checkpoint = torch.load(str(clas_tesk_dir) + 'weight/best_model_no_normal.pth')
    # checkpoint = torch.load(str(clas_tesk_dir) + 'weight/model_no_dropout.pth')

    classifier.load_state_dict(checkpoint['model_state_dict'])
    
    classifier.requires_grad_(False)
    classifier.eval().cuda()

    ##### load smaplenet insted FPS for seeds #######
    # if args.seed_maker =='samplenet':
    #     model_name = os.listdir(clas_tesk_dir+'/model')[0].split('.')[0]
    #     MODEL = importlib.import_module(model_name)
    #     classifier1 = MODEL.get_model(40,normal_channel=args.normal).cuda()
    #     criterion = MODEL.get_loss().cuda()
        
    
    #     checkpoint = torch.load(str(clas_tesk_dir) + 'checkpoints/best_model_no_normal.pth')
    #     classifier1.load_state_dict(checkpoint['model_state_dict'])
    #     classifier1.requires_grad_(False)
    #     classifier1.eval().cuda()
    
    #     sampler = SampleNet(
    #         num_out_points=8,
    #         bottleneck_size=128,
    #         group_size=7,
    #         initial_temperature=1.0,
    #         input_shape="bcn",
    #         output_shape="bcn",
    #         skip_projection=True, 
    #     ) 
        
    #     classifier1.sampler = sampler

    #     #checkpoint = torch.load(str(seed_net_dir) + 'checkpoints/sampler_cls_2609.pth')
    #     # classifier1.load_state_dict(checkpoint['model_state_dict'])
        
    #     sampler.requires_grad_(False)
    #     sampler.eval().cuda()


    ############# end of load samplenet ###############
    
    
    
    
    ### load local samplenet ###
    sampler= Local_samplenet(
        num_class = 10 , ##num of class sampler trained on
        batch_size=args.batch_size,
        num_output_points =args.num_out_points ,
        bottleneck_size = args.bottleneck_size,
        skip_projection=False,
        npatch = args.num_patchs ,
        nsample_per_patch = args.nsample_per_patch ,
        seed_choice = args.seeds_choice,
        trans_norm=args.trans_norm, 
        scale_norm=args.scale_norm,
        global_fetuers=args.concat_global_fetures,
        one_feture_vec = args.one_feture_vec,
        red_to_32 = args.reduce_to_8


        )
    
    classifier.sampler = sampler

    
    checkpoint = torch.load(str(localsample_net_dir) + 'checkpoints/sampler_cls_2609.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    sampler.requires_grad_(False)
    sampler.eval().cuda()


    with torch.no_grad():
        if args.seed_maker =="samplenet":
            instance_acc, class_acc,seed_idx = test(classifier.eval(),sampler.eval(), testDataLoader, vote_num=args.num_votes,sample_seed=classifier1.sampler.eval())
        else:
            instance_acc, class_acc = test(classifier.eval(),sampler.eval(), testDataLoader, vote_num=args.num_votes,sample_seed=None,one_feture_vec=args.one_feture_vec, reduce_to_8=args.reduce_to_8)

        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

         
   

if __name__ == '__main__':
    args = parse_args()
    main(args)


#from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
from models import pointnet_cls
import src.samplenet 
from data.ModelNetDataLoader import ModelNetDataLoader
from src import sputils
from src.samplenet import SampleNet
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# from src.laplotter import LossAccPlotter



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch',  default=400, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument("-in", "--num-in-points", type=int, default=1024, help="Number of input Points [default: 1024]")
    parser.add_argument("-out", "--num-out-points", type=int, default=128, help="Number of output points [2, 1024] [default: 64]")
    parser.add_argument("--bottleneck-size", type=int, default=128, help="bottleneck size [default: 128]")
    parser.add_argument("--alpha", type=float, default=30, help="Simplification regularization loss weight [default: 0.01]")
    parser.add_argument("--gamma", type=float, default=1, help="Lb constant regularization loss weight [default: 1]")
    parser.add_argument("--delta", type=float, default=0, help="Lb linear regularization loss weight [default: 0]")
    parser.add_argument("-gs", "--projection-group-size", type=int, default=7, help='Neighborhood size in Soft Projection [default: 8]')
    parser.add_argument("--lmbda", type=float, default=1, help="Projection regularization loss weight [default: 0.01]")
    parser.add_argument("--modelnet", type=float, default=10, help="chosie data base for training [default: 40")

    return parser.parse_args()

def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        
        classifier = model.eval()
        simp_pc, proj_pc = classifier.sampler(points)
        pred, trans_feat = classifier(proj_pc)

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
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = 'data/modelnet40_normal_resampled/'

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point , modelnet=args.modelnet, split='train',
                                                     normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, modelnet=args.modelnet, split='test', 
                                                    normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    # path_saved_cls = 'log/checkpoints/best_model_BU.pth'
    experiment_dir= 'log/pointnet_cls_0908/'
    # classifier = pointnet_cls.get_model(40,normal_channel=args.normal).cuda()
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    
    classifier = MODEL.get_model(40,normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model_no_normal.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # classifier.load_state_dict(torch.load(path_saved_cls)).

    classifier.requires_grad_(False)
   
   ## need to check
    classifier.eval().cuda()
   
    sampler = SampleNet(
        num_out_points=args.num_out_points,
        bottleneck_size=128,
        group_size=args.projection_group_size,
        initial_temperature=1.0,
        input_shape="bcn",
        output_shape="bcn",
        skip_projection=False, 
    ) 
    sampler.requires_grad_(True)
    sampler.train().cuda()
    
    

    # Attach sampler to cls_model
    classifier.sampler = sampler

    # learnable_params = filter(lambda p: p.requires_grad, classifier.parameters())
    learnable_params = [x for x in classifier.parameters() if x.requires_grad]
    
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(learnable_params, lr=args.learning_rate,  betas=(0.9, 0.999),eps=1e-08, weight_decay=1e-4)
    elif args.optimizer == "RMSProp":
        optimizer = torch.optim.RMSprop(learnable_params, lr=args.learning_rate)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=args.learning_rate, momentum=0.9)

    # if args.optimizer == 'Adam':
    #     optimizer = torch.optim.Adam(
    #         classifier.parameters(),
    #         lr=args.learning_rate,
    #         betas=(0.9, 0.999),
    #         eps=1e-08,
    #         weight_decay=args.decay_rate
    #     )
    # else:
    #     optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
   
    # help function to print in live losses
    
    # plotter = LossAccPlotter()

    # changing learning rate
   
   
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.7)
    global_epoch = 0
    global_step = 0
    
    tb = SummaryWriter(comment=f'samplenet: modelnet= {args.modelnet}: points={args.num_out_points} ')

    # best_instance_acc = 0.0
    # best_class_acc = 0.0
    mean_correct = []

   
    # num_class = 40
    # MODEL = importlib.import_module(args.model)
    # shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    # shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    # classifier = MODEL.get_model(num_class,normal_channel=args.normal).cuda()
    # criterion = MODEL.get_loss().cuda()

    # try:
    #     checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    #     start_epoch = checkpoint['epoch']
    #     classifier.load_state_dict(checkpoint['model_state_dict'])
    #     log_string('Use pretrain model')
    # except:
    #     log_string('No existing model, starting training from scratch...')
    #     start_epoch = 0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in (range(0,args.epoch)):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        scheduler.step()
        print('Epoch:', epoch,'LR:', scheduler.get_lr())
        
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            
            points, target = data
            points = points.data.numpy()
            # TODO amit: try to use augmentations when stable. 23/09
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
           
            points = torch.Tensor(points)
            target = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()


            simp_pc, proj_pc = classifier.sampler(points)
            pred, trans_feat = classifier(proj_pc)
            
             #loss
            simplification_loss = classifier.sampler.get_simplification_loss(points, simp_pc, args.num_out_points)
            projection_loss = sampler.get_projection_loss()
           
            # print('Proj = {}'.format(projection_loss))

            samplenet_loss = args.alpha * simplification_loss + args.lmbda * projection_loss
            task_loss = criterion(pred, target.long(), trans_feat)
            #print(f'Simp loss = {simplification_loss}, Task loss = {task_loss}')

            loss = task_loss + samplenet_loss

            # print('Epoch = {}, Step = {}, loss = {}'.format(epoch, batch_id, loss.cpu().data.numpy()))
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            
            tb.add_scalar('Loss/Total Loss', loss, global_step)
            tb.add_scalar('Loss/Samplenet loss ', samplenet_loss, global_step)
            tb.add_scalar('Loss/Simplification loss ', simplification_loss, global_step)
            tb.add_scalar('Loss/Projection_loss loss ', projection_loss, global_step)

            # plotter.add_values(global_epoch, loss_train=loss, redraw=False)

        # plotter.redraw()

        # plotter.block()
        
        train_instance_acc = np.mean(mean_correct)

        # with torch.no_grad():
        #     instance_acc, class_acc = test(classifier.eval(), testDataLoader)
           
        #     log_string('Test Instance Accuracy: %f, Class Accuracy: %f, trainig loss %f, samplent loss %f, g.step %f'% (instance_acc, class_acc,loss,samplenet_loss,global_step))
        
        # sampler.requires_grad_(True)
        # sampler.train().cuda()
        
        # classifier.sampler = sampler

        tb.add_scalar('Train Instance Accuracy ', train_instance_acc, epoch)
        log_string('Traim Instance Accuracy: %f, trainig loss %f, samplent loss %f, g.step %f'% (train_instance_acc, loss,samplenet_loss,global_step))

       
       
      

        savepath = str(checkpoints_dir) + '/sampler_cls_2609.pth'
        log_string('Saving at %s'% savepath)
        state = {
                    # 'test instance_acc': instance_acc,
                    'train instance_acc': train_instance_acc,
                    # 'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }         
                
        torch.save(state, savepath)
    
    

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)

import torch
from torch.utils import data
from torchsummary import summary
import matplotlib.pyplot as plt
from dataset import get_dataset, HyperX
from DMuCA import get_model
from train import test, train
from utils import get_device, sample_gt, compute_imf_weights, metrics, logger, display_goundtruth
import argparse
import numpy as np
import warnings
import datetime
import visdom


# 忽略警告
warnings.filterwarnings("ignore")

# 配置项目参数
parser = argparse.ArgumentParser(description="Run experiments on various hyperspectral datasets")

parser.add_argument('--dataset', type=str, default='IndianPines',
                    help="Choice one dataset for training"
                         "Dataset to train. Available:\n"
                         "PaviaU"
                         "Houston"
                         "IndianPines"
                         )
parser.add_argument('--model', type=str, default='Model_by_DMuCA',
                    help="Model to train.")

parser.add_argument('--folder', type=str, default='./dataset/',
                    help="Folder where to store the "
                         "datasets (defaults to the current working directory).")
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--restore', type=str, default=None,
                    help="Weights to use for initialization, e.g. a checkpoint")
parser.add_argument('--run', type=int, default=1,
                    help="Running times.")

group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--sampling_mode', type=str, default='random',
                           help="Sampling mode (random sampling or disjoint, default:  fixed)")
group_dataset.add_argument('--training_percentage', type=float, default=0.1,
                           help="Percentage of samples to use for training")
group_dataset.add_argument('--validation_percentage', type=float,
                           help="In the training data set, percentage of the labeled data are randomly "
                                "assigned to validation groups.")
group_dataset.add_argument('--train_gt', action='store_true',
                           help="Samples use of training")
group_dataset.add_argument('--test_gt', action='store_true',
                           help="Samples use of testing")
group_dataset.add_argument('--sample_nums', type=int, default=20,
                           help="Number of samples to use for training and validation")           
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int,
                         help="Training epochs")
group_train.add_argument('--save_epoch', type=int, default=5,
                         help="Training save epoch")
group_train.add_argument('--patch_size', type=int,
                         help="Size of the spatial neighbourhood (optional, if "
                              "absent will be set by the model)")
group_train.add_argument('--lr', type=float,
                         help="Learning rate, set by the model if not specified.")
group_train.add_argument('--batch_size', type=int,
                         help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--class_balancing', action='store_true',
                         help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")

# Data augmentation parameters
group_data = parser.add_argument_group('Data augmentation')

args = parser.parse_args()

RUN = args.run


# Dataset name
DATASET = args.dataset
# 生成日志
file_date = datetime.datetime.now().strftime('%Y-%m-%d')
log_date = datetime.datetime.now().strftime('%Y-%m-%d:%H:%M')
logger = logger('./logs/logs-' + file_date + DATASET +'.txt')
logger.info("---------------------------------------------------------------------")
logger.info("-----------------------------Next run log----------------------------")
logger.info("---------------------------{}--------------------------".format(log_date))
logger.info("---------------------------------------------------------------------")
# CUDA_DEVICE
CUDA_DEVICE = get_device(logger, args.cuda)
# Model name
MODEL = args.model
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Automated class balancing
SAMPLE_NUMS = args.sample_nums
PATCH_SIZE = args.patch_size
CLASS_BALANCING = args.class_balancing
TRAINING_PERCENTAGE = args.training_percentage
TEST_STRIDE = args.test_stride
TRAIN_GT = args.train_gt
TEST_GT = args.test_gt
EPOCH = args.epoch
hyperparams = vars(args)
# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(logger, DATASET, FOLDER)
# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Instantiate the experiment based on predefined networks
hyperparams.update(
    {'n_classes': N_CLASSES, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
# Get model
model, optimizer, loss, hyperparams = get_model(DATASET, **hyperparams)
# Open visdom server
if SAMPLING_MODE == 'fixed':
    vis = visdom.Visdom(
        env=DATASET + ' ' + MODEL + ' ' + 'PATCH_SIZE' + str(
            hyperparams['patch_size']) + ',' + 'EPOCH' + str(hyperparams['epoch']))
else:
    vis = visdom.Visdom(env=DATASET + ' ' + MODEL + ' ' + 'PATCH_SIZE' + str(
        hyperparams['patch_size']) + ' ' + 'EPOCH' + str(hyperparams['epoch']))
if not vis.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")
acc_dataset = np.zeros([RUN, 1])
A = np.zeros([RUN, N_CLASSES-1])
for i in range(RUN):
    model, optimizer, loss, hyperparams = get_model(DATASET, **hyperparams)
    
    train_gt, test_gt = sample_gt(gt, TRAINING_PERCENTAGE, mode=SAMPLING_MODE)
    # logger.info("Save train_gt successfully!(PATH:{})".format(train_gt_file))
    # logger.info("Save test_gt successfully!(PATH:{})".format(test_gt_file))
    logger.info("Running an experiment with the {} model, RUN [{}/{}]".format(MODEL, i + 1, RUN))
    logger.info("RUN:{}".format(i))
    # Open visdom server
    if SAMPLING_MODE=='fixed':
        vis = visdom.Visdom(env='SAMPLENUMS' + str(SAMPLE_NUMS) + ' ' + DATASET + ' ' + MODEL + ' ' + 'PATCH_SIZE' + str(PATCH_SIZE) + ' ' + 'EPOCH' + str(EPOCH))
    else:
        vis = visdom.Visdom(env=DATASET + ' ' + MODEL + ' ' + 'PATCH_SIZE' + str(PATCH_SIZE) + ' ' + 'EPOCH' + str(EPOCH))
#        vis = visdom.Visdom(env='TRAINING_PERCENTAGE' + str(TRAINING_PERCENTAGE*100) + '% ' + DATASET + ' ' + MODEL + ' ' + 'PATCH_SIZE' + str(PATCH_SIZE) + ' ' + 'EPOCH' + str(EPOCH))
    if not vis.check_connection:
        print("Visdom is not connected. Did you run 'python -m visdom.server' ?")

    
    logger.info("{} samples selected for training(over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
    logger.info("{} samples selected for training(over {})".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
    logger.info("Running an experiment with the {} model, RUN [{}/{}]".format(MODEL, i + 1, RUN))
    logger.info("RUN:{}".format(i))

    val_gt, test_dataset = sample_gt(gt, train_size=hyperparams['validation_percentage'], mode=SAMPLING_MODE)

    # Show groundtruth
    display_goundtruth(gt=gt, vis=vis, caption = "Training {} samples selected".format(np.count_nonzero(gt)))

    logger.info(
        "{} samples selected for validation(over {})".format(np.count_nonzero(val_gt), np.count_nonzero(gt)))

    logger.info("Running an experiment with the {} model".format(MODEL))

    # Class balancing
    if CLASS_BALANCING:
        weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
        hyperparams['weights'] = torch.from_numpy(weights).float().cuda()

    # Generate the dataset
    train_dataset = HyperX(img, train_gt, **hyperparams)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   shuffle=True,
                                   drop_last=True)
    logger.info("Train dataloader:{}".format(len(train_loader)))
    val_dataset = HyperX(img, val_gt, **hyperparams)
    val_loader = data.DataLoader(val_dataset,
                                 batch_size=hyperparams['batch_size'],
                                 drop_last=True)
    logger.info("Validation dataloader:{}".format(len(val_loader)))
    logger.info('----------Training parameters----------')
    for k, v in hyperparams.items():
        logger.info("{}:{}".format(k, v))
    logger.info("Network :")
    with torch.no_grad():
        for input, _ in train_loader:
            break
        summary(model.to(hyperparams['device']), input.size()[1:])
        
    if CHECKPOINT is not None:
        logger.info('Load model {} successfully!!!'.format(CHECKPOINT))
        # model.load_state_dict(torch.load(CHECKPOINT))
    try:
        logger.info('----------Training process----------')
        net = train(logger=logger, net=model, optimizer=optimizer, criterion=loss, train_loader=train_loader,
            epoch=hyperparams['epoch'], save_epoch=hyperparams['save_epoch'], scheduler=hyperparams['scheduler'],
            device=hyperparams['device'], supervision=hyperparams['supervision'], val_loader=val_loader,
            vis_display=vis, RUN=i,test_img = img,test_gt=test_gt,hyperparams=hyperparams,gt=gt)
        prediction = test(net, img, hyperparams)
        results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=hyperparams['n_classes'])
        color_gt = display_goundtruth(gt=prediction, vis=vis, caption="Testing ground truth(full)" + "RUN{}".format(i+1))
        plt.imsave("./groundtruth/{} patch{} RUN{}Testing gt(full) ".format(hyperparams['dataset'],hyperparams['patch_size'],i+1), color_gt)
        mask = np.zeros(gt.shape, dtype='bool')
        for l in hyperparams['ignored_labels']:
            mask[gt == l] = True
        prediction[mask] = 0
        color_gt = display_goundtruth(gt=prediction, vis=vis, caption="Testing ground truth(semi)" + "RUN{}".format(i+1))
        plt.imsave("./groundtruth/{} patch{} RUN{}Testing gt(semi) ".format(hyperparams['dataset'],hyperparams['patch_size'],i+1), color_gt)
        acc_dataset[i,0] = results['Accuracy']
        
        logger.info('----------Training result----------')
        logger.info("\nConfusion matrix:\n{}".format(results['Confusion matrix']))
        logger.info("\nAccuracy:\n{:.4f}".format(results['Accuracy']))
        logger.info("\nF1 scores:\n{}".format(np.around(results['F1 scores'], 4)))
        logger.info("\nKappa:\n{:.4f}".format(results['Kappa']))
        A[i] = results['F1 scores'][1:]
        print("acc_dataset {}".format(acc_dataset))
    
    except KeyboardInterrupt:
        # Allow the user to stop the training
        pass

OA_std = np.std(acc_dataset)
OAMean = np.mean(acc_dataset)
AA_std = np.std(A,0)
AAMean = np.mean(A,0)
p = []

logger.info("dataset:{}".format(DATASET))
for item,std in zip(AAMean,AA_std):
    p.append(str(round(item*100,2))+"+-"+str(round(std,2)))
logger.info(np.array(p))
# print("AAMean {:.2f} +-{:.2f}".format(AAMean,AA_std))
logger.info("OA_list {}".format(acc_dataset))
logger.info("OA±std {:.2f} ±{:.2f}".format(OAMean,OA_std))

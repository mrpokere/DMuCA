import itertools
import re
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import logging
import seaborn as sns
import spectral
import torch.nn as nn


def logger(logfile_name='logs/logs.log'):
    # 先创建记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 创建处理器handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)

    # 创建处理器filehandler
    fileHandler = logging.FileHandler(filename=logfile_name)
    fileHandler.setLevel(logging.DEBUG)

    # 设置日志输出格式
    #    formatter = logging.Formatter("%(asctime)s-%(filename)s-%(message)s", "%Y-%m-%d-%H-%M")
    formatter = logging.Formatter("%(message)s")
    # 设置处理器格式
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # 将记录器与处理器相关联
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    logger.debug('creating {}'.format(logfile_name))

    return logger


def show_plot(iteration, loss):
    import matplotlib.pyplot as plt
    # 绘制损失变化图
    plt.plot(iteration, loss)
    plt.show()


def show_inference(pre_data, label, args):
    from spectral import imshow
    import matplotlib.pyplot as plt
    pred_matrix = np.zeros((args.size[0], args.size[1]))
    count = 0
    for i in range(args.size[0]):
        for j in range(args.size[1]):
            if label[i][j] != 0:
                pred_matrix[i][j] = pre_data[count]
                count += 1

    # save_rgb(args.save_fig_name, pred_matrix, colors=spy_colors)
    imshow(classes=pred_matrix.astype(int))
    plt.savefig(args.save_fig_name)
    return pred_matrix


def sample_gt(gt, train_size=None, mode='random', sample_nums=None):
    indices = np.nonzero(gt)
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
        train_size = int(train_size)

    if mode == 'random':
        train_indices, test_indices = train_test_split(X, train_size=train_size, stratify=y)
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[train_indices] = gt[train_indices]
        test_gt[test_indices] = gt[test_indices]
    elif mode == 'fixed':
        train_indices, test_indices = [], []
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features
            # 如果采样数量比样本数量大，对样本进行镜像填充
            if sample_nums / len(X) > 1:
                X_copy = X.copy()
                for i in range(sample_nums // len(X)):
                    X += X_copy

            train, test = train_test_split(X, train_size=sample_nums / len(X))
            train_indices += train
            test_indices += test
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[train_indices] = gt[train_indices]
        test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True  # 未定义的坐标物定义取True，其它取False
    ignored_mask = ~ignored_mask  # 再对标记矩阵取反，未定义取False，其它取True
    # target = target[ignored_mask] -1
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes
    # 计算混淆矩阵
    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
         float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    return results


def build_dataset(mat, gt, ignored_labels=None):
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.array(samples).astype(float), np.array(labels).astype(float)


def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.
    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)
    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format
    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        # Generate color palette
        palette = {0: (0, 0, 0)}

        # flatui = [,, "#DAA520", "#FFA500", "#8B4513"]
        flatui = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#CD853F", "#2E8B57", "#8A2BE2",
                  "#C71585", "#32CD32", "#4169E1", "#8B4513", "#5F9EA0", "#FFFACD", "#800000", "#B0C4DE", "#00FFFF",
                  "#D2B48C", "#FA8072"]
        # l-亮度 lightness / s-饱和 saturation
        for k, color in enumerate(sns.color_palette(flatui)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def display_goundtruth(gt, vis, caption=""):
    color_gt = convert_to_color_(gt)
    vis.images([np.transpose(color_gt, (2, 0, 1))], opts={'caption': caption})
    return color_gt


def display_dataset(img, vis):
    rgb = spectral.get_rgb(img, [29, 19, 9])
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')

    # Display the RGB composite image
    caption = "Data set ground truth"
    # send to visdom server
    vis.images([np.transpose(rgb, (2, 0, 1))],
               opts={'caption': caption})


def build_dataset(mat, gt, ignored_labels=None):
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.array(samples).astype(float), np.array(labels).astype(float)
    

def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def sliding_window(image, step=1, window_size=(9, 9), with_data=True):
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    # W2, H2 = spa_image.shape[:2]
#    offset_w = (W - w) % step
#    offset_h = (H - h) % step
    for x in range(0, W - w + 1, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + 1, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(img, step=1, window_size=(9, 9)):
    sw = sliding_window(img, step, window_size=window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def get_device(logger, ordinal):
    # Use GPU ?
    if ordinal < 0:
        logger.info("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        logger.info("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        logger.info("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0)

        target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

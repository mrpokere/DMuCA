import torch
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing

DATASETS_CONFIG = {
    'PaviaU': {
        'img': 'PaviaU.mat',
        'gt': 'PaviaU_gt.mat'
    },
    'KSC': {
        'img': 'KSC.mat',
        'gt': 'KSC_gt.mat'
    },
    'IndianPines': {
        'img': 'indian_pines.mat',
        'gt': 'indian_pines_gt.mat'
    },
    'Salinas': {
        'img': 'salinas.mat',
        'gt': 'salinas_gt.mat'
    },
    'Houston': {
        'img': 'Houston.mat',
        'gt': 'Houston_gt.mat'
    },
    'Botswana': {
        'img': 'Botswana.mat',
        'gt': 'Botswana_gt.mat'
    }
}


def get_dataset(logger, dataset_name, target_folder="./dataset/", datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    img_file = target_folder + dataset_name + '/' + datasets[dataset_name].get('img')
    gt_file = target_folder + dataset_name + '/' + datasets[dataset_name].get('gt')
    if dataset_name == 'Houston':
        # Load the image
        img = loadmat(img_file)['Houston']

        gt = loadmat(gt_file)['Houston_gt']

        label_values = ["Undefined", "Healthy grass", "Stressed grass", " Synthetic grass",
                        "Trees", "Soil", "Water", "Residential", "Commercial", "Road",
                        "Highway", "Railway", "Parking Lot 1", "Parking Lot 2",
                        "Tennis Court", "Running Track"]

        ignored_labels = [0]

    elif dataset_name == 'PaviaU':
        # Load the image
        img = loadmat(img_file)['paviaU']

        gt = loadmat(gt_file)['Data_gt']

        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

        ignored_labels = [0]

    elif dataset_name == 'IndianPines':
        # Load the image
        img = loadmat(img_file)
        img = img['HSI_original']

        gt = loadmat(gt_file)['Data_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]

        ignored_labels = [0]

    elif dataset_name == 'Botswana':
        # Load the image
        img = loadmat(img_file)['Botswana']

        gt = loadmat(gt_file)['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]

        ignored_labels = [0]

    elif dataset_name == 'KSC':
        # Load the image
        img = loadmat(img_file)['KSC']

        gt = loadmat(gt_file)['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]

        ignored_labels = [0]

    elif dataset_name == 'Salinas':
        # Load the image
        img = loadmat(img_file)['HSI_original']

        gt = loadmat(gt_file)['Data_gt']
        label_values = ["Undefined", "Brocoli green weeds 1", "Brocoli_green_weeds_2",
                        "Fallow", "Fallow rough plow", "Fallow smooth", "Stubble",
                        "Celery", "Grapes untrained", "Soil vinyard develop",
                        "Corn senesced green weeds", "Lettuce romaine 4wk",
                        "Lettuce romaine 5wk", "Lettuce romaine 6wk", "Lettuce romaine 7wk",
                        "Vinyard untrained", "Vinyard vertical trellis"]

        ignored_labels = [0]

    # 检查像元是否为空
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        logger.info("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN "
              "data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype='float32')
    data = img.reshape(np.prod(img.shape[:2]), np.prod(img.shape[2:]))
    data = preprocessing.minmax_scale(data, axis=1)
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    data = scaler.fit_transform(data)
    img = data.reshape(img.shape) 
    return img, gt, label_values, ignored_labels


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.dataset_name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def get_data(data, x, y, patch_size, data_3D=False):
        x1, y1 = x - patch_size // 2, y - patch_size // 2
        x2, y2 = x1 + patch_size, y1 + patch_size
        data = data[x1:x2, y1:y2]
        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        # Add a fourth dimension for 3D CNN
        if data_3D:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)  # uncommon if need.
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        data = self.get_data(self.data, x, y, self.patch_size, data_3D=False)
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        label = self.label[x1:x2, y1:y2]
        label = np.asarray(np.copy(label), dtype='int64')
        label = torch.from_numpy(label)
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        
        return data, label


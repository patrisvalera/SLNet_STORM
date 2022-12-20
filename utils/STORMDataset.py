import torch
from torch.utils import data
from tifffile import imread
import numpy as np
import torch.nn.functional as F
import multipagetiff as mtif
import tensorflow as tf
from utils.misc_utils import center_crop


class STORMDatasetFull(data.Dataset):
    def __init__(self, data_path, img_shape, images_to_use=None, load_sparse=False, temporal_shifts=[0, 1, 2],
                 use_random_shifts=False, maxWorkers=10):
        # Initialize arguments
        self.images_to_use = images_to_use
        self.data_path = data_path
        self.load_sparse = load_sparse
        self.temporal_shifts = temporal_shifts
        self.n_frames = len(temporal_shifts)
        self.use_random_shifts = use_random_shifts
        self.img_shape = img_shape

        # Tiff images are stored in single tiff stack
        imgs_path = data_path + '/STORM_image/STORM_image_stack.tif'
        imgs_path_sparse = data_path + '/STORM_image/STORM_image_stack_S.tif'

        self.img_dataset = imread(imgs_path, maxworkers=maxWorkers)
        n_frames, h, w = self.img_dataset.shape

        # Calculate the median of the whole dataset
        self.median, self.indexes = torch.from_numpy(self.img_dataset).median(dim=0)

        if self.load_sparse:
            try:
                self.img_dataset_sparse = imread(imgs_path_sparse, maxworkers=maxWorkers)
            except:
                self.load_sparse = False
                print('Dataset error: STORM_image/STORM_image_stack_S.tif not found')

        # If no images are specified to use, list sequentially
        if images_to_use is None:
            images_to_use = list(range(n_frames))
        self.n_images = min(len(images_to_use), n_frames)

        n_images_to_load = max(images_to_use) + max(temporal_shifts) + 1

        # Create image storage
        self.stacked_views = torch.zeros(n_images_to_load, self.img_shape[0], self.img_shape[1], dtype=torch.float32)

        if self.load_sparse:
            stacked_views_sparse = self.stacked_views.clone()

        for nImg in range(n_images_to_load):

            # Load the images indicated from the user
            curr_img = nImg  # images_to_use[nImg]

            image = torch.from_numpy(np.array(self.img_dataset[curr_img, :, :]).astype(np.float32)).type(torch.float32)

            image = self.pad_img_to_min(image)
            # self.stacked_views[nImg, ...] = center_crop(image.unsqueeze(0).unsqueeze(0), self.img_shape)[0, 0, ...]
            self.stacked_views[nImg, ...] = image

            if self.load_sparse:
                image = torch.from_numpy(np.array(self.img_dataset_sparse[curr_img, :, :]).astype(np.float32)).type(
                    torch.float32)
                image = self.pad_img_to_min(image)
                stacked_views_sparse[nImg, ...] = image

        if self.load_sparse:
            self.stacked_views = torch.cat((self.stacked_views.unsqueeze(-1), stacked_views_sparse.unsqueeze(-1)),
                                           dim=3)

        print('Loaded ' + str(self.n_images))

    def __len__(self):
        """Denotes the total number of samples"""
        return self.n_images

    def get_n_temporal_frames(self):
        return len(self.temporal_shifts)

    def get_max(self):
        """Get max intensity from images for normalization"""
        if self.load_sparse:
            return self.stacked_views[..., 0].float().max().type(self.stacked_views.type()), \
                   self.stacked_views[..., 1].float().max().type(self.stacked_views.type())
        else:
            return self.stacked_views.float().max().type(self.stacked_views.type()), \
                   self.stacked_views.float().max().type(self.stacked_views.type())

    def get_statistics(self):
        """Get mean and standard deviation from images for normalization"""
        if self.load_sparse:
            return self.stacked_views[..., 0].float().mean().type(self.stacked_views.type()), \
                   self.stacked_views[..., 0].float().std().type(self.stacked_views.type()), \
                   self.stacked_views[..., 1].float().mean().type(self.stacked_views.type()), \
                   self.stacked_views[..., 1].float().std().type(self.stacked_views.type())
        else:
            return self.stacked_views.float().mean().type(
                self.stacked_views.type()), self.stacked_views.float().std().type(self.stacked_views.type())

    def standarize(self, stats=None):
        mean_imgs, std_imgs, mean_imgs_s, std_imgs_s, mean_vols, std_vols = stats
        if self.load_sparse:
            self.stacked_views[..., 0] = (self.stacked_views[..., 0] - mean_imgs) / std_imgs
            self.stacked_views[..., 1] = (self.stacked_views[..., 1] - mean_imgs_s) / std_imgs_s
        else:
            self.stacked_views[...] = (self.stacked_views[...] - mean_imgs) / std_imgs

    def pad_img_to_min(self, image):
        min_size = min(image.shape[-2:])
        img_pad = [min_size - image.shape[-1], min_size - image.shape[-2]]
        img_pad = [img_pad[0] // 2, img_pad[0] // 2, img_pad[1], img_pad[1]]
        image = F.pad(image.unsqueeze(0).unsqueeze(0), img_pad)[0, 0]
        return image

    def __getitem__(self, index):
        n_frames = self.get_n_temporal_frames()
        new_index = self.images_to_use[index]

        temporal_shifts_ixs = self.temporal_shifts
        # if self.use_random_shifts:
        #     temporal_shifts_ixs = torch.randint(0, self.n_images-1,[3]).numpy()
        #     newIndex = 0

        indices = [new_index + temporal_shifts_ixs[i] for i in range(n_frames)]

        views_out = self.stacked_views[indices, ...]

        return views_out

    @staticmethod
    def read_tiff_stack(filename, out_datatype=torch.float32):
        tiffarray = mtif.read_stack(filename, units='voxels')
        try:
            max_val = torch.iinfo(out_datatype).max
        except:
            max_val = torch.finfo(out_datatype).max
        out = np.clip(tiffarray.raw_images, 0, max_val)
        return torch.from_numpy(out).permute(1, 2, 0).type(out_datatype)

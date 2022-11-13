import torch
from torch.utils import data
from tifffile import imread
import numpy as np
import torch.nn.functional as F
import multipagetiff as mtif


# TODO: delete comment
# removed volume stuff, lenslet coords, subimage, depths, border_blanking, left sparse, left padding,
# removed add_random_shot_noise_to_dataset

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
        self.stacked_views = torch.zeros(n_images_to_load, self.img_shape[0], self.img_shape[1], dtype=torch.float16)

        if self.load_sparse:
            stacked_views_sparse = self.stacked_views.clone()

        for nImg in range(n_images_to_load):

            # Load the images indicated from the user
            curr_img = nImg  # images_to_use[nImg]

            image = torch.from_numpy(np.array(self.img_dataset[curr_img, :, :]).astype(np.float16)).type(torch.float16)

            image = self.pad_img_to_min(image)
            self.stacked_views[nImg, ...] = image

            if self.load_sparse:
                image = torch.from_numpy(np.array(self.img_dataset_sparse[curr_img, :, :]).astype(np.float16)).type(
                    torch.float16)
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
            return self.stacked_views[..., 0].float().mean().type(self.stacked_views.type()), self.stacked_views[
                ..., 0].float().std().type(self.stacked_views.type()), \
                   self.stacked_views[..., 1].float().mean().type(self.stacked_views.type()), self.stacked_views[
                       ..., 1].float().std().type(self.stacked_views.type())
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

    # TODO: needed?
    @staticmethod
    def extract_views(image, lenslet_coords, subimage_shape, debug=False):
        half_subimg_shape = [subimage_shape[0] // 2, subimage_shape[1] // 2]
        n_lenslets = lenslet_coords.shape[0]
        stacked_views = torch.zeros(
            size=[image.shape[0], image.shape[1], n_lenslets, subimage_shape[0], subimage_shape[1]],
            device=image.device, dtype=image.dtype)

        if debug:
            debug_image = image.detach().clone()
            max_img = image.float().cpu().max()
        for nLens in range(n_lenslets):
            # Fetch coordinates
            currCoords = lenslet_coords[nLens, :]
            if debug:
                debug_image[:, :, currCoords[0] - 2:currCoords[0] + 2, currCoords[1] - 2:currCoords[1] + 2] = max_img
            # Grab patches
            lower_bounds = [currCoords[0] - half_subimg_shape[0], currCoords[1] - half_subimg_shape[1]]
            lower_bounds = [max(lower_bounds[kk], 0) for kk in range(2)]
            currPatch = image[:, :, lower_bounds[0]: currCoords[0] + half_subimg_shape[0],
                        lower_bounds[1]: currCoords[1] + half_subimg_shape[1]]
            stacked_views[:, :, nLens, -currPatch.shape[2]:, -currPatch.shape[3]:] = currPatch

        if debug:
            import matplotlib.pyplot as plt
            plt.imshow(debug_image[0, 0, ...].float().cpu().detach().numpy())
            plt.show()
        return stacked_views

    @staticmethod
    def read_tiff_stack(filename, out_datatype=torch.float16):
        tiffarray = mtif.read_stack(filename, units='voxels')
        try:
            max_val = torch.iinfo(out_datatype).max
        except:
            max_val = torch.finfo(out_datatype).max
        out = np.clip(tiffarray.raw_images, 0, max_val)
        return torch.from_numpy(out).permute(1, 2, 0).type(out_datatype)

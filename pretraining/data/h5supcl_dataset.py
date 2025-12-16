import os
import numpy as np
import torch
import torch.utils.data
from data.data_utils import normalize_img, random_crop
from data.base_dataset import BaseDataset
import torchio as tio
import h5py
import gc

class H5SupCLDataset(BaseDataset):
    """
    Dataset class for supervised contrastive learning using HDF5 files.

    This class loads 3D medical images and their segmentations from HDF5 files,
    supports two-view contrastive learning, and applies optional augmentations
    and preprocessing steps such as resizing and cropping.

    In its convention, `A` is view 1 and `B` is view 2. This is arbitrary.

    Parameters
    ----------
    opt : Namespace
        Options for dataset loading and preprocessing.

    Attributes
    ----------
    opt : Namespace
        Options for dataset loading and preprocessing.
    folder : str
        Path to the dataset root directory.
    isTrain : bool
        Whether we are in training mode (and not eval).
    dimension : int
        Number of spatial dimensions (only 3 is supported as of now).
    h5_data : str
        Path to the HDF5 file.
    load_mask : bool
        Whether to load masks (not implemented yet).
    percentile : float
        Percentile for normalization.
    zero_centered : bool
        Whether to zero-center images.
    mode : str
        Data loading mode (should be 'twoview').
    hf : h5py.File
        Opened HDF5 file.
    subj_id : list
        List of subject IDs in the HDF5 file.
    len : int
        Number of subjects.
    load_seg : bool
        Whether to load segmentations.
    crop_size : int
        Size for cropping or resizing.
    pre_resize : torchio.Resize
        TorchIO resize transform (if resizing is enabled).
    apply_same_inten_augment : bool
        Whether to apply the same intensity augmentation to both views.
    augment_fn_intensity : torchio.Compose
        Composed intensity augmentation transform.
    augment_fn_spatial : torchio.Compose
        Composed spatial augmentation transform.
    """

    def __init__(self, opt):
        """
        Initialize the H5SupCLDataset.

        Parameters
        ----------
        opt : Namespace
            Options for dataset loading and preprocessing.
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.folder = opt.dataroot
        self.isTrain = opt.isTrain
        key = "train" if self.isTrain else "val"
        self.dimension = opt.data_ndims
        self.h5_data = self.folder + f"/{key}_data.hdf5"
        self.load_mask = opt.load_mask
        self.percentile = 99.99
        self.zero_centered = False
        self.mode = opt.load_mode

        print(
            "----------- %dd Data Loader Initialization, load mode [%s] ----------"
            % (self.dimension, self.mode)
        )

        # Only 'twoview' mode is supported
        if self.mode == "twoview":
            print(self.h5_data, os.path.exists(self.h5_data))
            self.hf = None
            self.subj_id = None
            self.len = None

            # We need dataset length still
            with h5py.File(self.h5_data, "r") as f:
                self.subj_id = list(f.keys())
                self.len = len(self.subj_id)
            print("load views in order: {}".format(self.opt.view_order))
        else:
            raise NotImplementedError("Only 'twoview' mode is implemented.")

        self.load_seg = True
        self.crop_size = opt.crop_size

        # Set up resizing if enabled
        if self.opt.resize:
            print("Including resizing in preprocessing: %d" % opt.crop_size)
            assert (
                opt.crop_size > 0
            ), "Wrong size %d specified for resizing." % (opt.crop_size)
            self.pre_resize = tio.Resize(
                (opt.crop_size, opt.crop_size, opt.crop_size)
            )

        self.apply_same_inten_augment = opt.apply_same_inten_augment

        # Set up augmentations if enabled
        # TODO: these really should be in a separate config file...
        if opt.isTrain and opt.augment:
            print("Initializing augmentation ...")

            augment_list_intensity = []
            if opt.inten_augment:
                print("Add in intensity augmentation ")
                if opt.blur:
                    print("Random Blur")
                    augment_list_intensity += [tio.RandomBlur(p=0.33)]
                if opt.noise:
                    print("Noise")
                    augment_list_intensity += [tio.RandomNoise(p=0.33)]
                if opt.bias:
                    print("Bias Field")
                    augment_list_intensity += [tio.RandomBiasField(p=0.5)]
                if opt.gamma:
                    print("Gamma")
                    augment_list_intensity += [
                        tio.RandomGamma(p=0.5, log_gamma=(-0.4, 0.4))
                    ]
                if opt.motion:
                    print("Motion")
                    augment_list_intensity += [tio.RandomMotion(p=0.33)]
            self.augment_fn_intensity = tio.Compose(augment_list_intensity)

            augment_list_spatial = []
            if opt.geo_augment:
                print("Add in geometric augmentation ")
                augment_list_spatial += [tio.RandomFlip(p=0.9, axes=(0, 1, 2))]
                self.scale = 0.4
                self.degrees = 45
                print(
                    "affine configs: scale {}, degree {}".format(
                        self.scale, self.degrees
                    )
                )
                if self.dimension == 3:
                    augment_list_spatial += [
                        tio.RandomAffine(
                            p=0.5, scales=self.scale, degrees=self.degrees
                        )
                    ]
                elif self.dimension == 2:
                    augment_list_spatial += [
                        tio.RandomAffine(
                            p=0.5,
                            scales=(self.scale, 0, 0),
                            degrees=(self.degrees, 0, 0),
                        )
                    ]
                else:
                    raise NotImplementedError(
                        "Only 2D or 3D data supported for augmentation."
                    )
            self.augment_fn_spatial = tio.Compose(augment_list_spatial)

        else:
            print("No augmentation.")
            self.augment_fn_intensity, self.augment_fn_spatial = None, None


    def __getitem__(self, item):
        """
        Get a data sample for the given index.

        Parameters
        ----------
        item : int
            Index of the sample to retrieve.

        Returns
        -------
        return_dict : dict
            Dictionary containing image data, segmentations, and metadata.
        """
        return_dict = dict()

        # Only 'twoview' mode is supported (3D only)
        if self.mode != 'twoview':
            raise NotImplementedError("Only 'twoview' mode is implemented.")

        # opening hdf5 each iter incurs a memory overhead but seems to be the
        # only way to avoid a memory leak with num_workers > 0
        # TODO: figure out why 
        with h5py.File(self.h5_data, "r", libver="latest", swmr=False) as hf:
            # Ensure item is within bounds
            while item >= len(self.subj_id):
                item = torch.randint(0, self.len, ()).numpy()
            
            assert (
                self.dimension == 3
            ), f"Only support 3D data loading in mode {self.mode}"

            subj = self.subj_id[item]
            n_tps_per_subj = hf[subj]["img"].shape[0]

            # TODO: this can be simplified to just assign random i and j
            # between 0 and 1 (integers) and make sure that they dont match
            # ...or just remove this entirely
            if self.opt.view_order:
                i = torch.randint(0, n_tps_per_subj - 1, ()).numpy()
                j = i + 1
            else:
                # restrict the pairs to from random timepoints per subject
                i = torch.randint(0, n_tps_per_subj, ()).numpy()
                j = torch.randint(0, n_tps_per_subj, ()).numpy()
                while j == i:
                    j = torch.randint(0, n_tps_per_subj, ()).numpy()

            img_keys = ["A", "B", "A_seg", "B_seg"]

            # Load and normalize images for both views
            A_orig = normalize_img(
                hf[subj]["img"][i],
                percentile=self.percentile,
                zero_centered=self.zero_centered,
            )[None, ...]
            return_dict["A_id"] = np.asarray(
                [item]
            )  # Dummy variable that contains the subject id

            B_orig = normalize_img(
                hf[subj]["img"][j],
                percentile=self.percentile,
                zero_centered=self.zero_centered,
            )[None, ...]
            return_dict["meta"] = "%s" % (
                subj,
            )
            return_dict["B_id"] = np.asarray(
                [item]
            )  # Dummy variable that contains the subject id

            # Load segmentation (assumed to be the same for both views)
            AB_seg_orig = np.array(hf[subj]["seg"])

            if self.opt.augment and self.isTrain:
                # Create TorchIO subjects for augmentation
                A = tio.Subject(
                    img=tio.ScalarImage(tensor=torch.from_numpy(A_orig)),
                    label=tio.LabelMap(
                        tensor=torch.from_numpy(AB_seg_orig).unsqueeze(0)
                    ),
                )
                B = tio.Subject(
                    img=tio.ScalarImage(tensor=torch.from_numpy(B_orig)),
                    label=tio.LabelMap(
                        tensor=torch.from_numpy(AB_seg_orig).unsqueeze(0)
                    ),
                )

                # Apply resizing if enabled
                if self.opt.resize:
                    A = self.pre_resize(A)

                # Apply spatial augmentations
                A = self.augment_fn_spatial(A)

                if self.apply_same_inten_augment:
                    # Apply the same intensity augmentation to both A and B
                    A = self.augment_fn_intensity(A)
                    all_transform = A.get_composed_history()
                    B = all_transform(B)
                else:
                    # Apply geometric transform to both, then separate intensity augmentations
                    geo_transform = A.get_composed_history()
                    B = geo_transform(B)
                    A = self.augment_fn_intensity(A)
                    B = self.augment_fn_intensity(B)

                # Extract augmented data
                return_dict["A"] = A["img"][tio.DATA]
                return_dict["B"] = B["img"][tio.DATA]
                return_dict["A_seg"] = A["label"][tio.DATA]
                return_dict["B_seg"] = B["label"][tio.DATA]

                # free transform history
                A.clear_history()
                B.clear_history()
                gc.collect()

            else:
                # No augmentation branch
                if self.opt.resize:
                    # Apply resizing to A, then apply the same transform to B
                    A = tio.Subject(
                        img=tio.ScalarImage(tensor=torch.from_numpy(A_orig)),
                        label=tio.LabelMap(
                            tensor=torch.from_numpy(AB_seg_orig).unsqueeze(0)
                        ),
                    )
                    A = self.pre_resize(A)
                    A_orig = A["img"][tio.DATA]
                    reproduce_transform = A.get_composed_history()
                    B = tio.Subject(
                        img=tio.ScalarImage(tensor=torch.from_numpy(B_orig))
                    )
                    B = reproduce_transform(B)
                    B_orig = B["img"][tio.DATA]

                    A.clear_history()
                    B.clear_history()
                    gc.collect()

                # Convert to torch tensors
                return_dict["A"] = torch.from_numpy(A_orig).float()
                return_dict["B"] = torch.from_numpy(B_orig).float()
                return_dict["A_seg"] = torch.from_numpy(
                    AB_seg_orig[np.newaxis, ...]
                ).float()
                return_dict["B_seg"] = torch.from_numpy(
                    AB_seg_orig[np.newaxis, ...]
                ).float()

                # Mask loading is not implemented
                if self.load_mask:
                    raise NotImplementedError("Mask loading is not implemented.")
                    img_keys = ["A", "B", "A_seg", "B_seg", "A_mask", "B_mask"]
                    return_dict["A_mask"] = hf[subj]["mask"][i][None, ...]
                    return_dict["B_mask"] = hf[subj]["mask"][j][None, ...]



        # Store the keys for reference
        return_dict["keys"] = img_keys

        # Apply random cropping if enabled and not resizing
        if self.crop_size > 0 and self.opt.isTrain and (not self.opt.resize):
            return_dict = random_crop(return_dict, img_keys, self.crop_size, self.dimension)

        return return_dict

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            The maximum of the number of subjects and the batch size.
        """
        return max(self.len, self.opt.batch_size)

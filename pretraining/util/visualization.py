import SimpleITK as sitk
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from torch.utils.tensorboard import SummaryWriter

sys.path.append("..")
from data.data_utils import renormalize_img

import random


def get_random_color(pastel_factor=0.5):
    return [
        (x + pastel_factor) / (1.0 + pastel_factor)
        for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]
    ]


def color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor=pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        if opt.display_id is None:
            self.display_id = (
                np.random.randint(100000) * 10
            )  # just a random display id
        else:
            self.display_id = opt.display_id
        self.name = opt.name
        self.ncols = opt.display_ncols

        self.dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.writer = SummaryWriter(self.dir)
        # self.create_tensorboard_connections()
        print(f"creating tensorboard at {self.dir}")
        # create a logging file to store training losses
        self.log_name = os.path.join(
            opt.checkpoints_dir, opt.name, "loss_log.txt"
        )
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(
                "================ Training Loss (%s) ================\n" % now
            )

    def display_current_results(self, visuals, epoch, board_name="images"):
        """Display current results on tensorboard;.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
        """
        ncols = self.ncols
        if ncols > 0:  # show all the images in one panel
            images = []
            titles = []
            for label, image in visuals.items():
                if image is None:
                    continue
                image_numpy = tensor2img(image, self.opt.display_slice)
                images.append(image_numpy)
                titles.append(label)
            tensorboard_vis(
                self.writer,
                epoch,
                board_name,
                images,
                self.ncols,
                cmaps="gist_gray",
                titles=titles,
            )
        else:
            for label, image in visuals.item():
                image_numpy = tensor2img(image, self.opt.display_slice)
                self.writer.add_image(
                    board_name + "/" + label,
                    renormalize_img(image_numpy),
                    epoch,
                    dataformats="HW",
                )

    def plot_current_losses(self, epoch, losses):
        """display the current losses on tensorboard: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if len(losses) == 0:
            return

        for name, loss in losses.items():
            if loss == 0:
                continue
            self.writer.add_scalar("loss/" + name, loss, epoch)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(
        self, total_iters, epoch, iters, losses, t_comp, t_data
    ):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = (
            "[Total iters: %d] (epoch: %d, iters: %d, time: %.3f, data: %.3f) "
            % (total_iters, epoch, iters, t_comp, t_data)
        )
        for k, v in losses.items():
            message += "%s: %f " % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write("%s\n" % message)  # save the message


def tensorboard_vis(
    summarywriter,
    step,
    board_name,
    img_list,
    num_row,
    cmaps,
    titles,
    resize=True,
):
    """
    :param summarywriter: tensorboard summary writer handle
    :param step:
    :param board_name: display name
    :param img_list: a list of images to show
    :param num_row: specify the number of row
    :param cmaps: specify color maps for each image ('gray' for MRI, i.e.)
    :param titles: specify each image title
    :param resize: whether resize the image to show
    :return:
    """
    num_figs = len(img_list)
    if num_figs == 0:
        return summarywriter
    fig = plt.figure()
    num_col = int(np.ceil(num_figs / num_row))
    print(
        "Visualizing %d images in %d row %d column"
        % (num_figs, num_row, num_col)
    )

    for i in range(num_figs):
        ax = plt.subplot(num_row, num_col, i + 1)
        tmp = img_list[i]
        if isinstance(cmaps, str):
            c = cmaps
        else:
            c = cmaps[i]

        vmin = tmp.min()
        vmax = tmp.max()
        print(titles[i], vmin, vmax)
        ax.imshow(tmp, cmap=c, vmin=vmin, vmax=vmax), plt.title(
            titles[i]
        ), plt.axis("off")

    summarywriter.add_figure(board_name, fig, step)
    return summarywriter


def tensor2img(tensor, slc):
    with torch.no_grad():
        if (
            len(tensor.size()) == 5
        ):  # input is 3D volume, sample one slice for visualization purpose
            try:
                x = tensor[0, 0, slc, ...].detach().cpu().numpy()
            except:
                slc = int(tensor.size(2) / 2)
                x = tensor[0, 0, slc, ...].detach().cpu().numpy()

        elif len(tensor.size()) == 4:
            x = tensor[0, 0, ...].detach().cpu().numpy()
        else:
            raise NotImplementedError
        return x


def vis_slc(tf_writer, board_name, tensor_list, tensor_names, slc, step):
    with torch.no_grad():
        for x, name in zip(tensor_list, tensor_names):
            if len(x.size()) == 5:
                x = x[0, 0, ..., slc].detach().cpu().numpy()
            elif len(x.size()) == 4:
                x = x[slc, 0, ...].detach().cpu().numpy()
            elif len(x.size()) == 3:
                x = x[0, ...].detach().cpu().numpy()
            else:
                raise NotImplementedError
            tf_writer.add_image(
                board_name + "/" + name,
                renormalize_img(x, name),
                step,
                dataformats="HW",
            )


def make_axes(sx, sy, sz):
    axes = {"ax": (0, sz), "sag": (2, sx), "cor": (1, sy)}  # sitk/np
    return axes


def take_slice_from_vol(file_path, view, sx, sy, sz, sqrt_pad=False, psize=-1):
    axes = make_axes(sx, sy, sz)
    flip = False
    # if file_path.endswith('.nii.gz'):
    vol = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
    ax = axes[view]
    print(vol.shape)
    slice = np.take(vol, ax[1], axis=ax[0])
    if flip:
        print("Flip image")
        slice = np.flip(slice, 0)
    return slice


def colorbar_wrapper(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def create_group_fig(
    img_list,
    cmaps,
    titles=None,
    annot=None,
    save_name=None,
    fig_size=None,
    num_row=None,
    vmin=None,
    vmax=None,
    colorbar=False,
    suptitle=None,
    dpi=100,
    format="eps",
    zoom=False,
    gamma_correction=1,
    verbose=True,
    fontsize=16,
    adjust=False,
):
    """
    :param img_list: a list of images to show
    :param cmaps: specify color maps for each image ('gray' for MRI, i.e.)
    :param num_row: specify the number of row
    :param titles: specify each image title
    :return:
    """
    num_figs = len(img_list)
    if num_row is None:
        num_row = int(np.sqrt(num_figs))
    num_col = int(np.ceil(num_figs / num_row))

    if zoom and num_row == 1:
        num_row *= 2

    if fig_size is None:
        h, w = img_list[0].shape[:2]
        fig_size = [w * num_col / 50.0, h * num_row / 50.0]

    if titles is None:
        titles = [str(i) for i in range(len(img_list))]
    plt.rcParams["figure.figsize"] = fig_size
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams["font.size"] = fontsize
    # rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':14})
    # rc('text', usetex=True)
    # new_rc_params = {'text.usetex': False,
    #                 "svg.fonttype": 'none'
    #                 }
    # plt.rcParams.update(new_rc_params)
    # fig = plt.figure()
    fig, axs = plt.subplots(num_row, num_col)
    # print(axs)
    if num_row == 1:
        axs = [axs]
    if num_col == 1:
        axs = [[ax] for ax in axs]
    # gs = gridspec.GridSpec(num_row,num_col)
    # gs.update(wspace=0, hspace=0.)
    if verbose:
        print(
            "Visualizing %d images in %d row %d column"
            % (num_figs, num_row, num_col)
        )
        print("Figure size", fig_size)

    for i in range(num_figs):
        plt.sca(axs[int(i / num_col)][int(i % num_col)])
        if isinstance(cmaps, str):
            c = cmaps
        else:
            c = cmaps[i]
        if gamma_correction != 1 and c == "gist_gray":
            from skimage import exposure

            img_list[i] = exposure.adjust_gamma(img_list[i], gamma_correction)
            vmax = None
        # plt.subplot(gs[i])#num_row, num_col, i + 1)#
        tmp = img_list[i]
        if vmax is not None:
            if isinstance(vmax, list):
                v_min, v_max = vmin[i], vmax[i]
            else:
                v_min, v_max = vmin, vmax
        else:
            v_min, v_max = tmp.min(), tmp.max()
        if verbose:
            print(titles[int(i % num_col)], v_min, v_max)
        im = plt.imshow(tmp, cmap=c, vmin=v_min, vmax=v_max, aspect="equal")
        if titles[i] is not None:
            plt.title(titles[i])
        plt.axis("off")
        if colorbar:
            colorbar_wrapper(im)
    if adjust:
        plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout()
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize)
    if save_name:
        s = time.time()
        # if dpi:
        #    plt.savefig(save_name,  format=format, dpi=dpi)
        # else:
        plt.savefig(save_name, format=format, dpi=dpi, bbox_inches="tight")
        e = time.time()
        if verbose:
            print("save figure %s in %.5f seconds" % (save_name, (e - s)))

    return fig

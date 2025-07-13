import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import pretraining_networks as networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. 
                                                    You can define one optimizer for each network.
            -- self.schedulers (scheduler list):    define and initialize schedulers.
                                                    You can define one scheduler for each optimizer.
                                                    
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain

        self.device = (
            torch.device("cuda") if self.gpu_ids else torch.device("cpu")
        )  # get device name: CPU or GPU

        print("running on device: {}".format(self.device))
        self.save_dir = os.path.join(
            opt.checkpoints_dir, opt.name
        )  # save all the checkpoints to save_dir
        self.loss_names = []
        self.model_names = []
        self.load_model_names = []
        self.visual_names = []
        self.feat_names = []
        self.optimizers = []
        self.image_paths = []
        self.visualizer = None
        # self.metric = 0  # used for learning rate policy 'plateau'
        if self.opt.isTrain:
            self.old_lr = opt.lr
        self.unfreeze_layers = []
        self.reset_params = False

    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        saved_dict = dict()

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals

            return grad_hook

        return hook_gen, saved_dict

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self, step):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [
                networks.get_scheduler(optimizer, opt)
                for optimizer in self.optimizers
            ]
        if (
            not self.isTrain
            or opt.continue_train
            or (
                opt.pretrained_name is not None
                and opt.pretrained_name != "None"
            )
        ):
            load_suffix = opt.epoch
            self.load_networks(load_suffix)

        self.print_networks(opt.verbose)

    def parallelize(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                if len(self.opt.gpu_ids) > 0 and self.opt.gpu_ids != -1:
                    if not isinstance(net, torch.nn.DataParallel):
                        print("Parallelize: {}".format(name))
                        setattr(
                            self,
                            "net" + name,
                            torch.nn.DataParallel(net, self.opt.gpu_ids),
                        )

    def data_dependent_initialize(self, data):
        pass

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def train(self):
        """Make models train mode during train time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.train()
                Norm = networks.get_norm_layer(self.opt.ndims, self.opt.normG)
                if name == "G" and len(self.unfreeze_layers) > 0:
                    for layer_id, layer in enumerate(net.model):
                        if str(
                            layer_id
                        ) not in self.unfreeze_layers and isinstance(
                            layer, Norm
                        ):
                            print("Freezing BN stats {}".format(layer_id))
                            layer.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for HTML visualization"""
        pass

    def update_learning_rate_decay(self, decay_gamma=0.5):
        lr = self.old_lr * decay_gamma
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        if self.opt.verbose:
            print("update learning rate: %f -> %f" % (self.old_lr, lr))
        self.old_lr = lr

    def update_learning_rate(self, metric=None):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == "plateau":
                scheduler.step(metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]["lr"]
        print("learning rate = %.7f" % lr)

    def get_current_visuals(self):
        """Return visualization images."""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, "loss_" + name)
                )  # float(...) works for both scalar tensor and float number
        return errors_ret

    def get_latent_features(self):
        feat_ret = OrderedDict()
        for name in self.feat_names:
            if isinstance(name, str):
                feat_ret[name] = getattr(self, name)
        return feat_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "%s_net_%s.pth" % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, "net" + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    if isinstance(net, torch.nn.DataParallel):
                        torch.save(net.module.cpu().state_dict(), save_path)
                    else:
                        torch.save(net.cpu().state_dict(), save_path)

                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        assert (
            len(self.load_model_names) > 0
        ), "found empty model list to load, please double check"
        print("Loading models {}".format(self.load_model_names))
        for name in self.load_model_names:
            if isinstance(name, str):
                load_filename = "%s_net_%s.pth" % (epoch, name)
                if self.opt.isTrain and (
                    self.opt.pretrained_name is not None
                    and self.opt.pretrained_name != "None"
                ):
                    print(self.opt.pretrained_name)
                    load_dir = os.path.join(
                        self.opt.checkpoints_dir, self.opt.pretrained_name
                    )
                else:
                    load_dir = self.save_dir

                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, "net" + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print("loading the model from %s" % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(
                    load_path, map_location=str(self.device)
                )
                print(list(state_dict.keys())[0])
                if "module" in list(state_dict.keys())[0]:
                    state_dict = self.convert_dict(state_dict)
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                try:
                    net.load_state_dict(state_dict)
                except:
                    model_dict = net.state_dict()
                    not_initialized = []
                    for k, v in state_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v
                        else:
                            print(
                                f"mismatch for {k} size: checkpoint size {v.size()}, model size {model_dict[k].size()}"
                            )
                            model_dict[k]
                            not_initialized.append(k)
                    assert (
                        len(not_initialized) == 1
                    ), "only allow for last layer mismatch, found more than 1 param with mismatched shape {}".format(
                        not_initialized
                    )
                    net.load_state_dict(model_dict)
                    print("Partial network initialized")

    def convert_dict(self, state_dict):
        new_dict = OrderedDict()
        for k in list(state_dict.keys()):
            new_dict[k.replace("module.", "")] = state_dict[k]
        return new_dict

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = 0
                num_trainable_params = 0
                for param in net.parameters():
                    # print(param.name, param.numel())
                    num_params += param.numel()
                    if param.requires_grad:
                        num_trainable_params += param.numel()
                if verbose:
                    print(net)
                print(
                    "[Network %s] Total number of parameters : %d"
                    % (name, num_params)
                )
        print("-----------------------------------------------")

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


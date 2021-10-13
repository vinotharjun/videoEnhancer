import torch.nn as nn
import torch
from ..basenets import BaseSequenceCritic
from .critic_blocks import CriticBlocks
class Critic(BaseSequenceCritic):
    """ Spatial discriminator
    """

    def __init__(self, in_nc=3, spatial_size=128,**kwargs):
        super(Critic, self).__init__()
        use_cond = kwargs("use_cond",False)
        # basic settings
        self.use_cond = use_cond  # whether to use conditional input
        mult = 2 if self.use_cond else 1
        tempo_range = 1

        # input conv
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc*tempo_range*mult, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        # discriminator block
        self.discriminator_block = CriticBlocks()  # /16

        # classifier
        self.dense = nn.Linear(256 * spatial_size // 16 * spatial_size // 16, 1)

    def forward(self, data, args_dict):
        out = self.forward_sequence(data, args_dict)
        return out

    def step(self, x):
        out = self.conv_in(x)
        out, feature_list = self.discriminator_block(out)

        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out, feature_list

    def forward_sequence(self, data, args_dict):
        # === set params === #
        n, t, c, hr_h, hr_w = data.size()
        data = data.view(n * t, c, hr_h, hr_w)

        # === build up inputs for net_D === #
        if self.use_cond:
            bi_data = args_dict['bi_data'].view(n * t, c, hr_h, hr_w)
            input_data = torch.cat([bi_data, data], dim=1)
        else:
            input_data = data

        # === classify === #
        pred = self.step(input_data)

        # construct output dict (nothing to return)
        ret_dict = {}

        return pred, ret_dict
from ..imports import *
from ..utils.net_utils import space_to_depth, backward_warp, get_upsampling_func
from ..utils.net_utils import initialize_weights
from ..utils.data_utils import float32_to_uint8
from .bricks.opticalflow_net import FNet
from .bricks.vsrnet_pixel import SRNet
from .basenets import *


class VideoEnhancer(BaseSequenceGenerator):
    """Frame-recurrent network: https://arxiv.org/abs/1801.04590"""

    def __init__(self, scale=4, **kwargs):
        super(VideoEnhancer, self).__init__()
        self.scale = scale
        in_nc = kwargs.get("in_nc", 3)
        out_nc = kwargs.get("out_nc", 3)
        nf = kwargs.get("nf", 64)
        nb = kwargs.get("nb", 16)
        

        # get upsampling function according to degradation type
        self.upsample_func = get_upsampling_func(self.scale)

        # define fnet & srnet
        self.fnet = FNet(in_nc)
        self.srnet = SRNet(in_nc, out_nc, nf, nb, scale=scale)
        self.apply(functools.partial(initialize_weights, scale=scale))

    def forward(self, lr_data):
        out = self.forward_sequence(lr_data)
        return out

    def forward_sequence(self, lr_data):
        """
        Parameters:
            :param lr_data: lr data in shape ntchw
        """

        n, t, c, lr_h, lr_w = lr_data.size()
        hr_h, hr_w = lr_h * self.scale, lr_w * self.scale

        # calculate optical flows
        lr_prev = lr_data[:, :-1, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_curr = lr_data[:, 1:, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_flow = self.fnet(lr_curr, lr_prev)  # n*(t-1),2,h,w

        # upsample lr flows
        hr_flow = self.scale * self.upsample_func(lr_flow)
        hr_flow = hr_flow.view(n, (t - 1), 2, hr_h, hr_w)

        # compute the first hr data
        hr_data = []
        hr_prev = self.srnet(
            lr_data[:, 0, ...],
            torch.zeros(
                n,
                (self.scale ** 2) * c,
                lr_h,
                lr_w,
                dtype=torch.float32,
                device=lr_data.device,
            ),
        )
        hr_data.append(hr_prev)

        # compute the remaining hr data
        for i in range(1, t):
            # warp hr_prev
            hr_prev_warp = backward_warp(hr_prev, hr_flow[:, i - 1, ...])

            # compute hr_curr
            hr_curr = self.srnet(
                lr_data[:, i, ...], space_to_depth(hr_prev_warp, self.scale)
            )

            # save and update
            hr_data.append(hr_curr)
            hr_prev = hr_curr

        hr_data = torch.stack(hr_data, dim=1)  # n,t,c,hr_h,hr_w

        # construct output dict
        ret_dict = {
            "hr_data": hr_data,  # n,t,c,hr_h,hr_w
            "hr_flow": hr_flow,  # n,t,2,hr_h,hr_w
            "lr_prev": lr_prev,  # n(t-1),c,lr_h,lr_w
            "lr_curr": lr_curr,  # n(t-1),c,lr_h,lr_w
            "lr_flow": lr_flow,  # n(t-1),2,lr_h,lr_w
        }

        return ret_dict

    def step(self, lr_curr, lr_prev, hr_prev):
        """
        Parameters:
            :param lr_curr: the current lr data in shape nchw
            :param lr_prev: the previous lr data in shape nchw
            :param hr_prev: the previous hr data in shape nc(4h)(4w)
        """

        # estimate lr flow (lr_curr -> lr_prev)
        lr_flow = self.fnet(lr_curr, lr_prev)

        # pad if size is not a multiple of 8
        pad_h = lr_curr.size(2) - lr_curr.size(2) // 8 * 8
        pad_w = lr_curr.size(3) - lr_curr.size(3) // 8 * 8
        lr_flow_pad = F.pad(lr_flow, (0, pad_w, 0, pad_h), "reflect")

        # upsample lr flow
        hr_flow = self.scale * self.upsample_func(lr_flow_pad)

        # warp hr_prev
        hr_prev_warp = backward_warp(hr_prev, hr_flow)

        # compute hr_curr
        hr_curr = self.srnet(lr_curr, space_to_depth(hr_prev_warp, self.scale))

        return hr_curr

    def infer_sequence(self, lr_data):
        """
        Parameters:
            :param lr_data: torch.FloatTensor in shape tchw
            :param device: torch.device
            :return hr_seq: uint8 np.ndarray in shape tchw
        """

        # set params
        tot_frm, c, h, w = lr_data.size()
        s = self.scale

        # forward
        hr_seq = []
        lr_prev = torch.zeros(
            1, c, h, w, dtype=torch.float32).to(lr_data.device)
        hr_prev = torch.zeros(1, c, s * h, s * w,
                              dtype=torch.float32).to(lr_data.device)

        with torch.no_grad():
            for i in range(tot_frm):
                lr_curr = lr_data[i: i + 1, ...].to(lr_data.device)
                hr_curr = self.step(lr_curr, lr_prev, hr_prev)
                lr_prev, hr_prev = lr_curr, hr_curr

                hr_frm = hr_curr.squeeze(0).cpu().numpy()  # chw|rgb|uint8

                hr_seq.append(float32_to_uint8(hr_frm))

        return np.stack(hr_seq).transpose(0, 2, 3, 1)  # thwc

    def generate_dummy_data(self, lr_size, device):
        c, lr_h, lr_w = lr_size
        s = self.scale

        # generate dummy input data
        lr_curr = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32).to(device)
        lr_prev = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32).to(device)
        hr_prev = torch.rand(1, c, s * lr_h, s * lr_w,
                             dtype=torch.float32).to(device)

        data_list = [lr_curr, lr_prev, hr_prev]
        return data_list

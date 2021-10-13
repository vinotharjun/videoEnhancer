from ..imports import *

from ..utils.data_utils import imresize


class BaseDataset(Dataset):
    def __init__(self, scale, gtsize, sequence_length, **kwargs):
        self.scale = scale
        self.gtsize = gtsize
        self.sequence_length = sequence_length

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    @staticmethod
    def augment_sequence(**kwargs):
        pass

    def validate_dimension(self, lq_tensor, gt_tensor):
        lq_size = lq_tensor.size()
        lr_h, lr_w = lq_size[-1], lq_size[-2]
        gt_size = gt_tensor.size()
        gt_h, gt_w = gt_size[-1], gt_size[-2]
        if gt_h != lr_h * self.scale or gt_w != lr_w * self.scale:
            lq_tensor = torch.nn.functional.interpolate(
                gt_tensor,
                size=(self.gtsize // self.scale, self.gtsize // self.scale),
                mode="bicubic",
                align_corners=True,
            )
            gt_tensor = torch.nn.functional.interpolate(
                gt_tensor,
                size=(
                    int(self.gtsize // self.scale) * self.scale,
                    int(self.gtsize // self.scale) * self.scale,
                ),
                mode="bicubic",
                align_corners=True,
            )
        return lq_tensor, gt_tensor

    def crop_tensors(self, lq_tensor, gt_tensor):
        _, _, H, W = lq_tensor.shape
        LQ_size = self.gtsize // self.scale
        rnd_h = np.random.randint(0, max(0, H - LQ_size))
        rnd_w = np.random.randint(0, max(0, W - LQ_size))
        lq_tensor = lq_tensor[:, :, rnd_h : rnd_h + LQ_size, rnd_w : rnd_w + LQ_size]
        rnd_h_GT, rnd_w_GT = int(rnd_h * self.scale), int(rnd_w * self.scale)
        gt_tensor = gt_tensor[
            :, :, rnd_h_GT : rnd_h_GT + self.gtsize, rnd_w_GT : rnd_w_GT + self.gtsize
        ]
        return lq_tensor, gt_tensor

    def moving_frame_maker(self, frm: torch.Tensor):
        gt_imgs, lq_imgs = [], []
        _, h, w = frm.size()
        # generate random moving parameters
        offsets = np.floor(np.random.uniform(-3.5, 4.5, size=(self.sequence_length, 2)))
        offsets = offsets.astype(np.int32)
        pos = np.cumsum(offsets, axis=0)
        min_pos = np.min(pos, axis=0)
        topleft_pos = pos - min_pos
        range_pos = np.max(pos, axis=0) - min_pos
        c_h, c_w = h - range_pos[0], w - range_pos[1]
        c_h = (c_h // self.scale) * self.scale
        c_w = (c_w // self.scale) * self.scale
        # generate frames
        for i in range(self.sequence_length):
            top, left = topleft_pos[i]
            gt_img: torch.Tensor = frm[:, top : top + c_h, left : left + c_w]
            if self.scale == 1:
                lq_img = torch.empty_like(gt_img).copy_(gt_img)
            else:
                lq_img = imresize(gt_img, 1 / self.scale, True)
            lq_imgs.append(lq_img.unsqueeze(0))
            gt_imgs.append(gt_img.unsqueeze(0))
        return lq_imgs, gt_imgs

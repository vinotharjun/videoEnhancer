from ..imports import *
from .basedataset import BaseDataset
from ..utils.data_utils import (
    get_images_from_path,
    list_dir,
    imresize_np,
    channel_convert,
    read_img,
)


class LQGT(BaseDataset):
    def __init__(
        self,
        gt_path,
        lq_path=None,
        gtsize=128,
        scale=2,
        sequence_length=10,
        **kwargs
    ):
        super(LQGT, self).__init__(
            scale=scale, gtsize=gtsize, sequence_length=sequence_length
        )

        self.gt_path = gt_path
        self.lq_path = lq_path
        self.moving_first_frame = kwargs.get("moving_first_frame",True)
        self.pairs = []
        if self.lq_path and type(lq_path) == str:
            lq_path = [lq_path]
        if type(gt_path) == str:
            gt_path = [gt_path]

        gt_files = []
        lq_files = []
        # [ [ video1 , video2 ,..],[video23,video24,..] ]
        for gt_dir in sorted(gt_path):
            # [video1,video2 ,...]
            single_video_frame_paths = [
                sorted(get_images_from_path(frame_path))
                for frame_path in sorted(list_dir(gt_dir))
            ]
            gt_files.extend(single_video_frame_paths)
        if self.lq_path:
            for lq_dir in sorted(lq_path):
                # [video1,video2 ,...]
                single_video_frame_paths = [
                    sorted(get_images_from_path(frame_path))
                    for frame_path in sorted(list_dir(lq_dir))
                ]
                lq_files.extend(single_video_frame_paths)

        if self.lq_path:
            assert len(gt_files) == len(lq_files)
        if not self.lq_path or len(lq_files) == 0:
            pairs = [([], i) for i in gt_files]
        else:
            pairs = [(lq, gt) for lq, gt in zip(lq_files, gt_files)]

        # final check
        validated_pairs = []
        min_sequence = np.inf
        for lq, gt in pairs:
            if len(lq) > 0 and len(lq) != len(gt):
                validated_pairs.append(([], gt))
            else:
                if len(gt) < min_sequence:
                    min_sequence = len(gt)
                validated_pairs.append((lq, gt))

        self.pairs = validated_pairs

        if self.sequence_length > min_sequence:
            self.sequence_length = min_sequence

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        lq_sequences, gt_sequences = self.pairs[item]
        if len(lq_sequences) == 0:
            lq_sequences = [None] * len(gt_sequences)
        gt_imgs, lq_imgs = [], []
        pairs = list(zip(lq_sequences, gt_sequences))
        if self.moving_first_frame and self.scale > 1 and np.random.uniform(0, 1) < 0.2:
            # load data
            frm = read_img(gt_sequences[0])  # chw|rgb|uint8
            frm = channel_convert(frm.shape[2], "RGB", [frm])[0]
            frm = torch.from_numpy(
                np.ascontiguousarray(np.transpose(frm, (2, 0, 1)))
            ).float()
            lq_imgs, gt_imgs = self.moving_frame_maker(frm)
        else:
            start = np.random.randint(0, len(pairs) - self.sequence_length)
            pairs = pairs[start : start + self.sequence_length]
            for lq_frame, gt_frame in pairs:
                lq_resize = False
                img_GT = read_img(gt_frame)
                img_GT = channel_convert(img_GT.shape[2], "RGB", [img_GT])[0]
                H, W, _ = img_GT.shape
                if H < self.gtsize or W < self.gtsize:
                    img_GT = cv2.resize(
                        np.copy(img_GT),
                        (self.gtsize, self.gtsize),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    lq_resize = True
                if self.scale == 1 and None in (self.lq_path, lq_frame):
                    pass
                elif (
                    None in (self.lq_path, lq_frame)
                    and self.scale > 1
                    and not lq_resize
                ):
                    img_LQ = imresize_np(img_GT, 1 / self.scale, True)
                    if img_LQ.ndim == 2:
                        img_LQ = np.expand_dims(img_LQ, axis=2)
                else:
                    img_LQ = read_img(lq_frame)
                    img_LQ = channel_convert(img_LQ.shape[2], "RGB", [img_LQ])[0]
                img_GT = torch.from_numpy(
                    np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
                ).float()
                if self.scale > 1 or (self.lq_path and lq_frame):
                    img_LQ = torch.from_numpy(
                        np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))
                    ).float()
                    lq_imgs.append(img_LQ.unsqueeze(0))
                gt_imgs.append(img_GT.unsqueeze(0))
        gt_imgs = torch.cat(gt_imgs, dim=0)
        if self.scale > 1 or self.lq_path:
            lq_imgs = torch.cat(lq_imgs, dim=0)
        else:
            lq_imgs = torch.empty_like(gt_imgs).copy_(gt_imgs)
        lq_imgs, gt_imgs = self.crop_tensors(lq_imgs, gt_imgs)
        lq_imgs, gt_imgs = self.validate_dimension(lq_imgs, gt_imgs)

        return { "lq":lq_imgs, "gt":gt_imgs }

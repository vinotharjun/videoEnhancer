from .lqgt import LQGT
import torch
def create_dataloader(
    dataset, phase, num_workers=4, batch_size=1, is_shuffle=True, sampler=None,**kwargs
):
    if phase == "train":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_shuffle,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=False,
            drop_last=False
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )


def create_dataset(mode, lq_path, gt_path, gtsize, scale,sequence_length,**kwargs):
    if mode == "LQGT":
        from .lqgt import LQGT as Dataset
    else:
        raise NotImplementedError("Dataset [{:s}] is not recognized.".format(mode))
    ds = Dataset(gt_path=gt_path,lq_path=lq_path,gtsize=gtsize,scale=scale,sequence_length=sequence_length,**kwargs)
    return ds
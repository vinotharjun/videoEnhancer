from time import process_time
from typing import List, Tuple, Union, AnyStr, Callable
from PIL import Image
import numpy as np
import torch
import mimetypes
import math
import os
import cv2
import os.path
import filecmp
from pathlib import Path
import gc
import torch.nn.functional as F

try:
    from .logger import logger
except ImportError:
    from logger import logger


# set random seed for reproducibility


import numbers
import numpy as np
from PIL import ImageFilter
import pexpect
import errno
import subprocess


def execute_subprocess(command):

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    out, err = process.communicate()
    exit_code = process.wait()
    if exit_code == 1:
        raise subprocess.SubprocessError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            f"execution of Command : {command} is failed \n Error is {err}",
        )


def execute_command(cmd):

    thread = pexpect.spawn(cmd)
    print("started %s" % cmd)
    cpl = thread.compile_pattern_list([pexpect.EOF, "waited (\d+)"])
    while True:
        i = thread.expect_list(cpl, timeout=None)
        if i == 0:  # EOF
            print("the sub process exited")
            break
        elif i == 1:
            waited_time = thread.match.group(1)
            print("the sub process waited %d seconds" % int(waited_time))
    thread.close()


class GaussianBlur:
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = np.array(img)
        img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), sigma)
        return Image.fromarray(img.astype(np.uint8))


def pathlib_changer(func: Callable) -> Callable:
    """decorator to change every string as pathlib object

    Args:
        func (Callable):functions to be wrapped

    Returns:
        Callable: wrapper of given function
    """

    def wrapper(*args, **kwargs):
        args_new = []
        for i in args:
            if type(i) == str:
                args_new.append(Path(i))
            else:
                args_new.append(i)
        args = tuple(args_new)
        for key, value in kwargs.items():
            if type(value) == str:
                kwargs[key] = Path(i)
            else:
                kwargs[key] = value
        return func(*args, **kwargs)

    return wrapper


def pathlib_list_changer(func: Callable) -> Callable:
    """decorator to change every string in list as pathlib object

    Args:
        func (Callable):functions to be wrapped

    Returns:
        Callable: wrapper of given function
    """

    def wrapper(self, *args, **kwargs):
        args_new = []
        for i in args:
            if type(i) == list:
                args_new.append([Path(j) for j in i])
            else:
                args_new.append(i)
        args = tuple(args_new)
        for key, value in kwargs.items():
            if type(value) == list:
                kwargs[key] = [Path(i) for i in value]
            else:
                kwargs[key] = value
        return func(self, *args, **kwargs)

    return wrapper


def extract_frames(
    video_path: Union[Path, AnyStr],
    frames_dir: Union[Path, AnyStr],
    overwrite: bool = False,
    start: int = -1,
    end: int = -1,
    every: int = 1,
) -> int:
    """
    Extract frames from a video using OpenCVs VideoCapture

    Args:
        video_path (Union[Path, AnyStr]): path of the video
        frames_dir (Union[Path, AnyStr]): the directory to save the frames
        overwrite (bool, optional):  to overwrite frames that already exist? Defaults to False.
        start (int, optional): start frame index. Defaults to -1.
        end (int, optional): end frame index . Defaults to -1.
        every (int, optional): frame spacing. Defaults to 1.

    Returns:
        int:  count of images saved
    """
    video_path = os.path.normpath(video_path)
    frames_dir = os.path.normpath(frames_dir)

    assert os.path.exists(video_path)

    capture = cv2.VideoCapture(video_path)

    if start < 0:
        start = 0
    if end < 0:
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)
    frame = start
    while_safety = 0
    saved_count = 0

    while frame < end:

        _, image = capture.read()

        if while_safety > 500:
            break

        if image is None:
            while_safety += 1
            continue

        if frame % every == 0:
            while_safety = 0
            save_path = os.path.join(frames_dir, "{:010d}.png".format(frame))
            if not os.path.exists(save_path) or overwrite:
                cv2.imwrite(save_path, image)
                del image
                gc.collect()
                saved_count += 1
        frame += 1
    capture.release()
    return saved_count


def is_image_file(filename: Union[Path, str]) -> bool:
    """check the given file is image

    Args:
        filename (Union[Path, str]): path of the file

    Returns:
        bool: True if the file is image ,false otherwise
    """
    return any(
        str(filename).endswith(extension)
        for extension in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    )


@pathlib_changer
def validate_directory_trees(
    dir1: Path, dir2: Path, ignore_extension=False, delete_if_empty: bool = False
) -> bool:
    """
    Compare two directories layout (excluding content of file) recursively. Files in each directory are
    assumed to be equal if their names and contents are equal.

    Args:
        dir1 (Path):First directory path
        dir2 (Path): Second directory path
        delete if empty (bool) : delete both directory if empty Default False
        ignore_extension(bool) : not to care about extensions? Default False

    Returns:
        bool: True if the directory trees are the same and
        there were no errors while accessing the directories or files,
        False otherwise.
    """
    if not dir1.exists() or not dir2.exists():
        return False
    dir1_files = sorted(os.listdir(dir1))
    dir2_files = sorted(os.listdir(dir2))
    if delete_if_empty and (not len(dir1_files) or not len(dir2_files)):
        remove_directory(dir1)
        remove_directory(dir2)
        return False
    all_files = set(dir1_files) & set(dir2_files)
    dirs_cmp = filecmp.dircmp(dir1, dir2)
    special_check = False
    if ignore_extension and (
        len(dirs_cmp.left_only) > 0 or len(dirs_cmp.right_only) > 0
    ):
        left_only = ["".join(str(i).split(".")[:-1]) for i in dirs_cmp.left_only]
        right_only = ["".join(str(i).split(".")[:-1]) for i in dirs_cmp.right_only]
        special_check = left_only != right_only

    elif not ignore_extension:
        special_check = len(dirs_cmp.left_only) > 0 or len(dirs_cmp.right_only) > 0
    if (
        len(dirs_cmp.common_files) != len(all_files)
        or special_check
        or len(dirs_cmp.funny_files) > 0
    ):

        return False
    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = dir1 / common_dir
        new_dir2 = dir2 / common_dir
        if not validate_directory_trees(new_dir1, new_dir2):
            return False
    return True


def is_video(file_name: Union[Path, str]) -> bool:
    """Function to check the given file is video or not

    Args:
        file_name (Union[Path, str]): Path of filebane

    Returns:
        bool: True if the file is video False otherwise
    """
    try:
        if mimetypes.guess_type(file_name)[0].startswith("video"):
            return True
    except AttributeError as e:
        return False
    return False


def run_fast_scandir(
    dir: Path, ext: list = []
) -> Tuple[List[AnyStr], List[AnyStr], List[AnyStr]]:
    """function to get all files of given directory
    Args:
        dir (Path): Path of the directory
        ext (list, optional): extentions to be included if all extensions needed [] have to be given. Defaults to [].

    Returns:
        Tuple[List[AnyStr], List[AnyStr], List[AnyStr]]: subfolder paths , files paths , directory group file paths
    """
    subfolders, files = [], []
    groups = []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext or len(ext) == 0:
                files.append(f.path)

    for dir in list(subfolders):
        _, f, _ = run_fast_scandir(dir, ext)
        groups.append(f)
    return subfolders, files, groups


def calculate_valid_crop_size(crop_size: int, upscale_factor: int) -> int:
    return crop_size - (crop_size % upscale_factor)


def gaussian_noise(image, std_dev):
    noise = np.rint(np.random.normal(loc=0.0, scale=std_dev, size=np.shape(image)))
    return Image.fromarray(np.clip(image + noise, 0, 255).astype(np.uint8))


@pathlib_changer
def remove_directory(directory: Union[Path, str]):
    """remove all files and directory recursively for given directory path

    Args:
        directory (Union[Path, str]): Path of the directory
    """
    if not os.path.exists(directory):
        return
    try:
        for item in directory.iterdir():
            if item.is_dir():
                remove_directory(item)
            else:
                item.unlink()
        directory.rmdir()
    except OSError as e:
        logger.error(
            f"Error while deleting {directory}.ErrorType OsError Error is : {e}"
        )
        return


def cubic(x: torch.Tensor) -> torch.Tensor:
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
    ) * (((absx > 1) * (absx <= 2)).type_as(absx))


def calculate_weights_indices(
    in_length: int,
    out_length: int,
    scale: int,
    kernel_width: int,
    antialiasing: bool,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(
        0, P - 1, P
    ).view(1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


@torch.no_grad()
def imresize(
    img: torch.Tensor, scale: int, antialiasing: bool = True, **kwargs
) -> torch.Tensor:
    output_even = kwargs.get("maintain_even_size", False)
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round

    in_C, in_H, in_W = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    if output_even:
        if out_H & 1:
            out_H -= 1
        if out_W & 1:
            out_W -= 1
    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel_width, antialiasing
    )
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel_width, antialiasing
    )
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = (
            img_aug[0, idx : idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        )
        out_1[1, i, :] = (
            img_aug[1, idx : idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        )
        out_1[2, i, :] = (
            img_aug[2, idx : idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        )

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :, idx : idx + kernel_width].mv(weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :, idx : idx + kernel_width].mv(weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :, idx : idx + kernel_width].mv(weights_W[i])

    return torch.clamp(out_2, 0, 1)

import argparse
import os
import random
from typing import List
import yaml
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm, trange
from pathlib import Path
import numpy as np
from functools import partial
from multiprocessing import Pool
import multiprocessing
import gc
import torch
import errno
from generate_utils import (
    pathlib_changer,
    remove_directory,
    imresize,
    is_video,
    is_image_file,
    run_fast_scandir,
    extract_frames,
    validate_directory_trees,
    GaussianBlur,
    execute_subprocess,
)

try:
    from .logger import logger
except ImportError:
    from logger import logger

# create directory
#  extract frames (with or without resizing)
# extract lr frames (with or without resizing ) (with or without compressing)


class DataGenerator:
    def __init__(self, **kwargs):
        self.cleanup_factor = kwargs.get("cleanup_factor", 1)
        self.upscale_factor = kwargs.get("upscale_factor", 4)
        self.compress = kwargs.get("compress", False)
        self.need_blur = kwargs.get("need_blur", False)
        self.ignore_directory = kwargs.get("ignore_directory", False)
        self.ignore_video = kwargs.get("ignore_video", False)
        self.output_directory = kwargs.get("output_directory", None)
        self.validate_dir = kwargs.get("validate_directory", False)
        if not self.output_directory:
            logger.warning("output directory not given")
        else:
            self.output_directory = Path(self.output_directory)
        self.extensions = kwargs.get("extensions", [])
        self.process_with_ffmpeg = kwargs.get("process_video_with_ffmpeg", True)
        self.video_frame_limit = kwargs.get("video_frame_limit", 100)
        self.generate_lr = kwargs.get("generate_lr", True)
        self.save_hr = kwargs.get("save_hr", True)
        self.ignore_if_exists = kwargs.get("ignore_if_exists", True)
        if not self.generate_lr and not self.save_hr:
            self.save_hr = True
        self.create_lr_hr()

    def create_lr_hr(self):
        if self.output_directory is None:
            return
        self.hr_dir = self.output_directory / "HR"
        self.video_temp_path = self.output_directory / "VIDEOS"
        if self.generate_lr:
            lr_name = f"LRx{self.upscale_factor}"
            if self.need_blur:
                lr_name += "-blurred"
            if self.compress:
                lr_name += "-compressed"
            self.lr_dir = self.output_directory / Path(lr_name)
            self.lr_dir.mkdir(parents=True, exist_ok=True)
        self.hr_dir.mkdir(parents=True, exist_ok=True)
        self.video_temp_path.mkdir(parents=True, exist_ok=True)

    @pathlib_changer
    def read_and_save(self, video_file: Path) -> Path:

        save_frames = Path(
            f"{self.video_temp_path}/{video_file.parent.name}/{video_file.stem}"
        )
        save_frames.mkdir(parents=True, exist_ok=True)
        start = random.randint(0, 100)
        end = start + self.video_frame_limit
        count = extract_frames(
            video_file, save_frames, self.ignore_if_exists, start=start, end=end
        )
        return save_frames

    @pathlib_changer
    def process_video(self, video_path: Path):
        if self.ignore_video or not is_video(video_path):
            return
        if self.process_with_ffmpeg:
            hr_path = self.hr_dir / video_path.stem
            lr_path = self.lr_dir / video_path.stem
            try:
                hr_command = f"ffmpeg  -i {video_path} -vf  scale='ceil(iw*{self.cleanup_factor}):ceil(ih*{self.cleanup_factor})' -frames:v {self.video_frame_limit} -y {hr_path}/%010d.png"
                if self.save_hr:
                    hr_path.mkdir(parents=True, exist_ok=True)
                    execute_subprocess(hr_command)
                # hr clean and save
                if self.generate_lr:
                    lr_path.mkdir(parents=True, exist_ok=True)
                    # lr resize and compress (if needed ) and store
                    if self.compress:
                        quality = np.random.randint(1, 4) * 5
                        temp_file = Path(f"{lr_path}/temp.mp4")
                        temp_command = f"ffmpeg -i '{hr_path}/%010d.png' -vf scale='ceil(iw/{self.upscale_factor}):ceil(ih/{self.upscale_factor})'  -q:v {quality} -y {temp_file}"
                        lr_command = f"ffmpeg -i {temp_file}  -y {lr_path}/%010d.jpg"
                        execute_subprocess(temp_command)
                        execute_subprocess(lr_command)
                        temp_file.unlink()
                    else:
                        lr_command = f"ffmpeg -i '{hr_path}/%010d.png' -vf scale='ceil(iw/{self.upscale_factor}):ceil(ih/{self.upscale_factor})' -y {lr_path}/%010d.png"
                        execute_subprocess(lr_command)
            except Exception as e:
                remove_directory(hr_path)
                remove_directory(lr_path)
                raise e
        else:
            saved_frame_path = self.read_and_save(video_path)
            self.process_directory(saved_frame_path)

    @pathlib_changer
    def process_directory(self, directory_path: Path, **kwargs) -> None:
        if self.ignore_directory:
            return
        child = kwargs.get("child", False)
        _, files, groups = run_fast_scandir(directory_path, self.extensions)
        if len(groups) == 0:
            groups.append(files)
        for group in tqdm(
            groups, desc=f"generating data for {directory_path}", leave=not child
        ):
            self.process_group(group, parent_directory=directory_path, child=True)

    def process_group(self, image_array: List[str], **kwargs) -> None:

        if self.output_directory is None:
            raise NotImplementedError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                "otuput directory is not specified",
            )
        image_array: Path = [Path(i) for i in image_array]
        child = kwargs.get("child", False)
        parent_directory = kwargs.get("parent_directory", None)

        if parent_directory is not None:

            group_name = set(image_array[0].parent.parts) - set(
                parent_directory.parent.parts
            )
            group_name = "-".join(list(group_name))
        else:
            group_name = (
                f"{image_array[0].parent.parent.name}-{image_array[0].parent.name}"
            )
        hr_path = self.hr_dir / group_name
        lr_path = self.lr_dir / group_name
        if (
            self.ignore_if_exists
            and os.path.exists(hr_path)
            and os.path.exists(lr_path)
        ):
            logger.info(f"leaving {group_name} is already exists")
            return
        if self.save_hr:
            hr_path.mkdir(parents=True, exist_ok=True)
        if self.generate_lr:
            lr_path.mkdir(parents=True, exist_ok=True)
        quality = np.random.randint(3, 10) * 10
        blurrer = GaussianBlur(kernel_size=23)
        file: Path
        try:
            for file in tqdm(
                image_array,
                desc=f"generating groups {group_name}",
                leave=not child,
            ):
                if not is_image_file(file) or is_video(file):
                    self.process_video(file)
                    continue
                group_name = Path(group_name)
                input_img = Image.open(file).convert("RGB")
                input_img = TF.to_tensor(input_img)
                input_img = imresize(
                    input_img, self.cleanup_factor, True, maintain_even_size=True
                )
                hr_img_path = f"{hr_path/file.stem}.png"
                if self.save_hr:
                    TF.to_pil_image(input_img).save(str(hr_img_path), "PNG")
                if self.generate_lr:
                    resize_img = imresize(input_img, 1.0 / self.upscale_factor, True)
                    resize_img = TF.to_pil_image(resize_img)
                    if self.need_blur:
                        resize_img = blurrer(resize_img)
                    if self.compress:
                        resize_img.save(
                            str(f"{lr_path/file.stem}.jpg"),
                            "JPEG",
                            quality=quality,
                        )
                    else:
                        resize_img.save(str(f"{lr_path/file.stem}.png"), "PNG")
                del resize_img, input_img
                gc.collect()
            if self.validate_dir:
                validated = validate_directory_trees(
                    lr_path, hr_path, ignore_extension=True, delete_if_empty=True
                )
                if not validated:
                    remove_directory(lr_path)
                    remove_directory(hr_path)
        except Exception as e:
            remove_directory(lr_path)
            remove_directory(hr_path)
            raise e

    def process_single(self, path: Path, **kwargs) -> None:
        try:
            child = kwargs.get("child", False)
            if path.is_file():
                self.process_video(path)
                # logger.info(f"data generated successfully for video {path}")
            elif path.is_dir():
                self.process_directory(path, child=child)
                # logger.info(f"data generated successfully for directory{path}")
        except Exception as e:
            logger.error(
                f"error while processing {path} An exception of type {type(e).__name__} occurred. Arguments:\n{e.args}"
            )

    @pathlib_changer
    def process(self, dataset_path: Path, execute_parallel=True) -> None:
        if not os.path.exists(dataset_path):

            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), dataset_path
            )

        all_paths = [
            dataset_path / single_path for single_path in os.listdir(dataset_path)
        ]
        if len(all_paths) == 0:
            return
        if self.output_directory is None:
            self.output_directory = Path(f"../datasets/{dataset_path.stem}")
            self.create_lr_hr()

        if execute_parallel:
            cpu_count = multiprocessing.cpu_count()
            if cpu_count > 10:
                cpu_count = 8
            else:
                cpu_count = 2
            pool = Pool(processes=cpu_count)
            result_list_tqdm = []
            for result in tqdm(
                pool.imap_unordered(
                    func=partial(self.process_single, child=True),
                    iterable=all_paths,
                ),
                total=len(all_paths),
            ):
                result_list_tqdm.append(result)
            assert len(result_list_tqdm) == len(all_paths)
        else:
            for path in tqdm(all_paths):
                self.process_single(path)
        remove_directory(self.video_temp_path)


if __name__ == "__main__":
    yml_file_path = Path("./datapaths.yml")
    dataset_key = "datasets"
    PATHS = {}
    datasets = []
    generate_configs = {}
    try:
        with open(yml_file_path, "r") as stream:
            PATHS = yaml.safe_load(stream)
            if "generate_configs" in PATHS:
                generate_configs = PATHS["generate_configs"]
            datasets = list(PATHS[dataset_key].keys())
    except FileNotFoundError as e:
        logger.error(f"No yml file found  : {yml_file_path}")
        exit()
    except (TypeError, KeyError) as e:
        logger.warning(
            f"no dataset information provided  under key '{dataset_key}' in {yml_file_path} please check once"
        )
    parser = argparse.ArgumentParser(
        description="Apply the trained model to create a dataset"
    )

    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        choices=datasets,
        help="selecting different datasets",
    )

    parser.add_argument(
        "--cleanup_factor",
        default=1,
        type=int,
        help="downscaling factor for image cleanup",
    )

    parser.add_argument(
        "--upscale_factor",
        default=4,
        type=int,
        choices=[1, 2, 3, 4],
        help="super resolution upscale factor",
    )

    parser.add_argument(
        "--compress",
        default=False,
        type=bool,
        choices=[True, False],
        help="compress image",
    )

    parser.add_argument(
        "--need",
        default=False,
        type=bool,
        choices=[True, False],
        help="blur image",
    )

    parser.add_argument(
        "--ignore_directory",
        default=False,
        type=bool,
        choices=[True, False],
        help="flag to ignore directory",
    )
    parser.add_argument(
        "--ignore_video",
        default=False,
        type=bool,
        choices=[True, False],
        help="flag to ignore video files",
    )
    parser.add_argument(
        "--generate_lr",
        default=False,
        type=bool,
        choices=[True, False],
        help="flag to ignore generation of low resolution/quality",
    )
    parser.add_argument(
        "--video_frame_limit",
        default=100,
        type=int,
        help="to choose how many frames to extract from video , if your data doesnt have any video files please ignore this flag",
    )
    parser.add_argument(
        "--save_hr",
        default=True,
        type=bool,
        choices=[True, False],
        help="flag for not to ignore saving of HR frames,the default value is true,if you give generate_lr as false then it will automatically set as true ",
    )
    parser.add_argument(
        "--output_directory",
        default="./data",
        type=str,
        help="give the output directory",
    )
    parser.add_argument(
        "--input_from_yml",
        default=True,
        type=bool,
        choices=[True, False],
        help="specify the config from yml file",
    )
    opt = parser.parse_args()
    if not opt.input_from_yml or generate_configs == {}:
        generate_configs = vars(opt)
    d = DataGenerator(**generate_configs)
    d.process("../../../video-datasets/plaground/test/")

"""Core Code for all frame interpolations"""
import os
import cv2
import sys
import torch
import numpy as np
from tqdm import tqdm
from imageio import imsave
from typing import Callable
from ..utils.simple_utils import sortable_float_index
from ..utils.file_utils import split_filepath

"""==========import from our code=========="""
sys.path.append(".")
from ..benchmark.utils.padder import InputPadder  # pylint: disable=import-error

class Interpolate:
    """Encapsulate logic for the Frame Interpolation feature"""

    STD_MIDFRAME = 0.5

    def __init__(self, model, log_fn: Callable | None):
        self.model = model
        self.log_fn = log_fn
        self.output_paths = []

    def create_between_frame(
        self,
        before_filepath: str,
        after_filepath: str,
        middle_filepath: str,
        time_step: float = STD_MIDFRAME,
    ):
        """Invoke the Frame Interpolation feature"""
        # code borrowed from EMA-VFI/demo_2x.py
        I0 = cv2.imread(before_filepath)
        I2 = cv2.imread(after_filepath)

        I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.0).unsqueeze(0)
        I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.0).unsqueeze(0)

        padder = InputPadder(I0_.shape, divisor=32)
        I0_, I2_ = padder.pad(I0_, I2_)

        model = self.model["model"]
        TTA = self.model["TTA"]

        mid = (
            padder.unpad(
                model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA, timestep=time_step)
            )[0]
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
            * 255.0
        ).astype(np.uint8)
        images = [I0[:, :, ::-1], mid[:, :, ::-1], I2[:, :, ::-1]]
        imsave(middle_filepath, images[1])
        self.output_paths.append(middle_filepath)
        self.log("create_between_frame() saved " + middle_filepath)

    def create_between_frames(
        self,
        before_filepath: str,
        after_filepath: str,
        middle_filepath: str,
        frame_count: int,
    ):
        """Invoke the Frame Interpolation feature for multiple between frames
        frame_count is the number of new frames, ex: 8X interpolation, 7 new frames are needed
        """
        # code borrowed from EMA-VFI/demo_2x.py
        I0 = cv2.imread(before_filepath)
        I2 = cv2.imread(after_filepath)

        I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.0).unsqueeze(0)
        I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.0).unsqueeze(0)

        padder = InputPadder(I0_.shape, divisor=32)
        I0_, I2_ = padder.pad(I0_, I2_)

        model = self.model["model"]
        TTA = self.model["TTA"]
        set_count = 2 if frame_count < 1 else frame_count + 1

        output_path, filename, extension = split_filepath(middle_filepath)
        output_filepath = os.path.join(output_path, f"{filename}@0.0.png")
        images = [I0[:, :, ::-1]]
        imsave(output_filepath, images[0])
        self.output_paths.append(output_filepath)

        preds = model.multi_inference(
            I0_,
            I2_,
            TTA=TTA,
            time_list=[(i + 1) * (1.0 / set_count) for i in range(set_count - 1)],
            fast_TTA=TTA,
        )
        for pred in preds:
            images.append(
                (
                    padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0
                ).astype(np.uint8)[:, :, ::-1]
            )
        images.append(I2[:, :, ::-1])

        pbar_desc = "Frame"
        for index, image in enumerate(tqdm(images, desc=pbar_desc)):
            # for index, image in enumerate(images):
            if 0 < index < len(images) - 1:
                time = sortable_float_index(index / set_count)
                output_filepath = os.path.join(output_path, f"{filename}@{time}.png")
                imsave(output_filepath, image)
                self.output_paths.append(output_filepath)

        output_filepath = os.path.join(output_path, f"{filename}@1.0.png")
        imsave(output_filepath, images[-1])
        self.output_paths.append(output_filepath)

    def log(self, message):
        """Logging"""
        if self.log_fn:
            self.log_fn(message)

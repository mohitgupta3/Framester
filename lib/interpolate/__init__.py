import os
import sys
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from imageio import imsave
from tqdm import tqdm

sys.path.append(".")
from typing import Callable

from tqdm import tqdm

import lib.config as cfg  # pylint: disable=import-error
from lib.interpolate.model.Trainer import Model

from ..utils.file_utils import split_filepath
from ..utils.simple_utils import max_steps, sortable_float_index

class InputPadder:
    """Pads images such that dimensions are divisible by divisor"""

    def __init__(self, dims, divisor=16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]

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

class DeepInterpolate:
    """Encapsulates logic for the Frame Interpolation feature"""

    def __init__(
        self, interpolater: Interpolate, time_step: bool, log_fn: Callable | None
    ):
        self.interpolater = interpolater
        self.time_step = time_step
        self.log_fn = log_fn
        self.split_count = 0
        self.frame_register = []
        self.progress = None
        self.output_paths = []

    def split_frames(
        self,
        before_filepath,
        after_filepath,
        num_splits,
        output_path,
        base_filename,
        progress_label="Frame",
        continued=False,
        resynthesis=False,
    ):
        """Invoke the Frame Interpolation feature"""
        self.init_frame_register()
        self.reset_split_manager(num_splits)
        num_steps = max_steps(num_splits)
        self.init_progress(num_splits, num_steps, progress_label)
        output_filepath_prefix = os.path.join(output_path, base_filename)

        if self.time_step:
            self.interpolater.create_between_frames(
                before_filepath, after_filepath, output_filepath_prefix, num_steps
            )
            for path in self.interpolater.output_paths:
                self.register_frame(path)
            self.interpolater.output_paths = []
        else:
            self._set_up_outer_frames(
                before_filepath, after_filepath, output_filepath_prefix
            )
            self._recursive_split_frames(0.0, 1.0, output_filepath_prefix)
        self._integerize_filenames(output_path, base_filename, continued, resynthesis)
        self.close_progress()

    def _set_up_outer_frames(self, before_file, after_file, output_filepath_prefix):
        """Start with the original frames at 0.0 and 1.0"""
        img0 = cv2.imread(before_file)
        img1 = cv2.imread(after_file)

        # create outer 0.0 and 1.0 versions of original frames
        before_index, after_index = 0.0, 1.0
        before_file = self.indexed_filepath(output_filepath_prefix, before_index)
        after_file = self.indexed_filepath(output_filepath_prefix, after_index)

        cv2.imwrite(before_file, img0)
        self.register_frame(before_file)
        self.log("copied " + before_file)

        cv2.imwrite(after_file, img1)
        self.register_frame(after_file)
        self.log("copied " + after_file)

    def _recursive_split_frames(
        self, first_index: float, last_index: float, filepath_prefix: str
    ):
        """Create a new frame between the given frames, and re-enter to split deeper"""
        if self.enter_split():
            mid_index = first_index + (last_index - first_index) / 2.0
            first_filepath = self.indexed_filepath(filepath_prefix, first_index)
            last_filepath = self.indexed_filepath(filepath_prefix, last_index)
            mid_filepath = self.indexed_filepath(filepath_prefix, mid_index)

            self.interpolater.create_between_frame(
                first_filepath, last_filepath, mid_filepath
            )
            self.register_frame(mid_filepath)
            self.step_progress()

            # deal with two new split regions
            self._recursive_split_frames(first_index, mid_index, filepath_prefix)
            self._recursive_split_frames(mid_index, last_index, filepath_prefix)
            self.exit_split()

    def _integerize_filenames(self, output_path, base_name, continued, resynthesis):
        """Keep the interpolated frame files with an index number for sorting"""
        file_prefix = os.path.join(output_path, base_name)
        frame_files = self.sorted_registered_frames()
        num_files = len(frame_files)
        num_width = len(str(num_files))
        index = 0
        self.output_paths = []

        for file in frame_files:
            if resynthesis and (index == 0 or index == num_files - 1):
                # if a resynthesis process, keep only the interpolated frames
                os.remove(file)
                self.log("resynthesis - removed uneeded " + file)
            elif continued and index == 0:
                # if a continuation from a previous set of frames, delete the first frame
                # to maintain continuity since it's duplicate of the previous round last frame
                os.remove(file)
                self.log("continuation - removed uneeded " + file)
            else:
                new_filename = file_prefix + str(index).zfill(num_width) + ".png"
                os.replace(file, new_filename)
                self.output_paths.append(new_filename)
                self.log("renamed " + file + " to " + new_filename)
            index += 1

    def reset_split_manager(self, num_splits: int):
        """Start managing split depths of a new round of searches"""
        self.split_count = num_splits

    def enter_split(self):
        """Enter a split depth if allowed, returns True if so"""
        if self.split_count < 1:
            return False
        self.split_count -= 1
        return True

    def exit_split(self):
        """Exit the current split depth"""
        self.split_count += 1

    def init_frame_register(self):
        """Start managing interpolated frame files for a new round of searches"""
        self.frame_register = []

    def register_frame(self, filepath: str):
        """Register a found frame file"""
        self.frame_register.append(filepath)

    def sorted_registered_frames(self):
        """Return a sorted list of the currently registered found frame files"""
        return sorted(self.frame_register)

    def init_progress(self, num_splits, _max, description):
        """Start managing progress bar for a new found of searches"""
        if num_splits < 2:
            self.progress = None
        else:
            self.progress = tqdm(range(_max), desc=description)

    def step_progress(self):
        """Advance the progress bar"""
        if self.progress:
            self.progress.update()
            self.progress.refresh()

    def close_progress(self):
        """Done with the progress bar"""
        if self.progress:
            self.progress.close()

    # filepath prefix representing the split position while splitting
    def indexed_filepath(self, filepath_prefix, index):
        """Filepath prefix representing the split position while splitting"""
        float_index = sortable_float_index(index, fixed_width=True)
        return filepath_prefix + f"{float_index}.png"

    def log(self, message):
        """Logging"""
        if self.log_fn:
            self.log_fn(message)

class InterpolateSeries:
    """Encapsulate logic for the Video Inflation feature"""

    def __init__(self, deep_interpolater: DeepInterpolate, log_fn: Callable | None):
        self.deep_interpolater = deep_interpolater
        self.log_fn = log_fn

    def interpolate_series(
        self,
        file_list: list,
        output_path: str,
        num_splits: int,
        base_filename: str,
        offset: int = 1,
    ):
        """Invoke the Video Inflation feature"""
        file_list = sorted(file_list)
        count = len(file_list)
        num_width = len(str(count))
        pbar_desc = "Frames" if num_splits < 2 else "Total"

        for frame in tqdm(range(count - offset), desc=pbar_desc, position=0):
            # for other than the first around, the duplicated real "before" frame is deleted for
            # continuity, since it's identical to the "after" from the previous round
            continued = frame > 0

            # if the offset is > 1 treat this as a resynthesis of frames
            # and inform the deep interpolator to not keep the real frames
            resynthesis = offset > 1

            before_file = file_list[frame]
            after_file = file_list[frame + offset]

            # if a resynthesis, start the file numbering at 1 to match the restored frame
            # if an offset other than 2 is used, the frame numbers won't generally match
            base_index = frame + (1 if resynthesis else 0)
            filename = base_filename + "[" + str(base_index).zfill(num_width) + "]"

            inner_bar_desc = f"Frame #{frame}"

            self.deep_interpolater.split_frames(
                before_file,
                after_file,
                num_splits,
                output_path,
                filename,
                progress_label=inner_bar_desc,
                continued=continued,
                resynthesis=resynthesis,
            )

    def log(self, message):
        """Logging"""
        if self.log_fn:
            self.log_fn(message)

class InterpolateEngine:
    """Singleton class encapsulating the EMA-VFI engine and related logic"""
    # model should be "ours" or "ours_small", or your own trained model
    # gpu_ids is for *future use*
    # if use_time_step is True "_t" is appended to the model name
    def __new__(cls, model : str, gpu_ids : str, use_time_step : bool=False):
        if not hasattr(cls, 'instance'):
            cls.instance = super(InterpolateEngine, cls).__new__(cls)
            cls.instance.init(model, gpu_ids, use_time_step)
        elif cls.instance.model_name != model or cls.instance.use_time_step != use_time_step:
            cls.instance = super(InterpolateEngine, cls).__new__(cls)
            cls.instance.init(model, gpu_ids, use_time_step)
        return cls.instance

    def init(self, model : str, gpu_ids: str, use_time_step):
        """Iniitalize the class by calling into EMA-VFI code"""
        gpu_id_array = self.init_device(gpu_ids)
        self.model_name = model
        self.use_time_step = use_time_step
        self.model = self.init_model(model, gpu_id_array, use_time_step)

    def init_device(self, gpu_ids : str):
        """for *future use*"""
        str_ids = gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            _id = int(str_id)
            if _id >= 0:
                gpu_ids.append(_id)
        # for *future use*
        # if len(gpu_ids) > 0:
        #     torch.cuda.set_device(gpu_ids[0])
        # cudnn.benchmark = True
        return gpu_ids

    def init_model(self, model, gpu_id_array, use_time_step):
        """EMA-VFI code from demo_2x.py"""
        # for *future use*
        # device = torch.device('cuda' if len(gpu_id_array) != 0 else 'cpu')
        '''==========Model setting=========='''
        TTA = True
        if model == 'ours_small':
            TTA = False
            cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small' + ("_t" if use_time_step else "")
            cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
                F = 16,
                depth = [2, 2, 2, 2, 2]
            )
        else:
            cfg.MODEL_CONFIG['LOGNAME'] = 'ours' + ("_t" if use_time_step else "")
            cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
                F = 32,
                depth = [2, 2, 2, 4, 4]
            )
        model = Model(-1)
        model.load_model()
        model.eval()
        model.device()
        return {"model" : model, "TTA" : TTA}

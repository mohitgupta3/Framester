"""Video Inflation Core Code"""
import argparse
from typing import Callable
from tqdm import tqdm
from interpolate_engine import InterpolateEngine
from interpolate import Interpolate
from deep_interpolate import DeepInterpolate
from webui_utils.simple_log import SimpleLog
from webui_utils.file_utils import create_directory, get_files

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


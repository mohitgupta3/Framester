import glob
import os
from collections import namedtuple
import cv2

import numpy as np
import torch
import yaml
from gradio import (
    Blocks,
    Row,
    Column,
    Slider,
    Tab,
    PlayableVideo,
    Button,
    Radio,
    HighlightedText,
    Textbox,
    Checkbox,
    HTML,
    Progress,
    update,
)
from tqdm import tqdm

from lib.interpolate import DeepInterpolate, Interpolate, InterpolateEngine
from lib.upscale.upscale_series import UpscaleSeries
from lib.utils.file_utils import (
    count_images_in_directory,
    create_directory,
    get_files,
    remove_directory,
)
from lib.utils.simple_icons import SimpleIcons


class SimpleLog:
    """
    Class to manage simple logging to the console
    Collect log message and optionally print to the console
    """

    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.messages = []

    def log(self, message: str) -> None:
        """Add a new log message"""
        self.messages.append(message)
        if self.verbose:
            print(message)

    def reset(self):
        self.messages = []
        self.log("log messages cleared")


log = SimpleLog(verbose=False)


class SimpleConfig:
    """Manage a simple YAML config file"""

    def __new__(cls, path: str = "lib/config.yaml"):
        if not hasattr(cls, "instance"):
            cls.instance = super(SimpleConfig, cls).__new__(cls)
            cls.instance.init(path)
        return cls.instance

    def init(self, path: str):
        """Load the config"""
        with open(path, encoding="utf-8") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

    def get(self, key: str):
        """Get top-level config value"""
        return self.config[key]

    def config_obj(self):
        """Create an Object.Properties version of the config"""
        return namedtuple("ConfigObj", self.config.keys())(*self.config.values())


config = SimpleConfig(path="lib/config.yaml").config_obj()

"""Construct the Gradio Blocks UI"""
app_header = HTML("🎬 Framester", elem_id="appheading")
with Blocks(
    analytics_enabled=True, title="Framester", css=config.user_interface["css_file"]
) as app:
    input_video_info = {}

    def get_video_info(input_video):
        global input_video_info

        input_video_Obj = cv2.VideoCapture(input_video)
        input_video_info["FPS"] = int(input_video_Obj.get(cv2.CAP_PROP_FPS))
        input_video_info["height"] = int(input_video_Obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_video_info["width"] = int(input_video_Obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_video_info["frame_count"] = int(
            input_video_Obj.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        return input_video_info

    def update_splits_info(input_video_path: str, num_splits: float):
        global input_video_info

        def max_steps(num_splits: int) -> int:
            """Computing the count of work steps needed based on the number of splits"""
            # Before splitting, there's one existing region between the before and after frames.
            # Each split doubles the number of regions.
            # Work steps = the final number of regions - the existing region.
            return 2**num_splits - 1

        """Given a count of splits/search depth/search precision, compute the count of work steps"""
        total_steps = int(max_steps(num_splits))

        input_video_info = get_video_info(input_video_path)

        return {
            info_output_vi: update(value=total_steps),
            info_new_fps: update(value=input_video_info["FPS"] * (total_steps + 1)),
        }

    def convert_png_to_mp4(input_images_dir: str, output_path: str):
        global input_video_info
        """Convert button handler"""
        frame_rate = count_images_in_directory(input_images_dir) // (
            input_video_info["frame_count"] / input_video_info["FPS"]
        )
        # Retrieve the list of image filenames
        image_filenames = glob.glob(f"{input_images_dir}/*.png")
        # Sort the image filenames to ensure proper ordering
        image_filenames.sort()
        # Read the first image to get its dimensions
        first_image = cv2.imread(image_filenames[0])
        height, width, _ = first_image.shape
        # Define the video codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )  # Choose the appropriate codec for your desired output format
        video_output = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

        # Iterate over the image filenames and write each frame to the video
        for filename in image_filenames:
            frame = cv2.imread(filename)
            video_output.write(frame)

        # Release the VideoWriter and close any open windows
        video_output.release()
        return True

    def create_presentation_video(
        input_video: str,
        high_fps_video_path: str,
        output_path: str,
        presentation_mode: str,
    ):
        def create_separation_matrix(n_rows, n_cols):
            separation_matrix = []
            if n_rows == n_cols:
                for i in range(n_rows):
                    row = []
                    for j in range(n_cols):
                        if i == j:
                            row.append(0)
                        else:
                            row.append(1)
                    separation_matrix.append(row)
            else:
                if n_cols > n_rows:
                    overhead = (n_cols - n_rows) // 2
                    start_pt, end_pt = (0, overhead), (n_rows, n_cols - overhead)
                else:
                    overhead = (n_rows - n_cols) // 2
                    start_pt, end_pt = (overhead, 0), (n_rows - overhead, n_cols)

                separation_matrix = [[1] * (n_cols) for _ in range(n_rows)]

                x1, y1 = start_pt
                x2, y2 = end_pt

                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                x_step = 1 if x1 < x2 else -1
                y_step = 1 if y1 < y2 else -1

                error = dx - dy

                while x1 != x2 or y1 != y2:
                    separation_matrix[x1][y1] = 0
                    double_error = 2 * error

                    if double_error > -dy:
                        error -= dy
                        x1 += x_step

                    if double_error < dx:
                        error += dx
                        y1 += y_step

            # The pixels on the left of diagonal should be represented as 2
            lower_than_diagonal = False  # Only applicable if n_rows > n_cols
            for row_idx, row in enumerate(separation_matrix):
                if 0 in row:
                    lower_than_diagonal = True
                    conv_to_2 = False
                    for col_idx, col in enumerate(row):
                        if conv_to_2:
                            separation_matrix[row_idx][col_idx] = 2
                        elif col == 0:
                            conv_to_2 = True
                else:
                    separation_matrix[row_idx] = [
                        1 if lower_than_diagonal else 2 for _ in range(len(row))
                    ]

            return np.array(separation_matrix)

        def put_highlighted_text(
            input_arrey,
            text,
            position,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=1.0,
            color=(170, 255, 0),
            thickness=2,
            highlight_color=(0, 0, 0),
            alpha=0.5,
        ):
            # Convert the color to BGR format
            color = tuple(reversed(color))
            highlight_color = tuple(reversed(highlight_color))

            # Get the text size
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            # Calculate the bounding box coordinates
            x, y = position
            x2, y2 = x + text_width + 10, y - text_height - 10

            # Create a transparent overlay image
            overlay = input_arrey.copy()
            cv2.rectangle(
                overlay, (x, y), (x2 + 10, y2 - 10), highlight_color, cv2.FILLED
            )
            cv2.addWeighted(overlay, alpha, input_arrey, 1 - alpha, 0, input_arrey)

            # Put the highlighted text on the image
            cv2.putText(
                input_arrey,
                text,
                (x + 10, y - 10),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

            return input_arrey

        # Open the videos
        video1 = cv2.VideoCapture(input_video)
        video2 = cv2.VideoCapture(high_fps_video_path)

        # Get the properties of the videos
        fps1 = video1.get(cv2.CAP_PROP_FPS)
        fps2 = video2.get(cv2.CAP_PROP_FPS)
        width1 = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate the frame repetition ratio for the lower FPS video
        repetition_ratio = int(round(fps2 / fps1))
        repeat_frame = repetition_ratio

        if presentation_mode.startswith("Separate"):
            # If video width is greater than video height then merge videos vertically
            merge_top_down = width1 > height1

            # Create a VideoWriter object to save the merged video
            merged_video = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps2,
                (width1, height1 * 2 + 2)
                if merge_top_down
                else (width1 * 2 + 2, height1),
            )

            ret1, frame1 = video1.read()

            while True:
                # Read frames from video 2
                ret2, frame2 = video2.read()
                if not ret2:
                    break

                # Create a White canvas to merge the frames side by side
                merged_frame = 255 * np.ones(
                    shape=(2 + height1 * 2, width1, 3)
                    if merge_top_down
                    else (height1, 2 + width1 * 2, 3),
                    dtype=np.uint8,
                )

                # Repeat frames of video 1
                if repeat_frame == 0:
                    repeat_frame = repetition_ratio - 1
                    # Read frames from video 1
                    ret1, frame1 = video1.read()
                    if not ret1:
                        break
                else:
                    repeat_frame = repeat_frame - 1

                frame1 = put_highlighted_text(
                    input_arrey=frame1, text=f"Original: {fps1} FPS", position=(25, 50)
                )
                frame2 = put_highlighted_text(
                    input_arrey=frame2,
                    text=f"Enhanced: {fps2+1} FPS",
                    position=(25, 50),
                )

                if merge_top_down:
                    (
                        merged_frame[:height1, :width1],
                        merged_frame[height1 + 2 :, :width1],
                    ) = (frame1, frame2)
                else:
                    merged_frame[:, :width1], merged_frame[:, width1 + 2 :] = (
                        frame1,
                        frame2,
                    )

                # Write the merged frame to the output video
                merged_video.write(merged_frame)

        elif presentation_mode.startswith("Diagonal"):
            separation_matrix = create_separation_matrix(n_rows=height1, n_cols=width1)

            # Create a VideoWriter object to save the merged video
            merged_video = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps2, (width1, height1)
            )

            # Read frames from the two videos and merge them
            ret1, frame1 = video1.read()

            text_loc_x = (
                width1
                - cv2.getTextSize(
                    f"Enhanced: {fps2} FPS", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                )[0]
                - 25
            )
            while True:
                merged_frame = 255 * np.ones(
                    shape=(height1, width1, 3),
                    dtype=np.uint8,
                )

                # Read frames from video 2
                ret2, frame2 = video2.read()
                if not ret2:
                    break

                # Repeat frames of video 1
                if repeat_frame == 0:
                    repeat_frame = repetition_ratio - 1
                    # Read frames from video 1
                    ret1, frame1 = video1.read()
                    if not ret1:
                        break
                else:
                    repeat_frame = repeat_frame - 1

                for row_idx in range(len(separation_matrix)):
                    for col_idx, col in enumerate(separation_matrix[row_idx]):
                        merged_frame[row_idx][col_idx] = (
                            frame1[row_idx][col_idx]
                            if col == 1
                            else frame2[row_idx][col_idx]
                            if col == 2
                            else [255, 255, 255]
                        )

                merged_frame = put_highlighted_text(
                    input_arrey=merged_frame,
                    text=f"Original: {fps1} FPS",
                    position=(25, height1 - 50),
                )
                merged_frame = put_highlighted_text(
                    input_arrey=merged_frame,
                    text=f"Enhanced: {fps2} FPS",
                    position=(text_loc_x, 50),
                )
                # Write the merged frame to the output video
                merged_video.write(merged_frame)

        elif presentation_mode.startswith("Vertical"):
            merged_video = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps2,
                (width1 + 1, height1),
            )

            # Read frames from the two videos and merge them
            ret1, frame1 = video1.read()

            while True:
                merged_frame = 255 * np.ones(
                    shape=(height1, width1 + 1, 3),
                    dtype=np.uint8,
                )

                # Read frames from video 2
                ret2, frame2 = video2.read()
                if not ret2:
                    break

                # Repeat frames of video 1
                if repeat_frame == 0:
                    repeat_frame = repetition_ratio - 1
                    # Read frames from video 1
                    ret1, frame1 = video1.read()
                    if not ret1:
                        break
                else:
                    repeat_frame = repeat_frame - 1

                merged_frame[:, : width1 // 2], merged_frame[:, 1 + width1 // 2 :] = (
                    frame1[:, : width1 // 2],
                    frame2[:, width1 // 2 :],
                )

                merged_frame = put_highlighted_text(
                    input_arrey=merged_frame,
                    text=f"Original: {fps1} FPS",
                    position=(25, 50),
                )
                merged_frame = put_highlighted_text(
                    input_arrey=merged_frame,
                    text=f"Enhanced: {fps2} FPS",
                    position=(25 + width1 // 2, 50),
                )
                # Write the merged frame to the output video
                merged_video.write(merged_frame)

        elif presentation_mode.startswith("Horizontal"):
            merged_video = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps2,
                (width1, height1 + 1),
            )

            # Read frames from the two videos and merge them
            ret1, frame1 = video1.read()

            while True:
                merged_frame = 255 * np.ones(
                    shape=(height1 + 1, width1, 3),
                    dtype=np.uint8,
                )

                # Read frames from video 2
                ret2, frame2 = video2.read()
                if not ret2:
                    break

                # Repeat frames of video 1
                if repeat_frame == 0:
                    repeat_frame = repetition_ratio - 1
                    # Read frames from video 1
                    ret1, frame1 = video1.read()
                    if not ret1:
                        break
                else:
                    repeat_frame = repeat_frame - 1

                (
                    merged_frame[: height1 // 2, :width1],
                    merged_frame[height1 // 2 + 1 :, :width1],
                ) = (
                    frame1[: height1 // 2, :width1],
                    frame2[height1 // 2 :, :width1],
                )

                merged_frame = put_highlighted_text(
                    input_arrey=merged_frame,
                    text=f"Original: {fps1} FPS",
                    position=(25, 50),
                )
                merged_frame = put_highlighted_text(
                    input_arrey=merged_frame,
                    text=f"Enhanced: {fps2} FPS",
                    position=(25, height1 // 2 + 50),
                )
                # Write the merged frame to the output video
                merged_video.write(merged_frame)

        # Release the resources
        video1.release()
        video2.release()
        merged_video.release()

    def reset_ui():
        global input_video_info

        remove_directory("temp_input_frames")
        remove_directory("temp_interpolated_frames")
        remove_directory("temp_upscaled_frames")

        return {
            controle_panel: update(visible=False),
            interpolated_video: update(visible=False, value=None),
            info_original_fps: update(value=0),
            info_new_fps: update(value=0),
            interpolate_button: update(visible=False),
            video_info: update(visible=False, value=[]),
        }

    def upload_video(input_video_path: str, num_splits: int):
        def resize_input_frames():
            max_width, max_height = 1080, 720
            frames_name_list = glob.glob("temp_input_frames/*.png")
            original_height, original_width, _ = cv2.imread(frames_name_list[0]).shape
            # Check if the image is larger than the desired dimensions
            if original_width > max_width or original_height > max_height:
                # Calculate the aspect ratio
                aspect_ratio = min(
                    max_width / original_width, max_height / original_height
                )

                # Calculate the new dimensions
                new_width = int(original_width * aspect_ratio)
                new_height = int(original_height * aspect_ratio)

                for frame_idx in tqdm(
                    range(len(frames_name_list)), desc="Resizing Frame"
                ):
                    img = cv2.imread(frames_name_list[frame_idx])
                    img = cv2.resize(
                        src=img,
                        dsize=(new_width, new_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    cv2.imwrite(frames_name_list[frame_idx], img)
                return True
            return False

        input_video_info = get_video_info(input_video_path)

        create_directory("temp_input_frames")
        # Convert Input Video  frames to PNG files

        cap = cv2.VideoCapture(os.path.join(os.getcwd(), input_video_path))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if ret:
                output_filename = os.path.join(
                    "temp_input_frames", f"image{str(frame_idx).zfill(5)}.png"
                )
                frame_idx += 1

                cv2.imwrite(output_filename, frame)
            else:
                break

        # Check if the video frames are larger than 1280 x 720 pixels
        # If so, resize the frames...
        if resize_input_frames():
            convert_png_to_mp4(
                input_images_dir="temp_input_frames", output_path=input_video_path
            )

        return {
            input_video: update(value=input_video_path),
            controle_panel: update(visible=True),
            info_original_fps: update(value=input_video_info["FPS"]),
            info_new_fps: update(value=input_video_info["FPS"] * (num_splits + 1)),
            interpolate_button: update(visible=True),
            video_info: update(
                visible=True,
                value=[(k, str(v)) for k, v in input_video_info.items()],
            ),
        }

    def video_inflation(
        input_video_path: str,
        num_splits: float,
        output_mode: str,
        presentation_mode: str,
        progress=Progress(track_tqdm=True),
    ):
        remove_directory("temp_interpolated_frames")
        create_directory("temp_interpolated_frames")

        """Invoke the Video Inflation feature"""
        progress(0, desc="Frames")

        engine = InterpolateEngine(
            config.model, config.gpu_ids, use_time_step=config.use_time_step
        )
        deep_interpolater = DeepInterpolate(
            interpolater=Interpolate(engine.model, log.log),
            time_step=config.use_time_step,
            log_fn=log.log,
        )
        file_list = sorted(get_files("temp_input_frames", extension="png"))
        output_path = "temp_interpolated_frames"
        base_filename = "if_"
        count = len(file_list)
        num_width = len(str(count))
        offset = 1

        for frame in progress.tqdm(list(range(count - offset))):
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

            inner_bar_desc = f"Frame-{base_index}"

            deep_interpolater.split_frames(
                before_file,
                after_file,
                num_splits,
                output_path,
                filename,
                progress_label=inner_bar_desc,
                continued=continued,
                resynthesis=resynthesis,
            )

        del deep_interpolater
        del engine

        torch.cuda.empty_cache()
        # Convert Interpolated frames to video of original length
        convert_png_to_mp4(
            input_images_dir="temp_interpolated_frames", output_path="interpolated.mp4"
        )

        return_Obj = {}
        if output_mode == "Inplace":
            return_Obj[input_video] = update(visible=True, value="interpolated.mp4")
        elif output_mode == "Separate":
            return_Obj[interpolated_video] = update(
                visible=True, value="interpolated.mp4"
            )
        else:
            create_presentation_video(
                input_video_path, "interpolated.mp4", "merged.mp4", presentation_mode
            )
            return_Obj[interpolation_presentation_video] = update(
                visible=True, value="merged.mp4"
            )

        return_Obj[video_info] = update(
            visible=True, value=[(k, str(v)) for k, v in input_video_info.items()]
        )

        return return_Obj

    def video_upscale(input_video: str, upscale_factor: int, inplace: bool):
        remove_directory("temp_upscaled_frames")

        output_path = "temp_input_frames" if inplace else "temp_upscaled_frames"

        create_directory(output_path)

        model_name = config.realesrgan_settings["model_name"]
        gpu_ips = config.gpu_ids
        fp32 = config.realesrgan_settings["fp32"]

        upscaler = UpscaleSeries(
            model_name, gpu_ips, fp32, tiling=0, tile_pad=0, log_fn=log.log
        )

        file_list = get_files(path="temp_input_frames", extension="png")
        output_basename = "image"
        output_dict = upscaler.upscale_series(
            file_list, output_path, upscale_factor, output_basename, output_type="png"
        )

        file_list = [key for key in output_dict.keys() if output_dict[key] == None]

        if file_list:
            upscaler = UpscaleSeries(
                model_name,
                gpu_ips,
                fp32,
                tiling=config.realesrgan_settings["tiling"],
                tile_pad=config.realesrgan_settings["tile_pad"],
                log_fn=log.log,
            )
            output_dict = upscaler.upscale_series(
                file_list,
                output_path,
                upscale_factor,
                output_basename,
                output_type="png",
            )

        # Convert Interpolated frames to video of original length
        convert_png_to_mp4(
            input_images_dir="temp_input_frames" if inplace else "temp_upscaled_frames",
            output_path=input_video if inplace else "upscaled.mp4",
        )

        get_video_info(input_video=input_video if inplace else "output.mp4")

        return (
            {
                input_video: update(value=input_video),
                upscaled_video: update(value="upscaled.mp4", visible=True),
            }
            if inplace
            else {
                input_video: update(value=input_video),
                upscaled_video: update(value=None, visible=False),
            }
        )

    app_header.render()

    HTML(
        SimpleIcons.INCREASING
        + "Increase the number of video frames to any depth and upscale them to any size",
        elem_id="tabheading",
    )

    remove_directory("temp_input_frames")
    remove_directory("temp_interpolated_frames")
    remove_directory("temp_upscaled_frames")

    with Column() as video_panel:
        with Row():
            with Column(scale=3):
                input_video = PlayableVideo(label="Low FPS Input video", format="mp4")
                interpolated_video = PlayableVideo(
                    label="Video with increased FPS", visible=False, interactive=False
                )
                interpolation_presentation_video = PlayableVideo(
                    label="Original vs Interpolated Video",
                    visible=False,
                    interactive=False,
                )
                upscaled_video = PlayableVideo(
                    label="Upscaled Video", visible=False, interactive=False
                )
            with Column(scale=2, visible=False) as controle_panel:
                video_info = HighlightedText(
                    value=[(k, v) for k, v in input_video_info.items()],
                    label="Input video Info",
                )
                with Tab(label="Interpolate"):
                    with Column(variant="panel"):
                        with Row():
                            splits_input_slider = Slider(
                                value=1,
                                minimum=1,
                                maximum=10,
                                step=1,
                                label="Split Count",
                                info="Number of splits b/w two frames",
                            )
                            info_output_vi = Textbox(
                                value="1",
                                label="Interpolations",
                                max_lines=1,
                                interactive=False,
                                info="Interpolations per frame",
                            )
                        with Row():
                            info_original_fps = Textbox(
                                value="0",
                                label="Original FPS",
                                max_lines=1,
                                interactive=False,
                                info="Original FPS of the input video",
                            )
                            info_new_fps = Textbox(
                                value="0",
                                label="Updated FPS",
                                max_lines=1,
                                interactive=False,
                                info="FPS after interpolations",
                            )
                    with Column():
                        output_mode = Radio(
                            choices=["Inplace", "Separate", "Presentation"],
                            value="Inplace",
                            label="Output mode",
                            info="Result Video placement",
                            interactive=True,
                        )
                        presentation_mode = Radio(
                            choices=[
                                "Separate (Auto align)",
                                "Diagonal Split",
                                "Vertical Split",
                                "Horizontal Split",
                            ],
                            value="None",
                            label="Presentation mode",
                            info="How the presentation should appear?",
                            interactive=True,
                            visible=False,
                        )
                    interpolate_button = Button(
                        "Interpolate Video " + SimpleIcons.ROBOT,
                        variant="primary",
                    )

                with Tab(label="Upscale"):
                    with Column(variant="panel"):
                        scale_input = Slider(
                            value=4.0,
                            minimum=1.0,
                            maximum=8.0,
                            step=0.05,
                            label="Frame Upscale Factor",
                            interactive=True,
                        )
                        upscale_inplace = Checkbox(
                            value=False,
                            label="Inplace",
                            interactive=True,
                            info="Replace the original video with Upscaled video",
                        )
                    upscale_button = Button(
                        "Upscale Video " + SimpleIcons.ROCKET, variant="primary"
                    )

    """Event Handelers"""
    # For any update in input video component
    input_video.clear(
        reset_ui,
        inputs=[],
        outputs=[
            input_video,
            controle_panel,
            info_original_fps,
            info_new_fps,
            interpolated_video,
            interpolate_button,
            video_info,
        ],
        show_progress=True,
    )

    input_video.upload(
        upload_video,
        inputs=[input_video, splits_input_slider],
        outputs=[
            input_video,
            controle_panel,
            info_original_fps,
            info_new_fps,
            interpolate_button,
            video_info,
        ],
        show_progress=True,
    )

    # If Interpolate button is clicked
    interpolate_button.click(
        video_inflation,
        inputs=[input_video, splits_input_slider, output_mode, presentation_mode],
        outputs=[
            input_video,
            interpolated_video,
            interpolation_presentation_video,
            video_info,
        ],
    )

    def process_change_in_output_mode(output_mode_title):
        return {
            presentation_mode: update(
                visible=True if output_mode_title == "Presentation" else False
            )
        }

    output_mode.change(
        process_change_in_output_mode, inputs=[output_mode], outputs=[presentation_mode]
    )

    # If Upscale Button is clicked
    upscale_button.click(
        video_upscale,
        inputs=[input_video, scale_input, upscale_inplace],
        outputs=[input_video, upscaled_video],
    )

    # If Split Count slider is used
    splits_input_slider.change(
        update_splits_info,
        inputs=[input_video, splits_input_slider],
        outputs=[info_output_vi, info_new_fps],
        show_progress=False,
    )

app.launch(
    inbrowser=config.auto_launch_browser,
    server_name=config.server_name,
    server_port=config.server_port,
    prevent_thread_lock=False,
    share=False,
    debug=False,
    enable_queue=True,
)

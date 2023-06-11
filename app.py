from tqdm import tqdm
from collections import namedtuple
from signal import SIGINT, signal

import cv2
import glob
import numpy as np
import gradio as gr
import yaml

from lib.interpolate.deep_interpolate import DeepInterpolate
from lib.interpolate.interpolate import Interpolate
from lib.interpolate.interpolate_engine import InterpolateEngine
from lib.upscale.upscale_series import UpscaleSeries
from lib.utils.file_utils import create_directory, remove_directory, get_files, count_images_in_directory
from lib.utils.simple_icons import SimpleIcons
from lib.utils.video_utils import MP4toPNG as _MP4toPNG


"""Class to manage simple logging to the console"""
class SimpleLog:
    """Collect log message and optionally print to the console"""

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
app_header = gr.HTML("🎬 Framester", elem_id="appheading")
with gr.Blocks(analytics_enabled=True, title="Framester", theme=config.user_interface["theme"], css=config.user_interface["css_file"]) as app:
    input_video_info = {}

    def get_video_info(input_video):
        global input_video_info

        input_video_Obj                 = cv2.VideoCapture(input_video)
        input_video_info["FPS"]         = int(input_video_Obj.get(cv2.CAP_PROP_FPS))
        input_video_info["height"]      = int(input_video_Obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_video_info["width"]       = int(input_video_Obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_video_info["frame_count"] = int(input_video_Obj.get(cv2.CAP_PROP_FRAME_COUNT))

        return input_video_info

    def reset_ui(input_video: str, num_splits: int):
        global input_video_info
        
        remove_directory("temp_input_frames")
        remove_directory("temp_interpolated_frames")
        remove_directory("temp_upscaled_frames")

        if input_video == None:
            return {
                controle_panel: gr.update(visible=False),
                interpolated_video: gr.update(visible=False, value=None),
                info_original_fps: gr.update(value=0),
                info_new_fps: gr.update(value=0),
                interpolate_button: gr.update(visible=False),
                video_info: gr.update(visible=False, value=[])
            }
        else:
            input_video_info = get_video_info(input_video)

            create_directory("temp_input_frames")
            # Convert Input Video  frames to PNG files
            _MP4toPNG(
                input_path=input_video, filename_pattern="image%05d.png", frame_rate=input_video_info["FPS"], output_path="temp_input_frames"
            )
            
            return {
                controle_panel: gr.update(visible=True),
                info_original_fps: gr.update(value=input_video_info["FPS"]),
                info_new_fps: gr.update(value=input_video_info["FPS"]*(num_splits+1)),
                interpolate_button: gr.update(visible=True),
                video_info: gr.update(value=[(k, str(v)) for k, v in input_video_info.items()])
            }
            
    def update_splits_info(input_video_path: str, num_splits : float):
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
            info_output_vi: gr.update(value=total_steps),
            info_new_fps: gr.update(value=input_video_info["FPS"]*(total_steps+1))
        }
    
    def convert_png_to_mp4(input_images_dir: str, output_path: str):
        """Convert button handler"""
        frame_rate = count_images_in_directory(input_images_dir)//(input_video_info["frame_count"]/input_video_info["FPS"])
        # Retrieve the list of image filenames
        image_filenames = glob.glob(f"{input_images_dir}/*.png")
        # Sort the image filenames to ensure proper ordering
        image_filenames.sort()
        # Read the first image to get its dimensions
        first_image = cv2.imread(image_filenames[0])
        height, width, _ = first_image.shape
        # Define the video codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Choose the appropriate codec for your desired output format
        video_output = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

        # Iterate over the image filenames and write each frame to the video
        for filename in image_filenames:
            frame = cv2.imread(filename)
            video_output.write(frame)

        # Release the VideoWriter and close any open windows
        video_output.release()
        return True

    def merge_videos(input_video: str, high_fps_video_path: str, output_path: str):
        # Open the videos
        video1 = cv2.VideoCapture(input_video)
        video2 = cv2.VideoCapture(high_fps_video_path)

        # Get the properties of the videos
        fps1 = video1.get(cv2.CAP_PROP_FPS)
        fps2 = video2.get(cv2.CAP_PROP_FPS)
        width1 = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width2 = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate the frame repetition ratio for the lower FPS video
        repetition_ratio = int(round(fps2 / fps1))
        
        # Create a VideoWriter object to save the merged video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_path, fourcc, fps2, (width1 + 2 + width2, max(height1, height2)))

        repeat_frame = repetition_ratio-1
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()
        while True:
            # Read frames from video 2
            ret2, frame2 = video2.read()
            if not ret2:
                break
            
            # Create a black canvas to merge the frames side by side
            merged_frame = 255 * np.ones(shape=(max(height1, height2), width1 + 2 + width2, 3), dtype=np.uint8)
            # Resize the frames to have the same height
            frame1_resized = cv2.resize(frame1, (int(width1 * (height2 / height1)), height2))
            merged_frame[:, :width1] = frame1_resized

            # Repeat frames of video 1
            if repeat_frame == 0:
                repeat_frame = repetition_ratio-1
                # Read frames from video 1
                ret1, frame1 = video1.read()
                if not ret1:
                    break
            else:
                repeat_frame = repeat_frame-1
            
            # Resize the frames to have the same height
            frame1_resized = cv2.resize(frame1, (int(width1 * (height2 / height1)), height2))
            frame1_resized = cv2.putText(frame1_resized, f"Original: {fps1} FPS", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 170), 1, cv2.LINE_AA)
            merged_frame[:, :width1] = frame1_resized

            # Resize the frames to have the same height
            frame2_resized = cv2.resize(frame2, (int(width2 * (height1 / height2)), height1))
            frame2_resized = cv2.putText(frame2_resized, f"Enhanced: {fps2+1} FPS", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 170), 1, cv2.LINE_AA)
            merged_frame[:, width1+2:] = frame2_resized

            # Write the merged frame to the output video
            output_video.write(merged_frame)

        # Release the resources
        video1.release()
        video2.release()
        output_video.release()

        print("Merged video saved successfully.")

    def video_inflation(input_video_path: str, num_splits: float, output_mode: str, progress=gr.Progress(track_tqdm=True)):        
        remove_directory("temp_interpolated_frames")
        create_directory("temp_interpolated_frames")
        
        """Invoke the Video Inflation feature"""
        progress(0, desc="Frames")

        engine            = InterpolateEngine(config.model, config.gpu_ids, use_time_step=config.use_time_step)
        deep_interpolater = DeepInterpolate(interpolater=Interpolate(engine.model, log.log), time_step=config.use_time_step, log_fn=log.log)
        file_list         = sorted(get_files("temp_input_frames", extension="png"))
        output_path       = "temp_interpolated_frames"
        base_filename     = "if_"
        count             = len(file_list)
        num_width         = len(str(count))
        offset            = 1

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
                before_file, after_file, num_splits, output_path, filename, progress_label=inner_bar_desc, continued=continued, resynthesis=resynthesis,
            )

        del deep_interpolater
        del engine

        # Convert Interpolated frames to video of original length
        convert_png_to_mp4(
            input_images_dir="temp_interpolated_frames", 
            output_path="interpolated.mp4"
        )
        
        return_Obj = {}
        if output_mode == "Inplace":
            return_Obj[input_video] = gr.update(visible=True, value="interpolated.mp4")
        elif output_mode == "Separate":
            return_Obj[interpolated_video] = gr.update(visible=True, value="interpolated.mp4")
        elif output_mode == "Presentation":
            merge_videos(input_video_path, "interpolated.mp4", "merged.mp4")
            return_Obj[interpolation_presentation_video] = gr.update(visible=True, value="merged.mp4")
        
        return_Obj[video_info] = gr.update(visible=True, value=[(k, str(v)) for k, v in input_video_info.items()])

        return return_Obj

    def video_upscale(input_video: str, upscale_factor: int, inplace: bool):
        remove_directory("temp_upscaled_frames")
        
        output_path = "temp_input_frames" if inplace else "temp_upscaled_frames" 
        
        create_directory(output_path)

        model_name = config.realesrgan_settings["model_name"]
        gpu_ips    = config.gpu_ids
        fp32       = config.realesrgan_settings["fp32"]
        
        upscaler = UpscaleSeries(
            model_name, gpu_ips, fp32, tiling=0, tile_pad=0, log_fn=log.log
        )

        file_list = get_files(path="temp_input_frames", extension="png")
        output_basename = "image"
        output_dict = upscaler.upscale_series(
            file_list, 
            output_path, 
            upscale_factor, 
            output_basename, 
            output_type="png"
        )

        file_list = [
            key for key in output_dict.keys() if output_dict[key] == None
        ]

        if file_list:
            upscaler = UpscaleSeries(
                model_name, gpu_ips, fp32, tiling=config.realesrgan_settings["tiling"], tile_pad=config.realesrgan_settings["tile_pad"], log_fn=log.log
            )
            output_dict = upscaler.upscale_series(
                file_list,
                output_path,
                upscale_factor,
                output_basename,
                output_type="png"
            )

        # Convert Interpolated frames to video of original length
        convert_png_to_mp4(
            input_images_dir="temp_input_frames" if inplace else "temp_upscaled_frames", 
            output_path=input_video if inplace else "upscaled.mp4"
        )

        get_video_info(input_video=input_video if inplace else "output.mp4")

        return {
            input_video: gr.update(value=input_video),
            upscaled_video: gr.update(value="upscaled.mp4", visible=True)
        } if inplace else {
            input_video: gr.update(value=input_video),
            upscaled_video: gr.update(value=None, visible=False)
        } 

    app_header.render()

    gr.HTML(SimpleIcons.INCREASING + "Increase the number of video frames to any depth and upscale them to any size", elem_id="tabheading")
    
    with gr.Column() as video_panel:
        with gr.Row():
            with gr.Column(scale=3):
                input_video = gr.PlayableVideo(label="Low FPS Input video", format="mp4")
                interpolated_video = gr.PlayableVideo(label="Video with increased FPS", visible=False, interactive=False)
                interpolation_presentation_video = gr.PlayableVideo(label="Original vs Interpolated Video", visible=False, interactive=False)
                upscaled_video = gr.PlayableVideo(label="Upscaled Video", visible=False, interactive=False)
            with gr.Column(scale=2, visible=False) as controle_panel:
                video_info = gr.HighlightedText(value=[(k, v) for k, v in input_video_info.items()], label="Input video Info")
                with gr.Tab(label="Interpolate"):
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            splits_input_slider = gr.Slider(value=1, minimum=1, maximum=10, step=1, label="Split Count", info="Number of splits b/w two frames")
                            info_output_vi = gr.Textbox(value="1", label="Interpolations", max_lines=1, interactive=False, info="Interpolations per frame")
                        with gr.Row():
                            info_original_fps = gr.Textbox(value="0", label="Original FPS", max_lines=1, interactive=False, info="Original FPS of the input video")
                            info_new_fps = gr.Textbox(value="0", label="Updated FPS", max_lines=1, interactive=False, info="FPS after interpolations")
                    output_mode = gr.Radio(choices=["Inplace", "Separate", "Presentation"], value="Inplace", label="Output mode", info="How the result video should appear?", interactive=True)
                    interpolate_button = gr.Button("Interpolate Video " + SimpleIcons.ROBOT, variant="primary",)
                
                with gr.Tab(label="Upscale"):
                    with gr.Column(variant="panel"):
                        scale_input = gr.Slider(value=4.0, minimum=1.0, maximum=8.0, step=0.05, label="Frame Upscale Factor", interactive=True)
                        upscale_inplace = gr.Checkbox(value=False, label="Inplace", interactive=True, info="Replace the original video with Upscaled video")
                    upscale_button = gr.Button("Upscale Video " + SimpleIcons.ROCKET, variant="primary")

    """Event Handelers"""
    # For any update in input video component
    input_video.change(
        reset_ui, 
        inputs=[input_video, splits_input_slider], 
        outputs=[controle_panel, info_original_fps, info_new_fps, interpolated_video, interpolate_button, video_info]
    )

    # If Interpolate button is clicked
    interpolate_button.click(
        video_inflation, 
        inputs=[input_video, splits_input_slider, output_mode], 
        outputs=[input_video, interpolated_video, interpolation_presentation_video, video_info]
    )

    # If Upscale Button is clicked
    upscale_button.click(
        video_upscale,
        inputs=[input_video, scale_input, upscale_inplace],
        outputs=[input_video, upscaled_video]
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
    debug=True,
    enable_queue=True
)
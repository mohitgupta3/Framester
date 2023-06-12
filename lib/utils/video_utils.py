"""Functions for dealing with video using FFmpeg"""
import cv2
import os
import glob
from ffmpy import FFmpeg, FFprobe
from .image_utils import gif_frame_count
from .file_utils import split_filepath

QUALITY_NEAR_LOSSLESS = 17
QUALITY_SMALLER_SIZE = 28
QUALITY_DEFAULT = 23


def determine_pattern(input_path: str):
    """Determine the FFmpeg wildcard pattern needed for a set of files"""
    files = sorted(glob.glob(os.path.join(input_path, "*.png")))
    first_file = files[0]
    file_count = len(files)
    num_width = len(str(file_count))
    _, name_part, ext_part = split_filepath(first_file)
    return f"{name_part[:-num_width]}%0{num_width}d{ext_part}"


def PNGtoMP4(
    input_path: str,  # pylint: disable=invalid-name
    filename_pattern: str,
    frame_rate: int,
    output_filepath: str,
    crf: int = QUALITY_DEFAULT,
):
    """Encapsulate logic for the PNG Sequence to MP4 feature"""
    # if filename_pattern is "auto" it uses the filename of the first found file
    # and the count of file to determine the pattern, .png as the file type
    # ffmpeg -framerate 60 -i .\upscaled_frames%05d.png -c:v libx264 -r 60  -pix_fmt yuv420p
    #   -crf 28 test.mp4    if filename_pattern == "auto":
    filename_pattern = "if\_\[\%02d\]\%d.png" #determine_pattern(input_path)
    
    ffcmd = FFmpeg(
        inputs={os.path.join(input_path, filename_pattern): f"-framerate {frame_rate}"},
        outputs={
            output_filepath: f"-c:v libx264 -r {frame_rate} -pix_fmt yuv420p -crf {crf}"
        },
        global_options="-y",
    )
    cmd = ffcmd.cmd
    ffcmd.run()
    return cmd

def MP4toPNG(
    input_path: str,  # pylint: disable=invalid-name
    filename_pattern: str,
    frame_rate: int,
    output_path: str,
    start_number: int = 0,
):
    """Encapsulate logic for the MP4 to PNG Sequence feature"""
    cap = cv2.VideoCapture(input_path)
    success, frame = cap.read()
    count = start_number
    print(success)
    while success:
        output_filename = os.path.join(output_path, filename_pattern.format(count))
        output_filename = os.path.splitext(output_filename)[0] + '.png'  # Update file extension

        cv2.imwrite(output_filename, frame)

        count += 1
        for _ in range(frame_rate - 1):
            # Skip frames based on the desired frame rate
            cap.read()

        success, frame = cap.read()

    cap.release()

def mp4_frame_count(input_path: str) -> int:
    """Using FFprobe to determine MP4 frame count"""
    # ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -print_format default=nokey=1:noprint_wrappers=1 Big_Buck_Bunny_1080_10s_20MB.mp4
    ff = FFprobe(
        inputs={
            input_path: "-count_frames -show_entries stream=nb_read_frames -print_format default=nokey=1:noprint_wrappers=1"
        }
    )
    return ff.run()

import cv2
import glob
import os
import torch
import numpy as np
from os import path as osp
from torch.utils.data import DataLoader, Dataset

from models import RVRT as net

class utils_video:
    ...

class VideoRecurrentDataset(Dataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames.
    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoRecurrentDataset, self).__init__()
        self.opt = opt
        self.lq_root = opt["dataroot_lq"]
        self.data_info = {"lq_path": [], "folder": [], "idx": [], "border": []}

        self.imgs_lq = {}
        if "meta_info_file" in opt:
            with open(opt["meta_info_file"], "r") as fin:
                subfolders = [line.split(" ")[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, "*")))

        for subfolder_lq in subfolders_lq:
            # get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_lq)
            img_paths_lq = sorted(
                list(utils_video.scandir(subfolder_lq, full_path=True))
            )

            max_idx = len(img_paths_lq)

            self.data_info["lq_path"].extend(img_paths_lq)
            self.data_info["folder"].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info["idx"].append(f"{i}/{max_idx}")
            border_l = [0] * max_idx
            for i in range(self.opt["num_frame"] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info["border"].extend(border_l)

            self.imgs_lq[subfolder_name] = img_paths_lq

        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info["folder"])))

    def __getitem__(self, index):
        folder = self.folders[index]

        imgs_lq = utils_video.read_img_seq(self.imgs_lq[folder])

        return {
            "L": imgs_lq,
            "folder": folder,
            "lq_path": self.imgs_lq[folder],
        }

    def __len__(self):
        return len(self.folders)


def prepare_model(task, folder_lq):
    """prepare model according to task."""

    # define model
    if task == "upscale":
        model = net(
            upscale=4,
            clip_size=2,
            img_size=[2, 64, 64],
            window_size=[2, 8, 8],
            num_blocks=[1, 2, 1],
            depths=[2, 2, 2],
            embed_dims=[144, 144, 144],
            num_heads=[6, 6, 6],
            inputconv_groups=[1, 1, 1, 1, 1, 1],
            deformable_groups=12,
            attention_heads=12,
            attention_window=[3, 3],
            cpu_cache_length=100,
        )
        scale = 4
        window_size = [2, 8, 8]
        nonblind_denoising = False

    elif task in ["deblur"]:
        model = net(
            upscale=1,
            clip_size=2,
            img_size=[2, 64, 64],
            window_size=[2, 8, 8],
            num_blocks=[1, 2, 1],
            depths=[2, 2, 2],
            embed_dims=[192, 192, 192],
            num_heads=[6, 6, 6],
            inputconv_groups=[1, 3, 3, 3, 3, 3],
            deformable_groups=12,
            attention_heads=12,
            attention_window=[3, 3],
            cpu_cache_length=100,
        )
        scale = 1
        window_size = [2, 8, 8]
        nonblind_denoising = False

    elif task == "denoise":
        model = net(
            upscale=1,
            clip_size=2,
            img_size=[2, 64, 64],
            window_size=[2, 8, 8],
            num_blocks=[1, 2, 1],
            depths=[2, 2, 2],
            embed_dims=[192, 192, 192],
            num_heads=[6, 6, 6],
            inputconv_groups=[1, 3, 4, 6, 8, 4],
            deformable_groups=12,
            attention_heads=12,
            attention_window=[3, 3],
            nonblind_denoising=True,
            cpu_cache_length=100,
        )
        scale = 1
        window_size = [2, 8, 8]
        nonblind_denoising = True

    model_path = f"ckpt/{task}.pth"

    pretrained_model = torch.load(model_path)
    model.load_state_dict(
        pretrained_model["params"]
        if "params" in pretrained_model.keys()
        else pretrained_model,
        strict=True,
    )

    return model, scale, window_size, nonblind_denoising


def test_video(lq, model, scale, window_size, tile, tile_overlap, nonblind_denoising):
    """test the video as a whole or as clips (divided temporally)."""

    def test_clip(
        lq, model, scale, window_size, tile, tile_overlap, nonblind_denoising
    ):
        """test the clip as a whole or as patches."""
        size_patch_testing = tile[1]
        assert (
            size_patch_testing % window_size[-1] == 0
        ), "testing patch size should be a multiple of window_size."

        if size_patch_testing:
            # divide the clip to patches (spatially only, tested patch by patch)
            overlap_size = tile_overlap[1]
            not_overlap_border = True

            # test patch by patch
            b, d, c, h, w = lq.size()
            c = c - 1 if nonblind_denoising else c
            stride = size_patch_testing - overlap_size
            h_idx_list = list(range(0, h - size_patch_testing, stride)) + [
                max(0, h - size_patch_testing)
            ]
            w_idx_list = list(range(0, w - size_patch_testing, stride)) + [
                max(0, w - size_patch_testing)
            ]
            E = torch.zeros(b, d, c, h * scale, w * scale)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = lq[
                        ...,
                        h_idx : h_idx + size_patch_testing,
                        w_idx : w_idx + size_patch_testing,
                    ]
                    out_patch = model(in_patch).detach().cpu()

                    out_patch_mask = torch.ones_like(out_patch)

                    if not_overlap_border:
                        if h_idx < h_idx_list[-1]:
                            out_patch[..., -overlap_size // 2 :, :] *= 0
                            out_patch_mask[..., -overlap_size // 2 :, :] *= 0
                        if w_idx < w_idx_list[-1]:
                            out_patch[..., :, -overlap_size // 2 :] *= 0
                            out_patch_mask[..., :, -overlap_size // 2 :] *= 0
                        if h_idx > h_idx_list[0]:
                            out_patch[..., : overlap_size // 2, :] *= 0
                            out_patch_mask[..., : overlap_size // 2, :] *= 0
                        if w_idx > w_idx_list[0]:
                            out_patch[..., :, : overlap_size // 2] *= 0
                            out_patch_mask[..., :, : overlap_size // 2] *= 0

                    E[
                        ...,
                        h_idx * scale : (h_idx + size_patch_testing) * scale,
                        w_idx * scale : (w_idx + size_patch_testing) * scale,
                    ].add_(out_patch)
                    W[
                        ...,
                        h_idx * scale : (h_idx + size_patch_testing) * scale,
                        w_idx * scale : (w_idx + size_patch_testing) * scale,
                    ].add_(out_patch_mask)
            output = E.div_(W)

        else:
            _, _, _, h_old, w_old = lq.size()
            h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
            w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

            lq = (
                torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3)
                if h_pad
                else lq
            )
            lq = (
                torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4)
                if w_pad
                else lq
            )

            output = model(lq).detach().cpu()

            output = output[:, :, :, : h_old * scale, : w_old * scale]

        return output

    num_frame_testing = tile[0]
    if num_frame_testing:
        # test as multiple clips if out-of-memory
        num_frame_overlapping = tile_overlap[0]
        not_overlap_border = False
        b, d, c, h, w = lq.size()
        c = c - 1 if nonblind_denoising else c
        stride = num_frame_testing - num_frame_overlapping
        d_idx_list = list(range(0, d - num_frame_testing, stride)) + [
            max(0, d - num_frame_testing)
        ]
        E = torch.zeros(b, d, c, h * scale, w * scale)
        W = torch.zeros(b, d, 1, 1, 1)

        for d_idx in d_idx_list:
            lq_clip = lq[:, d_idx : d_idx + num_frame_testing, ...]
            out_clip = test_clip(
                lq_clip,
                model,
                scale,
                window_size,
                tile,
                tile_overlap,
                nonblind_denoising,
            )
            out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

            if not_overlap_border:
                if d_idx < d_idx_list[-1]:
                    out_clip[:, -num_frame_overlapping // 2 :, ...] *= 0
                    out_clip_mask[:, -num_frame_overlapping // 2 :, ...] *= 0
                if d_idx > d_idx_list[0]:
                    out_clip[:, : num_frame_overlapping // 2, ...] *= 0
                    out_clip_mask[:, : num_frame_overlapping // 2, ...] *= 0

            E[:, d_idx : d_idx + num_frame_testing, ...].add_(out_clip)
            W[:, d_idx : d_idx + num_frame_testing, ...].add_(out_clip_mask)
        output = E.div_(W)
    else:
        # test as one clip (the whole video) if you have enough memory
        window_size = window_size
        d_old = lq.size(1)
        d_pad = (window_size[0] - d_old % window_size[0]) % window_size[0]
        lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1) if d_pad else lq
        output = test_clip(
            lq, model, scale, window_size, tile, tile_overlap, nonblind_denoising
        )
        output = output[:, :d_old, :, :, :]

    return output


def main(
    task,
    folder_lq,
    tile: list,
    tile_overlap: list,
    sigma: int = 50,
    num_workers: int = 2,
    save_result: bool = True,
):
    # define model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scale, window_size, nonblind_denoising = prepare_model(
        task, folder_lq
    )
    model.eval()
    model = model.to(device)

    test_set = VideoRecurrentDataset(
        {
            "dataroot_lq": folder_lq,
            "sigma": sigma,
            "num_frame": -1,
            "cache_data": False,
        }
    )

    test_loader = DataLoader(
        dataset=test_set, num_workers=num_workers, batch_size=1, shuffle=False
    )

    save_dir = f"results/{task}"
    if save_result:
        os.makedirs(save_dir, exist_ok=True)

    for idx, batch in enumerate(test_loader):
        folder = batch["folder"]

        # inference
        with torch.no_grad():
            output = test_video(
                batch["L"].to(device),
                model,
                scale,
                window_size,
                tile,
                tile_overlap,
                nonblind_denoising,
            )

        for i in range(output.shape[1]):
            # save image
            img = output[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if img.ndim == 3:
                img = np.transpose(
                    img[[2, 1, 0], :, :], (1, 2, 0)
                )  # CHW-RGB to HCW-BGR
            img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
            if save_result:
                seq_ = osp.basename(batch["lq_path"][i][0]).split(".")[0]
                os.makedirs(f"{save_dir}/{folder[0]}", exist_ok=True)
                cv2.imwrite(f"{save_dir}/{folder[0]}/{seq_}.png", img)

        print("Testing {:20s}  ({:2d}/{})".format(folder[0], idx, len(test_loader)))

# Params for upscaling operation
task         = "upscale"
folder_lq    = "temp_input_frames"
tile         = [30, 64, 64]
tile_overlap = [2, 20, 20]
num_workers  = 2
save_result  = True
sigma        = 50

main(task, folder_lq, tile, tile_overlap, sigma, num_workers, save_result)


import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2
import imageio
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from rembg import remove
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor


def sample(
    input_path: str = "assets/test_image.png",
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,
    device: str = "cuda",
    output_folder: Optional[str] = None,
    elevations_deg: Optional[float | List[float]] = 10.0,
    azimuths_deg: Optional[List[float]] = None,
    image_frame_ratio: Optional[float] = None,
    verbose: Optional[bool] = False,
):
    if version == "svd":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd/")
        model_config = "scripts/sampling/configs/svd.yaml"

    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt/")
        model_config = "scripts/sampling/configs/svd_xt.yaml"

    elif version == "sv3d_u":
        num_frames = 21
        num_steps = default(num_steps, 50)
        output_folder = default(output_folder, "outputs/simple_video_sample/sv3d_u/")
        model_config = "scripts/sampling/configs/sv3d_u.yaml"
        cond_aug = 1e-5

    elif version == "sv3d_p":
        num_frames = 21
        num_steps = default(num_steps, 50)
        output_folder = default(output_folder, "outputs/simple_video_sample/sv3d_p/")
        model_config = "scripts/sampling/configs/sv3d_p.yaml"
        cond_aug = 1e-5

        if isinstance(elevations_deg, (int, float)):
            elevations_deg = [elevations_deg] * num_frames

        polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]

        if azimuths_deg is None:
            azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360

        azimuths_rad = [np.deg2rad(a % 360) for a in azimuths_deg]
        azimuths_rad[:-1].sort()

    else:
        raise ValueError(f"Unknown version: {version}")

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )

    torch.manual_seed(seed)

    path = Path(input_path)
    if path.is_file():
        all_img_paths = [path]
    elif path.is_dir():
        all_img_paths = sorted(
            p for p in path.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        )
    else:
        raise ValueError("Invalid input path")

    for input_img_path in all_img_paths:
        # ================= IMAGE LOADING =================

        if "sv3d" in version:
            image = Image.open(input_img_path)

            if image.mode != "RGBA":
                image.thumbnail([768, 768], Image.Resampling.LANCZOS)
                image = remove(image.convert("RGBA"), alpha_matting=True)

            image_arr = np.array(image)
            alpha = image_arr[..., -1]

            _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
            x, y, w, h = cv2.boundingRect(mask)

            side = max(w, h)
            padded = np.zeros((side, side, 4), dtype=np.uint8)

            cx = side // 2
            padded[
                cx - h // 2 : cx - h // 2 + h,
                cx - w // 2 : cx - w // 2 + w,
            ] = image_arr[y : y + h, x : x + w]

            rgba = Image.fromarray(padded).resize((576, 576), Image.LANCZOS)
            rgba = np.array(rgba) / 255.0
            rgb = rgba[..., :3] * rgba[..., 3:4] + (1 - rgba[..., 3:4])
            input_image = (rgb * 255).astype(np.uint8)

        else:
            with Image.open(input_img_path) as img:
                if img.mode == "RGBA":
                    img = img.convert("RGB")

                w, h = img.size
                input_image = img

                if w % 64 != 0 or h % 64 != 0:
                    nw, nh = w - w % 64, h - h % 64
                    input_image = input_image.resize((nw, nh))
                    print(f"Resized from {w}x{h} → {nw}x{nh}")

                input_image = np.array(input_image)

        # ================= TENSOR =================

        image = ToTensor()(input_image)
        image = image * 2.0 - 1.0
        image = image.unsqueeze(0).to(device)

        # ================= CONDITIONING =================

        value_dict = {
            "cond_frames_without_noise": image,
            "cond_frames": image + cond_aug * torch.randn_like(image),
            "motion_bucket_id": torch.tensor([motion_bucket_id], device=device),
            "fps_id": torch.tensor([fps_id], device=device),
            "cond_aug": cond_aug,
        }

        if "sv3d_p" in version:
            value_dict["polars_rad"] = polars_rad
            value_dict["azimuths_rad"] = azimuths_rad

        with torch.no_grad(), torch.autocast(device):
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                device=device,
            )

            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )

            for k in ["crossattn", "concat"]:
                c[k] = rearrange(repeat(c[k], "b ... -> b t ...", t=num_frames), "b t ... -> (b t) ...")
                uc[k] = rearrange(repeat(uc[k], "b ... -> b t ...", t=num_frames), "b t ... -> (b t) ...")

            shape = (num_frames, 4, image.shape[2] // 8, image.shape[3] // 8)
            noise = torch.randn(shape, device=device)

            samples_z = model.sampler(
                lambda x, s, c: model.denoiser(model.model, x, s, c),
                noise,
                cond=c,
                uc=uc,
            )

            model.en_and_decode_n_samples_a_time = decoding_t
            samples = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples + 1) / 2, 0, 1)

            os.makedirs(output_folder, exist_ok=True)
            idx = len(glob(f"{output_folder}/*.mp4"))

            samples = embed_watermark(samples)
            samples = filter(samples)

            video = (rearrange(samples, "t c h w -> t h w c") * 255).byte().cpu().numpy()
            imageio.mimwrite(f"{output_folder}/{idx:06d}.mp4", video)


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list({e.input_key for e in conditioner.embedders})


def get_batch(keys, value_dict, N, T, device):
    batch, batch_uc = {}, {}

    for key in keys:
        if key in value_dict:
            val = value_dict[key]
            batch[key] = repeat(val, "1 ... -> b ...", b=N[0]) if isinstance(val, torch.Tensor) else val

    batch["num_video_frames"] = T

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_uc[k] = v.clone()

    return batch, batch_uc


def load_model(config, device, num_frames, num_steps, verbose):
    cfg = OmegaConf.load(config)
    cfg.model.params.sampler_config.params.num_steps = num_steps
    cfg.model.params.sampler_config.params.guider_config.params.num_frames = num_frames

    model = instantiate_from_config(cfg.model).to(device).eval()
    filter = DeepFloydDataFiltering(device=device)

    return model, filter


if __name__ == "__main__":
    Fire(sample)

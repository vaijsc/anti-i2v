import gc
import logging
import math
import sys
import os
import random
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict
import json
import open_clip
import yaml
import lpips
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import transformers
import diffusers
from glob import glob
import kornia
import numpy as np
import argparse

from einops import rearrange, repeat
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, T5EncoderModel, get_scheduler, SchedulerType

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)
# from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers

# from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
# from huggingface_hub import create_repo, upload_folder
from utils import (
    load_image_prompts,
    load_vbench_prompts,
    pil_to_pt,
    readFrames,
    compute_prompt_embeddings,
    get_optimizer,
    reset_memory,
    print_memory,
    augment_image,
    prepare_rotary_positional_embeddings,
    calculate_vae_loss,
    calculate_clip_loss_model,
    to_dct_freq,
    eot_transform,
    load_yaml,
    update_args_with_yaml,
    load_mask,
    get_gradient_norm,
    tensor_to_image,
    REPALoss
)
logger = get_logger(__name__)
# from attention_processor import AttentionStore, StateStore, CogVideoXAttnProcessorTrain2_0, EnhanceTemporalCogVideoXAttnProcessor2_0, CFIStore, num_frames_hook
# from dct_utils import dct_2d, idct_2d
# from color_space import rgb2lab, lab2rgb
# from face import FaceProcessing

# sys.path.append("arcface_torch")
# from backbones import get_model 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--annotation", type=str, default="data_cog.json", help="a txt/csv containing all info")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="Save adversarial examples",
    )
    parser.add_argument(
        "--seed", 
        default=None, 
        type=int, 
        help="seed for reproducibility"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vgg_checkpoint_path",
        type=str,
        default="backbone.pth",
        help="Path to pretrained vgg models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU. "
            "Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this "
            "argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform.",
    )

    ######## Training settings ########
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    ## TODO: Not support for now
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--ref_folder",
        default="",
        type=str,
        help="Reference folder",
    )
    parser.add_argument(
        "--use_cpu_offload_optimizer",
        action="store_true",
        help="Whether or not to use the CPUOffloadOptimizer from TorchAO to perform optimization step and maintain parameters on the CPU.",
    )
    parser.add_argument(
        "--offload_gradients",
        action="store_true",
        help="Whether or not to offload the gradients to CPU when using the CPUOffloadOptimizer from TorchAO.",
    )
    
    ######## Attack settings ########
    parser.add_argument(
        "--precompute_video_latents",
        action="store_true",
        help="precompute_video_latents",
    )
    parser.add_argument("--pgd_alpha", type=float, default=1.0, help="The step size for pgd.")
    parser.add_argument("--pgd_eps", type=float, default=16.0, help="The noise budget for pgd.")
    parser.add_argument("--num_frames_train", type=int, default=16)

    parser.add_argument("--caption", action="store_true", help="use video caption")
    parser.add_argument("--vbench", action="store_true")
    parser.add_argument("--celebv", action="store_true")
    parser.add_argument("--caption_path", type=str, default="", help="IDCT Augment")
    
    parser.add_argument("--use_ref", action="store_true", help="use reference video")
    parser.add_argument("--start_index", default=None, type=int, help="start index for sample name")
    parser.add_argument("--end_index", default=None, type=int, help="end index for sample name")
    parser.add_argument(
        "--sign", type=str, default="-",
        # help="Save adversarial examples",
    )
    
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )
    parser.add_argument(
        "--ignore_learned_positional_embeddings",
        action="store_true",
        default=False,
        help=(
            "Whether to ignore the learned positional embeddings when training CogVideoX Image-to-Video. This setting "
            "should be used when performing multi-resolution training, because CogVideoX-I2V does not support it "
            "otherwise. Please read the comments in https://github.com/a-r-r-o-w/cogvideox-factory/issues/26 to understand why."
        ),
    )
    parser.add_argument(
        "--nccl_timeout",
        type=int,
        default=600,
        help="Maximum timeout duration before which allgather, or related, operations fail in multi-GPU/multi-node training settings.",
    )

    return parser.parse_args()


def velocity_to_eps(model_output, alphas_cumprod, noisy_video_latents, timesteps):
    alpha_prod_t = alphas_cumprod[timesteps]
    beta_prod_t = 1 - alpha_prod_t
    pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * noisy_video_latents
    return pred_epsilon


def transformer_training(
    args,
    images,
    vae,
    transformer,
    scheduler,
    videos=None,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    model_config=None,
    device=None,
    dtype=None
):
    alphas_cumprod = scheduler.alphas_cumprod.to(device, dtype=dtype)
    bsz, c, h, w = images.shape
    
    ################################ Both ################################
    if args.precompute_video_latents:
        video_latents = videos.permute(0, 2, 1, 3, 4)
        video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=dtype)
    else:
        if videos is None:
            videos_data = images.unsqueeze(2).repeat(1, 1, args.num_frames_train, 1, 1)  # Init videos_data as stack of images
        else:
            videos_reshaped = rearrange(
                videos,
                '(b f) c h w -> b c f h w',
                f=args.num_frames_train, b=bsz, c=c, h=h, w=w
            ).to(device, dtype=dtype)

            if args.replace_first:
                videos_data = torch.cat(
                    [images.unsqueeze(2), videos_reshaped[:, :, 1:, :, :]], dim=2
                ).to(device, dtype=dtype)
            else:
                videos_data = videos_reshaped

        video_latents = vae.encode(videos_data).latent_dist.sample() * vae.config.scaling_factor
        # video_latents = model.encode(videos_data)
        video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=dtype)

    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    # Sample noise that will be added to the latents
    noise = torch.randn_like(video_latents)
    batch_size, num_frames, _, height, width = video_latents.shape

    # Sample a random timestep for each image
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (batch_size,),
        dtype=torch.int64,
        device=device
    )

    # VAE_SCALING_FACTOR = model.vae.config.scaling_factor
    VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae.config.block_out_channels) - 1)
    image_rotary_emb = tuple(
        prepare_rotary_positional_embeddings(
            height=height * VAE_SCALE_FACTOR_SPATIAL,
            width=width * VAE_SCALE_FACTOR_SPATIAL,
            num_frames=num_frames,
            vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
            patch_size=model_config.patch_size,
            patch_size_t=model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None,
            attention_head_dim=model_config.attention_head_dim,
            device=device,
        ) if model_config.use_rotary_positional_embeddings else None
    )
    freqs_cos, freqs_sin = image_rotary_emb
    freqs_cos, freqs_sin = freqs_cos.to(dtype=dtype), freqs_sin.to(dtype=dtype)
    image_rotary_emb = (freqs_cos, freqs_sin)

    noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)

    ################################ Image to add noise ################################
    images_in = images.unsqueeze(2) if images.dim() == 4 else images
    image_latents = vae.encode(images_in).latent_dist
    image_latents = image_latents.sample() * vae.config.scaling_factor

    image_latents = image_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
    image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=dtype)

    padding_shape = (video_latents.shape[0], video_latents.shape[1] - 1, *video_latents.shape[2:])
    latent_padding = image_latents.new_zeros(padding_shape)
    image_latents_padded = torch.cat([image_latents, latent_padding], dim=1)

    # noisy_model_input = torch.cat([noisy_video_latents, image_latents_padded], dim=2)
    # ofs_emb = None if model_config.ofs_embed_dim is None else latents.new_full((batch_size,), fill_value=2.0)

    # Predict the noise residual
    with torch.no_grad():
        noisy_model_input = torch.cat([noisy_video_latents, image_latents_padded], dim=2)
        model_output = transformer(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
        )
        
        # update z
        sds_noise_pred = velocity_to_eps(model_output, noisy_video_latents, timesteps)
    
    score = (sds_noise_pred - noise)
    target = (video_latents - score).detach()
    loss = 0.5 * F.mse_loss(video_latents.float(), target.float(), reduction="mean")
                        
    return loss


def main(args):
    if args.pgd_alpha >= 0.9:
        args.pgd_alpha = args.pgd_alpha / 255.0
    if args.pgd_eps >= 5:
        args.pgd_eps = args.pgd_eps / 255.0
        
    if not (args.num_frames_train % 4 == 0 or args.num_frames_train % 4 == 1):
        raise ValueError(
            "Num Frames should be 4k or 4k+1"
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    from datetime import datetime
    current_time = datetime.now()
    formatted_time = current_time.strftime("%m/%d/%Y_%H:%M")
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir, formatted_time)
    os.makedirs(logging_dir, exist_ok=True)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )
    
     # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        args_dict = vars(args)
        with open(f"{args.output_dir}/config.yaml", 'w') as f:
            yaml.dump(args_dict, f)
            f.close()
    
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    
    accelerator.wait_for_everyone()
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    if args.vbench or args.celebv:
        prompts, _, input_img_list, name_list = load_vbench_prompts(args.annotation)
    else:
        prompts, input_img_list, _ = load_image_prompts(args.annotation, args.caption)
        name_list = ["null" for i in range(len(prompts))]
    
    if args.celebv:
        assert args.caption_path != ""
        with open(args.caption_path, "r") as f:
            celeb_caption = f.readlines()
            f.close()
        celeb_caption = [i.strip() for i in celeb_caption]
        
    prompts = prompts[args.start_index:args.end_index]
    input_img_list = input_img_list[args.start_index:args.end_index]
    name_list = name_list[args.start_index:args.end_index]

    # Load models
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    ).to(accelerator.device, dtype=weight_dtype)
    
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    ).to(accelerator.device, dtype=weight_dtype)

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    null_embedding = compute_prompt_embeddings(
        tokenizer,
        text_encoder,
        "",
        model_config.max_text_seq_length,
        text_encoder.device,
        weight_dtype,
        requires_grad=False
    ).detach().cpu()

    prompt = "a video"
    prompt_embedding = compute_prompt_embeddings(
        tokenizer,
        text_encoder,
        prompt if isinstance(prompt, str) else prompt[0],
        model_config.max_text_seq_length,
        text_encoder.device,
        weight_dtype,
        requires_grad=False
    ).detach().cpu()

    caption_embedding = None
    caption = "a video"
    if caption is not None:
        caption_embedding = compute_prompt_embeddings(
            tokenizer,
            text_encoder,
            caption if isinstance(caption, str) else caption[0],
            model_config.max_text_seq_length,
            text_encoder.device,
            weight_dtype,
            requires_grad=False
        ).detach().cpu()

    if args.use_ref:
        print("Using image BLIP caption: ", caption)
        input_embedding = caption_embedding
    else:
        print("Using video captionn: ", prompt)
        input_embedding = prompt_embedding
    
    # if args.target_prompt != "":
    #     print("Using target caption: ", args.target_prompt)

    del text_encoder

    for prompt, input_img, n_id in zip(prompts, input_img_list, name_list):
        if args.celebv:
            name = n_id + ".png"
        else:
            name = input_img
        
        save_path_final = os.path.join(f"{args.output_dir}/noise-ckpt/{args.max_train_steps}", f"{os.path.basename(name)}")

        if os.path.exists(save_path_final):
            print("Skipping: ", save_path_final)
            continue
        else:
            all_save_files = glob(os.path.join(f"{args.output_dir}/noise-ckpt/*", f"{os.path.basename(input_img)}"))
            if len(all_save_files) != 0: 
                resume_step = max([int(i.split("/noise-ckpt/")[-1].split("/")[0]) for i in all_save_files])
                
                original_input_img = input_img
                input_img = os.path.join(f"{args.output_dir}/noise-ckpt/{str(resume_step)}", f"{os.path.basename(input_img)}")
                assert os.path.exists(input_img)
            else:
                resume_step = 0
                original_input_img = input_img
        print(prompt, input_img, original_input_img)

        i_list = [input_img]
        images = [Image.open(input_img).resize((args.width, args.height))]
        original_images = [Image.open(original_input_img).resize((args.width, args.height))]
        caption = None

        if args.use_ref:
            video_name = os.path.basename(input_img).split(".")[0]
            if args.celebv:
                assert video_name == n_id, (video_name, n_id)
            all_ref_videos = glob(os.path.join(args.ref_folder, video_name, "*"))
            assert len(all_ref_videos) > 0
            
            # Image caption
            if args.celebv:
                info = [i for i in celeb_caption if n_id in i]
                assert len(info) == 1
                info = info[0]
                
                caption = " ".join(info.split()[1:])                       
            else:
                caption = os.path.basename(all_ref_videos[0]).split(".")[0][:-2]  # Hard-coded format

            caption = "a video"

            video_pil = [readFrames(i, (args.width, args.height)) for i in all_ref_videos]
            video_pil = [pil_to_pt(i) for i in video_pil]
            video_pil = [((i * 2.0) - 1.0) for i in video_pil]  # [-1, 1]
            video_pil = [i[:args.num_frames_train] for i in video_pil]

            if args.precompute_video_latents:
                video_data = []
                vae.to(accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    for vid in video_pil:
                        out = vid.permute(1, 0, 2, 3).unsqueeze(0).to(accelerator.device, weight_dtype)
                        out = vae.encode(out).latent_dist.sample() * vae.config.scaling_factor
                        video_data.append(out.detach().cpu())
                # vae.to("cpu")
            else:
                video_data = video_pil
        else:
            video_data = None
        
        perturbed_data = pil_to_pt(images).to(accelerator.device, dtype=weight_dtype)
        perturbed_data = (perturbed_data * 2.0) - 1.0  # [-1, 1]
        perturbed_data.requires_grad_(True)

        original_data = pil_to_pt(original_images).to(accelerator.device, dtype=weight_dtype)
        original_data = (original_data * 2.0) - 1.0  # [-1, 1]
        original_data.requires_grad_(False)
        
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = 1
        # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        
        torch.cuda.empty_cache()
        
        if accelerator.is_main_process:
            # tracker_name = args.tracker_name or "cogvideox-lora"
            accelerator.init_trackers(prompt)
            accelerator.print("===== Memory before training =====")
            reset_memory(accelerator.device)
            print_memory(accelerator.device)

        logger.info("***** Running training *****")
        logger.info(f"  Image = {input_img}")
        logger.info(f"  Image Size = {(args.height, args.width)}")
        logger.info(f"  Budget = {args.pgd_eps} | {args.pgd_alpha}")
        logger.info(f"  Num Steps = {args.max_train_steps}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

        progress_bar = tqdm(
            range(int(args.max_train_steps / accelerator.num_processes)), disable=not accelerator.is_local_main_process
        )

        for epoch in range(args.num_train_epochs):
            loss_sds = 0
            perturbed_data.requires_grad = True
            transformer.eval()
            vae.eval()
            
            videos = None
            if video_data is not None:
                videos = random.choice(video_data)
                videos.requires_grad = False
                
            loss = transformer_training(
                args,
                perturbed_data,
                vae,
                transformer,
                scheduler,
                videos=videos,
                prompt_embeds=input_embedding,
                negative_prompt_embeds=null_embedding,
                model_config=model_config,
                device=accelerator.device,
                dtype=weight_dtype
            )
            
            transformer.zero_grad()
            vae.zero_grad()
            
            loss.backward()
            
            sign = 1 if args.sign == "+" else -1
            adv_images = perturbed_data + (sign) * args.pgd_alpha * perturbed_data.grad.sign()
            
            eta = torch.clamp(adv_images - original_data, min=-args.pgd_eps, max=+args.pgd_eps)
            perturbed_data = torch.clamp(original_data + eta, min=-1, max=+1).detach_()
            
            if accelerator.sync_gradients:
                if (epoch + 1) % args.checkpointing_steps == 0 and (accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED):
                    save_folder = f"{args.output_dir}/noise-ckpt/{epoch+1}"
                    os.makedirs(save_folder, exist_ok=True)
                
                    noised_imgs = perturbed_data.detach()
                    for img_pixel, img_name in zip(noised_imgs, i_list):
                        save_path = os.path.join(save_folder, f"{os.path.basename(img_name)}")
                        Image.fromarray(
                            (((img_pixel + 1)/2)*255.0).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                        ).save(save_path)
                    print(f"Saved noise at step {epoch+1} to {save_folder}")
    
    accelerator.end_training()
    

if __name__ == "__main__":
    args = parse_args()
    yaml_config = load_yaml(args.config)
    args = update_args_with_yaml(args, yaml_config)

    main(args)
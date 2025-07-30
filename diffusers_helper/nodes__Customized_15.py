import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' 

import torch
import torch.nn.functional as F
import gc
import numpy as np
import math
from tqdm import tqdm

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
import comfy.model_base
import comfy.latent_formats
from comfy.cli_args import args, LatentPreviewMethod

# Added for VAE loading and core ComfyUI VAE model path
import comfy.sd
import comfy
script_directory = os.path.dirname(os.path.abspath(__file__))
vae_scaling_factor = 0.476986

from .diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
# from .diffusers_helper.memory import DynamicSwapInstaller, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation
from .diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from .diffusers_helper.utils import crop_or_pad_yield_mask
from .diffusers_helper.bucket_tools import find_nearest_bucket

# New class for assiging the clip model to specific GPU
class CustomCLIPVisionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("clip_vision"),),
                "gpu_id": ("INT", {"default": 0, "min": 0, "max": torch.cuda.device_count() - 1, "tooltip": "GPU device ID to load CLIP Vision onto"}),
            }
        }
    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "load_clip_vision"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Loads a CLIP Vision model onto a specified GPU."

    def load_clip_vision(self, clip_name, gpu_id):
        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_name)
        target_device = torch.device(f"cuda:{gpu_id}")
        
        # Load the original CLIP Vision model and move it to the target device
        clip_vision = comfy.clip_vision.load_clipvision(clip_path) 
        clip_vision.to(target_device) # Explicitly move to assigned GPU
        
        return (clip_vision,)

class CustomDualCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        clip_names = folder_paths.get_filename_list("clip")
        return {
            "required": {
                "clip_name": (clip_names,),
                "text_model_name": (clip_names,), 
                "clip_type": (["sdxl", "sd3", "flux", "hunyuan_video"], {"default": "hunyuan_video"}), 
                "gpu_id": ("INT", {"default": 0, "min": 0, "max": torch.cuda.device_count() - 1, "tooltip": "GPU device ID to load CLIP models onto"}),
            }
        }
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_dual_clip"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Loads Dual CLIP models onto a specified GPU."

    def load_dual_clip(self, clip_name, text_model_name, clip_type, gpu_id):
        clip_paths = [folder_paths.get_full_path_or_raise("clip", clip_name)]
        if text_model_name: 
            # For dual CLIPs, often a second path is implied
            clip_paths.append(folder_paths.get_full_path_or_raise("clip", text_model_name))

        target_device = torch.device(f"cuda:{gpu_id}")

        model_options = {
            "load_device": target_device,
            "offload_device": mm.text_encoder_offload_device(), 
            "dtype": mm.text_encoder_dtype(target_device),
            "initial_device": target_device, 
        }

        # Use comfy.sd.load_clip for the actual loading and let it manage device based on model_options
        loaded_clip = comfy.sd.load_clip(clip_paths, clip_type=getattr(comfy.sd.CLIPType, clip_type.upper()), model_options=model_options)
        
        # Ensure the clip model is explicitly on the target_device after loading
        loaded_clip.cond_stage_model.to(target_device) 
        
        return (loaded_clip,)

class HyVideoModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}
        self.load_device = kwargs.get("device", mm.get_torch_device())

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v

class HyVideoModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = comfy.latent_formats.HunyuanVideo
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        self.memory_usage_factor = 2.0
        self.unet_config["disable_unet_model_creation"] = True

class FramePackTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_single_blocks": ("BOOLEAN", {"default": True, "tooltip": "Enable single block compilation"}),
                "compile_double_blocks": ("BOOLEAN", {"default": True, "tooltip": "Enable double block compilation"}),
            },
        }
    RETURN_TYPES = ("FRAMEPACKCOMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch 2.5.0 is recommended"

    def loadmodel(self, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_single_blocks, compile_double_blocks):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "compile_single_blocks": compile_single_blocks,
            "compile_double_blocks": compile_double_blocks
        }

        return (compile_args, )

#Region Model loading
class DownloadAndLoadFramePackModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["lllyasviel/FramePackI2V_HY"],),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            # New for manual GPU Parallelization
            "gpu_id": ("INT", {"default": 0, "min": 0, "max": torch.cuda.device_count() - 1, "tooltip": "GPU device ID to load model onto"}), # min corrected to 0
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn",
                    "sageattn",
                    ], {"default": "sdpa"}),
                "compile_args": ("FRAMEPACKCOMPILEARGS", ),
            }
        }

    RETURN_TYPES = ("FramePackMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "FramePackWrapper"

    def loadmodel(self, model, base_precision, quantization, gpu_id, # gpu_id added here
                  compile_args=None, attention_mode="sdpa"):
        
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]

        target_device = torch.device(f"cuda:{gpu_id}") # Use torch.device directly
        
        model_path = os.path.join(folder_paths.models_dir, "diffusers", "lllyasviel", "FramePackI2V_HY")
        if not os.path.exists(model_path):
            print(f"Downloading clip model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )

        # Load model and move to target_device directly
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(model_path, torch_dtype=base_dtype, attention_mode=attention_mode).to(target_device)
        
        params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
        if quantization == 'fp8_e4m3fn' or quantization == 'fp8_e4m3fn_fast':
            transformer = transformer.to(torch.float8_e4m3fn)
            if quantization == "fp8_e4m3fn_fast":
                from .fp8_optimization import convert_fp8_linear
                convert_fp8_linear(transformer, base_dtype, params_to_keep=params_to_keep)
        elif quantization == 'fp8_e5m2':
            transformer = transformer.to(torch.float8_e5m2)
        else:
            transformer = transformer.to(base_dtype)

        # Use target_device for DynamicSwapInstaller
        DynamicSwapInstaller.install_model(transformer, device=target_device)

        if compile_args is not None:
            if compile_args["compile_single_blocks"]:
                for i, block in enumerate(transformer.single_transformer_blocks):
                    transformer.single_transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            if compile_args["compile_double_blocks"]:
                for i, block in enumerate(transformer.transformer_blocks):
                    transformer.transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                
        pipe = {
            "transformer": transformer.eval(),
            "dtype": base_dtype,
        }
        return (pipe, )
    
class LoadFramePackModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            # New to allow manual GPU Parallelization
            "gpu_id": ("INT", {"default": 2, "min": 2, "max": torch.cuda.device_count() - 1, "tooltip": "GPU device ID to load model onto"}), 
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn",
                    "sageattn",
                    ], {"default": "sdpa"}),
                "compile_args": ("FRAMEPACKCOMPILEARGS", ),
            }
        }

    RETURN_TYPES = ("FramePackMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "FramePackWrapper"

    def loadmodel(self, model, base_precision, quantization, gpu_id, 
                  compile_args=None, attention_mode="sdpa"):
        
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]

        target_device = torch.device(f"cuda:{gpu_id}") # Use torch.device directly
        offload_device = mm.unet_offload_device() # This is usually CPU when not specifically offloading to another GPU

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        model_config_path = os.path.join(script_directory, "transformer_config.json")
        import json
        with open(model_config_path, "r") as f:
            config = json.load(f)
        
        # Load state_dict to CPU first
        sd = load_torch_file(model_path, device=torch.device("cpu"), safe_load=True)
        
        # Instantiate model with empty weights
        with init_empty_weights():
            transformer = HunyuanVideoTransformer3DModelPacked(**config, attention_mode=attention_mode)
        
        # Load state dict and then move to target_device
        param_count = sum(1 for _ in transformer.named_parameters())
        for name, param in tqdm(transformer.named_parameters(), 
                desc=f"Loading transformer parameters to {offload_device}", 
                total=param_count,
                leave=True):
            dtype_to_use = base_dtype if any(keyword in name for keyword in ["norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"]) else base_dtype 
            set_module_tensor_to_device(transformer, name, device=torch.device("cpu"), dtype=dtype_to_use, value=sd[name]) # Load to CPU
        
        # After loading all params to CPU, move the entire transformer to the target GPU
        # transformer = transformer.to(target_device) 
        transformer = transformer.to(torch.device("cpu")) 
        
        params_to_keep_quant = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"} 
        if quantization == 'disabled':
            pass
        elif quantization == 'fp8_e4m3fn' or quantization == 'fp8_e4m3fn_fast':
            transformer = transformer.to(torch.float8_e4m3fn)
            if quantization == "fp8_e4m3fn_fast":
                from .fp8_optimization import convert_fp8_linear
                convert_fp8_linear(transformer, base_dtype, params_to_keep=params_to_keep_quant)
        elif quantization == 'fp8_e5m2':
            transformer = transformer.to(torch.float8_e5m2)
        
        # Use target_device for DynamicSwapInstaller
        # DynamicSwapInstaller.install_model(transformer, device=target_device)

        if compile_args is not None:
            if compile_args["compile_single_blocks"]:
                for i, block in enumerate(transformer.single_transformer_blocks):
                    transformer.single_transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            if compile_args["compile_double_blocks"]:
                for i, block in enumerate(transformer.transformer_blocks):
                    transformer.transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                
        pipe = {
            "transformer": transformer.eval(),
            "dtype": base_dtype,
        }
        return (pipe, )

class FramePackFindNearestBucket:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Image to resize"}),
            "base_resolution": ("INT", {"default": 640, "min": 64, "max": 2048, "step": 16, "tooltip": "Width of the image to encode"}),
            },
        }

    RETURN_TYPES = ("INT", "INT", )
    RETURN_NAMES = ("width","height",)
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Finds the closes resolution bucket as defined in the orignal code"

    def process(self, image, base_resolution):

        H, W = image.shape[1], image.shape[2]

        new_height, new_width = find_nearest_bucket(H, W, resolution=base_resolution)

        return (new_width, new_height, )

class FramePackSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image_embeds": ("CLIP_VISION_OUTPUT", ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "use_teacache": ("BOOLEAN", {"default": True, "tooltip": "Use teacache for faster sampling."}),
                "teacache_rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The threshold for the relative L1 loss."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 32.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "latent_window_size": ("INT", {"default": 9, "min": 1, "max": 33, "step": 1, "tooltip": "The size of the latent window to use for sampling."}),
                "total_second_length": ("FLOAT", {"default": 5, "min": 1, "max": 120, "step": 0.1, "tooltip": "The total length of the video in seconds."}),
                "gpu_memory_preservation": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 128.0, "step": 0.1, "tooltip": "The amount of GPU memory to preserve."}),
                "sampler": (["unipc_bh1", "unipc_bh2"],
                    {
                        "default": 'unipc_bh1'
                    }),
            },
            "optional": {
                "start_latent": ("LATENT", {"tooltip": "init Latents to use for image2video"} ),
                "end_latent": ("LATENT", {"tooltip": "end Latents to use for image2video"} ),
                "end_image_embeds": ("CLIP_VISION_OUTPUT", {"tooltip": "end Image's clip embeds"} ),
                "embed_interpolation": (["weighted_average", "linear"], {"default": 'linear', "tooltip": "Image embedding interpolation type. If linear, will smoothly interpolate with time, else it'll be weighted average with the specified weight."}),
                "start_embed_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Weighted average constant for image embed interpolation. If end image is not set, the embed's strength won't be affected"}),
                "initial_samples": ("LATENT", {"tooltip": "init Latents to use for video2video"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"

    def process(self, model, shift, positive, negative, latent_window_size, use_teacache, total_second_length, teacache_rel_l1_thresh, image_embeds, steps, cfg,
                guidance_scale, seed, sampler, gpu_memory_preservation, start_latent=None, end_latent=None, end_image_embeds=None, embed_interpolation="linear", start_embed_strength=1.0, initial_samples=None, denoise_strength=1.0, sampler_gpu_id=None):
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))
        print("total_latent_sections: ", total_latent_sections)

        transformer = model["transformer"]
        base_dtype = model["dtype"]

        # Determine the actual GPU device to use for sampling operations.
        if sampler_gpu_id is not None and torch.cuda.is_available() and sampler_gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{sampler_gpu_id}")
        elif torch.cuda.is_available():
            device = torch.device("cuda:0") # Fallback to cuda:0 if no specific ID given or invalid.
        else:
            device = torch.device("cpu") # Fallback to CPU if no CUDA is available at all.
            print("WARNING: No CUDA device detected for FramePackSampler. Falling back to CPU. Performance will be severely impacted.")

        offload_device = mm.unet_offload_device()

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
                
        if sampler_gpu_id is not None and torch.cuda.is_available() and sampler_gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{sampler_gpu_id}")
        elif torch.cuda.is_available():
            device = torch.device("cuda:0") 
            # Fallback to cuda:0 if no specific ID given or invalid.
        else:
            # Fallback to CPU if no CUDA is available at all.
            device = torch.device("cpu") 
            print("WARNING: No CUDA device detected for FramePackSampler. Falling back to CPU. Performance will be severely impacted.")

        if device.type == 'cuda': 
            transformer.to(device) 
            # move_model_to_device_with_memory_preservation(transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation)
            
        else:
            print("WARNING: Transformer model will remain on CPU for sampling.")

        # Ensure all input tensors are moved to the 'device'
        start_latent = start_latent["samples"] * vae_scaling_factor
        start_latent = start_latent.to(device) # Move to sampler's device
        if initial_samples is not None:
            initial_samples = initial_samples["samples"] * vae_scaling_factor
            initial_samples = initial_samples.to(device) # Move to sampler's device
        if end_latent is not None:
            end_latent = end_latent["samples"] * vae_scaling_factor
            end_latent = end_latent.to(device) # Move to sampler's device
        has_end_image = end_latent is not None
        print("start_latent", start_latent.shape)
        B, C, T, H, W = start_latent.shape

        start_image_encoder_last_hidden_state = image_embeds["last_hidden_state"].to(base_dtype).to(device)

        if has_end_image:
            assert end_image_embeds is not None
            end_image_encoder_last_hidden_state = end_image_embeds["last_hidden_state"].to(base_dtype).to(device)
        else:
            end_image_encoder_last_hidden_state = torch.zeros_like(start_image_encoder_last_hidden_state).to(device) 

        llama_vec = positive[0][0].to(base_dtype).to(device)
        clip_l_pooler = positive[0][1]["pooled_output"].to(base_dtype).to(device)

        if not math.isclose(cfg, 1.0):
            llama_vec_n = negative[0][0].to(base_dtype).to(device) 
            clip_l_pooler_n = negative[0][1]["pooled_output"].to(base_dtype).to(device)
        else:
            llama_vec_n = torch.zeros_like(llama_vec, device=device)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler, device=device)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
            
        # Sampling
        rnd = torch.Generator("cpu").manual_seed(seed)
        
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, H, W), dtype=torch.float32).to(device) # Initialize on device
        
        total_generated_latent_frames = 0

        latent_paddings_list = list(reversed(range(total_latent_sections)))
        latent_paddings = latent_paddings_list.copy()  # Create a copy for iteration

        comfy_model = HyVideoModel(
                HyVideoModelConfig(base_dtype),
                model_type=comfy.model_base.ModelType.FLOW,
                device=device
            )
        
        patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, torch.device("cpu"))
        from latent_preview import prepare_callback
        callback = prepare_callback(patcher, steps)

        # move_model_to_device_with_memory_preservation(transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation)

        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            latent_paddings_list = latent_paddings.copy()

        for i, latent_padding in enumerate(latent_paddings):
            print(f"latent_padding: {latent_padding}")
            is_last_section = latent_padding == 0
            is_first_section = latent_padding == latent_paddings[0]
            latent_padding_size = latent_padding * latent_window_size

            if embed_interpolation == "linear":
                if total_latent_sections <= 1:
                    frac = 1.0  
                else:
                    frac = 1 - i / (total_latent_sections - 1) 
            else:
                frac = start_embed_strength if has_end_image else 1.0

            image_encoder_last_hidden_state = start_image_encoder_last_hidden_state * frac + (1 - frac) * end_image_encoder_last_hidden_state

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0).to(device) # Move to device
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2).to(device) # Move to device

            # Use end image latent for the first section if provided
            if has_end_image and is_first_section:
                clean_latents_post = end_latent.to(history_latents)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2).to(device) # Move to device

            #vid2vid
            if initial_samples is not None:
                total_length = initial_samples.shape[2]
                max_padding = max(latent_paddings_list)
                
                if is_last_section:
                    start_idx = max(0, total_length - latent_window_size)
                else:
                    if max_padding > 0:  
                        progress = (max_padding - latent_padding) / max_padding
                        start_idx = int(progress * max(0, total_length - latent_window_size))
                    else:
                        start_idx = 0
                
                end_idx = min(start_idx + latent_window_size, total_length)
                print(f"start_idx: {start_idx}, end_idx: {end_idx}, total_length: {total_length}")
                input_init_latents = initial_samples[:, :, start_idx:end_idx, :, :].to(device) # Move to device
            
            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps, rel_l1_thresh=teacache_rel_l1_thresh)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=base_dtype, enabled=True):
                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler=sampler,
                    initial_latent=input_init_latents if initial_samples is not None else None,
                    strength=denoise_strength,
                    width = W * 8,
                    height = H * 8,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=guidance_scale,
                    guidance_rescale=0,
                    shift=shift if shift != 0 else None,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=device, # Use the actual device of the transformer
                    dtype=base_dtype,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if is_last_section:
                break

        mm.soft_empty_cache() # Still good to clear cache

        return {"samples": real_history_latents / vae_scaling_factor},

 # New custom VAELoader for explicit GPU assignment
class FramePackVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "gpu_id": ("INT", {"default": 2, "min": 2, "max": torch.cuda.device_count() - 1, "tooltip": "GPU device ID to load VAE onto"}),
            }
        }
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Loads a VAE onto a specified GPU."

    def load_vae(self, vae_name, gpu_id):
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        target_device = torch.device(f"cuda:{gpu_id}")
        
        # Load state_dict first (to CPU)
        vae_sd = comfy.utils.load_torch_file(vae_path, safe_load=True) 
        
        # Create an empty VAE instance
        # vae = comfy.vae.vae_model.VAE().eval() 
        vae = comfy.sd.VAE(sd=vae_sd, device=target_device)
        # .eval() 
        
        # Load the state_dict into the VAE instance
        # vae.load_state_dict(vae_sd)
        
        return (vae, )     

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadFramePackModel": DownloadAndLoadFramePackModel,
    "FramePackSampler": FramePackSampler,
    "FramePackTorchCompileSettings": FramePackTorchCompileSettings,
    "FramePackFindNearestBucket": FramePackFindNearestBucket,
    "LoadFramePackModel": LoadFramePackModel,
    "FramePackVAELoader": FramePackVAELoader, # Add the new VAE loader
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadFramePackModel": "(Down)Load FramePackModel",
    "FramePackSampler": "FramePackSampler",
    "FramePackTorchCompileSettings": "Torch Compile Settings",
    "FramePackFindNearestBucket": "Find Nearest Bucket",
    "LoadFramePackModel": "Load FramePackModel",
    "FramePackVAELoader": "FramePack VAELoader", # Add the display name
    }
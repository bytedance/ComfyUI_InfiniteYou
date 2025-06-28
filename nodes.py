# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import folder_paths
import cv2
import numpy as np
from PIL import Image
import comfy
from huggingface_hub import snapshot_download, hf_hub_download
import shutil
import glob

from insightface.app import FaceAnalysis

from .utils import extract_arcface_bgr_embedding, tensor_to_np_image, np_image_to_tensor, resize_and_pad_pil_image, draw_kps, escape_path_for_url, init_arcface_model
from .infuse_net import load_infuse_net_flux
from .resampler import Resampler

folder_paths.add_model_folder_path("infinite_you", os.path.join(folder_paths.models_dir, "infinite_you"))

class FaceDetector:
    def __init__(self, 
                 det_sizes,
                 root_dir,
                 providers) -> None:
        self.apps = []
        for det_size in det_sizes:
            app = FaceAnalysis(name="antelopev2", root=root_dir, providers=providers)
            app.prepare(ctx_id=0, det_size=(det_size, det_size))
            self.apps.append(app)

    def __call__(self, np_image_bgr):
        for app in self.apps:
            faces = app.get(np_image_bgr)
            if len(faces) > 0:
                return faces
        return []

class IDEmbeddingModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_proj_model_name": (IDEmbeddingModelLoader.get_image_proj_names(), ),
                'image_proj_num_tokens': ([8, 16], ),
                'face_analysis_provider': (['CUDA', 'CPU'], ),
                'face_analysis_det_size': (["AUTO", "640", "320", "160"], )
            },
        }

    RETURN_NAMES = ("FACE_DETECTOR", "ARCFACE_MODEL", "IMAGE_PROJ_MODEL")
    RETURN_TYPES = ("MODEL", "MODEL", "MODEL")

    FUNCTION = "load_insightface"
    CATEGORY = "infinite_you"

    def get_image_proj_names():
        names = [
            os.path.join("sim_stage1", "image_proj_model.bin"),
            os.path.join("aes_stage2", "image_proj_model.bin"),
            *folder_paths.get_filename_list("infinite_you"),
        ]
        return list(filter(lambda x: x.endswith(".bin"), list(set(names))))

    def load_insightface(self, image_proj_model_name, image_proj_num_tokens, face_analysis_provider, face_analysis_det_size):
        insight_facedir = os.path.join(folder_paths.models_dir, "insightface")

        # Download insightface models
        antelopev2_dir = os.path.join(insight_facedir, 'models', 'antelopev2')
        if not os.path.exists(antelopev2_dir) or len(glob.glob(os.path.join(antelopev2_dir, "*.onnx"))) == 0:
            os.makedirs(antelopev2_dir, exist_ok=True)
            snapshot_download(repo_id="MonsterMMORPG/tools", allow_patterns="*.onnx", local_dir=antelopev2_dir)

        # Download infinite you models
        infinite_you_dir = os.path.join(folder_paths.models_dir, "infinite_you")
        image_proj_model_path = os.path.join(infinite_you_dir, image_proj_model_name)
        if not os.path.exists(image_proj_model_path):
            dst_dir = os.path.dirname(image_proj_model_path)
            os.makedirs(dst_dir, exist_ok=True)

            downloaded_file = hf_hub_download(repo_id="ByteDance/InfiniteYou", 
                            filename=escape_path_for_url(os.path.join("infu_flux_v1.0", image_proj_model_name)),
                            local_dir=infinite_you_dir)
            shutil.move(downloaded_file, image_proj_model_path)

        provider = 'CPUExecutionProvider'
        if face_analysis_provider == 'CUDA':
            provider = 'CUDAExecutionProvider'
        det_sizes = []
        if face_analysis_det_size == 'AUTO':
            det_sizes = [640, 320, 160]
        else:
            det_sizes = [int(face_analysis_det_size)]
        face_detector = FaceDetector(det_sizes=det_sizes, root_dir=insight_facedir, providers=[provider])

        device = comfy.model_management.get_torch_device()

        # Load arcface model
        arcface_model = init_arcface_model(device=device)

         # Load image proj model
        image_emb_dim = 512
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=image_proj_num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=4096,
            ff_mult=4,
        )
        ipm_state_dict = torch.load(image_proj_model_path, map_location="cpu")
        image_proj_model.load_state_dict(ipm_state_dict['image_proj'])
        del ipm_state_dict
        image_proj_model.to(device, torch.bfloat16)
        image_proj_model.eval()

        return (face_detector, arcface_model, image_proj_model)

class ExtractFacePoseImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_detector": ("MODEL", ),
                "image": ("IMAGE", ),
                "width": ("INT", {"default": 864, "min": 0, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 1152, "min": 0, "max": 2048, "step": 1}),
            },
            "optional": {
                "mask": ("MASK", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extract_face_pose"
    CATEGORY = "infinite_you"

    def extract_face_pose(self, face_detector, image, width, height, mask = None):
        np_image = tensor_to_np_image(image)[0]
        if mask is not None:
            np_mask = tensor_to_np_image(mask)[0]
            np_mask = cv2.resize(np_mask, (np_image.shape[1], np_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_3ch = np.expand_dims(np_mask, axis=-1)
            mask_3ch = np.repeat(mask_3ch, 3, axis=-1)  # Shape: (H, W, 3)
            np_image = np_image * mask_3ch

        pil_image = resize_and_pad_pil_image(Image.fromarray(np_image), (width, height))        
        face_info = face_detector(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
        if len(face_info) == 0:
            raise ValueError('No face detected in the input pose image')
        
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        pil_image = draw_kps(pil_image, face_info['kps'])
        
        return (np_image_to_tensor(np.array(pil_image)).unsqueeze(0), )

class ExtractIDEmbedding:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_detector": ("MODEL", ),
                "arcface_model": ("MODEL", ),
                "image_proj_model": ("MODEL", ),
                "image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "extract_id_embedding"
    CATEGORY = "infinite_you"
    
    def extract_id_embedding(self, face_detector, arcface_model, image_proj_model, image):
        np_image = tensor_to_np_image(image)
        id_image_cv2 = cv2.cvtColor(np_image[0], cv2.COLOR_RGB2BGR)
        face_info = face_detector(id_image_cv2)
        if len(face_info) == 0:
            raise ValueError('No face detected in the input ID image')
        
        device = comfy.model_management.get_torch_device()

        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        landmark = face_info['kps']
        id_embed = extract_arcface_bgr_embedding(id_image_cv2, landmark, arcface_model, device=device)
        id_embed = id_embed.clone().unsqueeze(0).float()
        id_embed = id_embed.reshape([1, -1, 512])
        id_embed = id_embed.to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            id_embed = image_proj_model(id_embed)
            bs_embed, seq_len, _ = id_embed.shape
            id_embed = id_embed.repeat(1, 1, 1)
            id_embed = id_embed.view(bs_embed * 1, seq_len, -1)
            id_embed = id_embed.to(device=device, dtype=torch.bfloat16)
            
        return ({'id_embedding': id_embed}, )

class InfuseNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "controlnet_name": (InfuseNetLoader.get_controlnet_names(), )}}

    def get_controlnet_names():
        names = [
            os.path.join("sim_stage1", "infusenet_sim_bf16.safetensors"),
            os.path.join("sim_stage1", "infusenet_sim_fp8e4m3fn.safetensors"),
            os.path.join("aes_stage2", "infusenet_aes_bf16.safetensors"),
            os.path.join("aes_stage2", "infusenet_aes_fp8e4m3fn.safetensors"),
            *folder_paths.get_filename_list("infinite_you"),
        ]
        return list(filter(lambda x: x.endswith(".safetensors"), list(set(names))))

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "infinite_you"

    def load_controlnet(self, controlnet_name):
        infinite_you_dir = os.path.join(folder_paths.models_dir, "infinite_you")
        controlnet_path = os.path.join(infinite_you_dir, controlnet_name)

        if not os.path.exists(controlnet_path):
            dst_dir = os.path.dirname(controlnet_path)
            os.makedirs(dst_dir, exist_ok=True)
            downloaded_file = hf_hub_download(repo_id="ByteDance/InfiniteYou", 
                            filename=escape_path_for_url(os.path.join("infu_flux_v1.0", controlnet_name)),
                            local_dir=infinite_you_dir)
            
            shutil.move(downloaded_file, controlnet_path)
        
        controlnet = load_infuse_net_flux(controlnet_path)
        return (controlnet,)

class InfuseNetApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "id_embedding": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "image": ("IMAGE", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                             },
                "optional": {
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "control_mask": ("MASK", ),
                             }
    }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"

    CATEGORY = "infinite_you"

    def apply_controlnet(self, positive, id_embedding, control_net, image, strength, start_percent, end_percent, negative = None, vae=None, control_mask=None, extra_concat=[]):
        if strength == 0:
            return (positive, negative)

        if control_mask is not None:
            if control_mask.dim() > 3:
                control_mask = control_mask.squeeze(-1)
            elif control_mask.dim() < 3:
                control_mask = control_mask.unsqueeze(0)

        control_hint = image.movedim(-1,1)
        cnets = {}
        
        out = []
        for conditioning in [positive, negative]:
            c = []
            if conditioning is None:
                out.append(None)
                continue

            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae=vae, extra_concat=extra_concat)
                    c_net.id_embedding = id_embedding['id_embedding']
                    c_net.set_previous_controlnet(prev_cnet)
                    c_net.set_extra_arg("control_mask", control_mask)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])

NODE_CLASS_MAPPINGS = {
    "IDEmbeddingModelLoader": IDEmbeddingModelLoader,
    "ExtractIDEmbedding": ExtractIDEmbedding,
    "ExtractFacePoseImage": ExtractFacePoseImage,
    "InfuseNetApply": InfuseNetApply,
    "InfuseNetLoader": InfuseNetLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IDEmbeddingModelLoader": "ID Embedding Model Loader",
    "ExtractIDEmbedding": "Extract ID Embedding",
    "ExtractFacePoseImage": "Extract Face Pose Image",
    "InfuseNetApply": "Apply InfuseNet",
    "InfuseNetLoader": "Load InfuseNet",
}

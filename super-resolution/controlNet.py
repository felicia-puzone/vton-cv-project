
import torch
from PIL import Image
from diffusers import ControlNetModel, DiffusionPipeline
from diffusers.utils import load_image

def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile', 
                                             torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                         custom_pipeline="stable_diffusion_controlnet_img2img",
                                         controlnet=controlnet,
                                         torch_dtype=torch.float16).to('cuda')
#pipe.enable_xformers_memory_efficient_attention()

source_image = load_image('test8.png')

condition_image = resize_for_condition_image(source_image, 1024)
 image = pipe(prompt="best quality, clothes, garment, model, shop, upper clothes",
             negative_prompt="blur, lowres, bad anatomy, bad clothes, bad hands, cropped, worst quality, fading, glitch, robot, tech, high saturation, medieval",
             image=condition_image, 
             controlnet_conditioning_image=condition_image, 
             width=condition_image.size[0],
             height=condition_image.size[1],
             strength=1.0,
             generator=torch.manual_seed(0),
             num_inference_steps=14,
            ).images[0]

image.save('output2.png')

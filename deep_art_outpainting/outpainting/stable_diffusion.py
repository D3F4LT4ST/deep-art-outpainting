import os
import torch
import PIL
from diffusers import StableDiffusionInpaintPipeline
from argparse import ArgumentParser

def outpaint_stable_diffusion(
        data_path: str,
        output_path: str,
        style: str,
    ):
    '''
    Performs outpainting with RunwayML Stable Diffusion inpainting model

    Args:
        data_path: path to data directory
        output_path: path to output directory
        style: style of the paintings
    '''
    os.makedirs(output_path, exist_ok=True)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting',
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to('cuda')

    for fname in filter(lambda fname: not fname.split(".")[0].endswith('mask'), os.listdir(data_path)):

        orig_img= PIL.Image.open(os.path.abspath(data_path + f'/{fname}')).convert("RGB")
        mask = PIL.Image.open(os.path.abspath(data_path + f'/{fname.split(".")[0]}_mask.png')).convert("RGB")

        orig_width, orig_height, = orig_img.size

        image = orig_img.resize((512, 512))
        mask_image = mask.resize((512, 512))

        img_outpainted = pipe(
            prompt=f'a painting in {style} style',
            image=image,
            mask_image=mask_image,
            guidance_scale=7.5,
            generator=torch.Generator(device="cuda").manual_seed(0),
            num_images_per_prompt=1,
        ).images[0]

        img_outpainted = img_outpainted.resize((orig_width, orig_height))

        img_outpainted.save(f'{output_path}/{fname}')

if __name__ == '__main__':
    parser = ArgumentParser(description="Stable Diffusion outpainting.")

    parser.add_argument('-data-path', type=str, required=True)
    parser.add_argument('-output-path', type=str, required=True) 
    parser.add_argument('-style', type=str, required=False)

    args = parser.parse_args()

    outpaint_stable_diffusion(**vars(args))






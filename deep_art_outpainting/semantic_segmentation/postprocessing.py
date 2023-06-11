import os
import cv2 as cv
import numpy as np
from argparse import ArgumentParser

def postprocess_masked_data(
        data_path: str,
        output_path: str,
        size: int=512
    ):
    '''
    Postprocesses masked data to contain binary masks and match outpainting resolution requirements.

    Args:
        data_path: path to data directory
        output_path: path to output directory
        int: smallest side size for images
    '''
    os.makedirs(output_path, exist_ok=True)

    for fname in filter(lambda fname: not fname.split(".")[0].endswith('mask'), os.listdir(data_path)):

        orig_img = cv.imread(os.path.abspath(data_path + f'/{fname}'))
        orig_mask = cv.imread(os.path.abspath(data_path + f'/{fname.split(".")[0]}_mask.png'), cv.IMREAD_GRAYSCALE) 

        orig_height, orig_width, _ = orig_img.shape
        aspect_ratio = orig_width / orig_height

        smallest_dim = min(orig_width, orig_height)
        if smallest_dim == orig_height:
            height_resized = size
            width_resized = round(size * aspect_ratio)
        else:
            width_resized = size
            height_resized = round(size / aspect_ratio)

        interp = cv.INTER_AREA if smallest_dim >= 512 else cv.INTER_CUBIC
        img_resized = cv.resize(orig_img, (width_resized, height_resized), interpolation=interp)
        mask_resized = cv.resize(orig_mask, (width_resized, height_resized), interpolation=interp)

        bin_mask_resized = np.where(mask_resized > 0, 0, 255)

        cv.imwrite(f'{output_path}/{fname.split(".")[0]}.png', img_resized)
        cv.imwrite(f'{output_path}/{fname.split(".")[0]}_mask.png', bin_mask_resized)

if __name__ == '__main__':
    parser = ArgumentParser(description="Semantic segmentation postprocessing.")

    parser.add_argument('-data-path', type=str, required=True)
    parser.add_argument('-output-path', type=str, required=True)

    args = parser.parse_args()

    postprocess_masked_data(**vars(args))

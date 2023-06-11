import os
import cv2 as cv
from argparse import ArgumentParser

def prepare_sr_dataset(
        data_path: str,
        hr_size: int=2048, 
        lr_factor: int=4 
    ):
    '''
    Prepares WikiArt super resolution dataset by creating high and low resolution versions of the images.

    Args:
        data_path: path to dataset directory
        hr_size: high resolution image size (smallest dimension)
        lr_factor: low resolution downscaling factor
    '''
    for subset in ['train', 'valid', 'test']:

        orig_subset_path = os.path.abspath(data_path + f'/WikiArtSR/{subset}')
        hr_subset_path = os.path.abspath(data_path + f'/WikiArtSR_HR/{subset}')
        lr_subset_path = os.path.abspath(data_path + f'/WikiArtSR_LR/{subset}')

        os.makedirs(hr_subset_path, exist_ok=True)
        os.makedirs(lr_subset_path, exist_ok=True)

        for fname in filter(lambda fname: not fname.startswith('.'), os.listdir(orig_subset_path)):
            
            orig_img = cv.imread(os.path.abspath(orig_subset_path + f'/{fname}'))

            height, width, _ = orig_img.shape
            aspect_ratio = width / height

            smallest_dim = min(width, height)
            if smallest_dim == height:
                height_hr = hr_size
                width_hr = round((hr_size * aspect_ratio / lr_factor)) * lr_factor
            else:
                width_hr = hr_size
                height_hr = round((hr_size / aspect_ratio / lr_factor)) * lr_factor
                
            hr_interp = cv.INTER_AREA if smallest_dim > hr_size else cv.INTER_CUBIC
            img_hr = cv.resize(orig_img, (width_hr, height_hr), interpolation=hr_interp)

            height_lr = height_hr // lr_factor
            width_lr = width_hr // lr_factor

            img_lr = cv.resize(img_hr, (width_lr, height_lr), interpolation=cv.INTER_AREA)

            cv.imwrite(os.path.abspath(hr_subset_path + f'/{fname.split(".")[0]}.jpg'), img_hr)
            cv.imwrite(os.path.abspath(lr_subset_path + f'/{fname.split(".")[0]}x{lr_factor}.jpg'), img_lr)

if __name__ == '__main__':
    parser = ArgumentParser(description="Super resolution dataset preparation.")

    parser.add_argument('-data-path', type=str, required=True)
    parser.add_argument('-hr-size', type=int, default=2048, required=False) 
    parser.add_argument('-lr-factor', type=int, default=4, required=False)

    args = parser.parse_args()

    prepare_sr_dataset(**vars(args))


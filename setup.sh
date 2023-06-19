eval "$(conda shell.bash hook)"

# Fetch data
mkdir -p deep-art-outpainting/data

# WikiArt Base
wget 'https://docs.google.com/uc?export=download&id=1Hgb6Spt2oJZI46-zMisRtNfrZkmMd-hS&confirm=t' -O WikiArt.zip
unzip WikiArt.zip
rm WikiArt.zip
mv WikiArt deep-art-outpainting/data/WikiArt

# DRAM
wget "https://docs.google.com/uc?export=download&id=1Cn1dzdmWpgzbXGSOhTif90j--71vK9WP&confirm=t" -O DRAM_processed.zip
unzip DRAM_processed.zip
mv DRAM_processed deep-art-outpainting/data/DRAM_500

# WikiArt SR
wget 'https://docs.google.com/uc?export=download&id=1bkQOuEBZW6rWrGEocz7ySIHCY-zgUvMr&confirm=t' -O WikiArtSR.zip
unzip WikiArtSR.zip
rm WikiArtSR.zip
mv WikiArtSR deep-art-outpainting/data/WikiArtSR


# Fetch weights & checkpoints
mkdir -p deep-art-outpainting/weights

# Semantic segmentation
wget "https://docs.google.com/uc?export=download&id=1OxXG18BtSvmN-dnau7C7Fvrq-IXgx8mQ&confirm=t" -O vgg_gatys.pth
mkdir deep-art-outpainting/libs/SemanticSegmentationInArtPaintings/gram_embedding/weights
mv vgg_gatys.pth deep-art-outpainting/libs/SemanticSegmentationInArtPaintings/gram_embedding/weights/vgg_gatys.pth

wget "https://docs.google.com/uc?export=download&id=1GvUGVRVRpK676Rb8WtEcZTfLRc6gGOHP&confirm=t" -O checkpoints.zip
unzip checkpoints.zip
rm checkpoints.zip
rm -r checkpoints/step1
rm -r checkpoints/step2/adv_pascal_sbd12_filtered_expressionism_lr2.5e4_source_prestyled
rm -r checkpoints/step2/adv_pascal_sbd12_filtered_post_impressionism_lr2.5e4_source_prestyled
mv checkpoints deep-art-outpainting/libs/SemanticSegmentationInArtPaintings/

# ArtificiallyInspired outpainting
wget 'https://docs.google.com/uc?export=download&id=1gS6gMSqcXlN_hDQJT5dXT0QOYp6AjvNZ&confirm=t' -O models.zip
unzip models.zip
rm track1_Places_comodgan.pkl  
rm track1_Places_unet.pth  
rm track1_FFHQ_comodgan.pkl  
rm track1_FFHQ_unet.pth  
mv *.pth deep-art-outpainting/libs/NTIRE22_Inpainting/inference/ArtificiallyInspired/track1/checkpoint
mv *.pkl deep-art-outpainting/libs/NTIRE22_Inpainting/inference/ArtificiallyInspired/track1/checkpoint

# Super resolution
mkdir -p deep-art-outpainting/weights/SR

wget 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rrdn-C4-D3-G32-G032-T10-x4-GANS/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5'
mv *.hdf5 deep-art-outpainting/weights/SR

wget 'https://docs.google.com/uc?export=download&id=19qzrIGWUW3eu5troIjHzhCIbTZONSRjX&confirm=t' -O 4xPSNR.pth
wget 'https://docs.google.com/uc?export=download&id=1IVJFmC9S6N9v7dpTuNuO0U3D0wxbKyau&confirm=t' -O ART_ESRGAN_x4_G.pth
mv *.pth deep-art-outpainting/weights/SR


# Setup environment

# Semantic segmentation
conda create --yes --name semantic_seg python=3.8.5
conda activate semantic_seg
pip install -r deep-art-outpainting/libs/SemanticSegmentationInArtPaintings/requirements.txt
pip install scikit-learn opencv-python

# Stable Diffusion outpainting
conda create --yes -n outpainting_sd python=3.9
conda activate outpainting_sd
pip install diffusers==0.11.1 transformers ftfy accelerate opencv-python

# Super resolution training
conda create --yes -n sr_train python=3.9
conda activate sr_train
pip install -r deep-art-outpainting/libs/traiNNer/requirements.txt
pip install -r deep-art-outpainting/libs/traiNNer/optional_requirements.txt

# Super resolution inference
conda create --yes -n sr_infer_esrgan python=3.9
conda activate sr_infer_esrgan
pip install -r deep-art-outpainting/libs/ESRGAN/requirements.txt

conda create --yes -n sr_infer_isr python=3.9
conda activate sr_infer_isr
cd deep-art-outpainting/libs/image-super-resolution/; python setup.py install
pip install opencv-python

# ArtificiallyInspired outpainting
conda create --yes -n outpainting_ai python=3.6
conda activate outpainting_ai 
pip install torch==1.10.1 torchvision==0.11.2 tensorflow==1.15.0 tqdm easydict pandas numpy requests
pip install opencv-python --verbose
conda install --yes -c conda-forge cudatoolkit=10.0 cudnn=7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# Fetch data
wget 'https://docs.google.com/uc?export=download&id=1YYO1eIJyfpIchl-0Z8owbO1SWVWhjwaX&confirm=t' -O WikiArtSR.zip
unzip WikiArtSR.zip
rm WikiArtSR.zip
mkdir -p data
mv WikiArtSR data/WikiArtSR

# Fetch weights
wget 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rrdn-C4-D3-G32-G032-T10-x4-GANS/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5'
mkdir -p weights/SR
mv *.hdf5 weights/SR

wget 'https://docs.google.com/uc?export=download&id=19qzrIGWUW3eu5troIjHzhCIbTZONSRjX&confirm=t' -O 4xPSNR.pth
wget 'https://docs.google.com/uc?export=download&id=1IVJFmC9S6N9v7dpTuNuO0U3D0wxbKyau&confirm=t' -O ART_ESRGAN_x4_G.pth
mv *.pth weights/SR

# Install dependencies
pip install -r requirements.txt
cd libs/image-super-resolution/; python setup.py install
pip install -r libs/traiNNer/requirements.txt
pip install -r libs/traiNNer/optional_requirements.txt
pip install -r libs/ESRGAN/requirements.txt
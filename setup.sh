# Fetch data
wget 'https://docs.google.com/uc?export=download&id=1YYO1eIJyfpIchl-0Z8owbO1SWVWhjwaX&confirm=t' -O WikiArtSR.zip
unzip WikiArtSR.zip
rm WikiArtSR.zip
mv WikiArtSR data/WikiArtSR

# Fetch weights
wget 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rrdn-C4-D3-G32-G032-T10-x4-GANS/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5'
mkdir weights/SR
mv *.hdf5 weights/SR

# Install dependencies
pip install -r requirements.txt
cd libs/image-super-resolution/; python setup.py install
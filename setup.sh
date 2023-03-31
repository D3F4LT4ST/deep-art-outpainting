# Fetch data
wget 'https://docs.google.com/uc?export=download&id=1YYO1eIJyfpIchl-0Z8owbO1SWVWhjwaX&confirm=t' -O WikiArtSR.zip
unzip WikiArtSR.zip
rm WikiArtSR.zip
mv WikiArtSR ../data/WikiArtSR

# Fetch weights
wget 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rdn-C6-D20-G64-G064-x2/PSNR-driven/rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5'
wget 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rdn-C3-D10-G64-G064-x2/PSNR-driven/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5'
wget 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rdn-C6-D20-G64-G064-x2/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5'
wget 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rrdn-C4-D3-G32-G032-T10-x4-GANS/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5'
mkdir ../weights/SR
mv *.hdf5 ../weights/SR

# Install dependencies
pip install -r requirements.txt
python libs/image-super-resolution/setup.py install
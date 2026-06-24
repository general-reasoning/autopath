import dbx

PANCAN_CPTAC_ROOT = dbx.env("PANCAN_CPTAC_ROOT", "/mnt/labshare/SLIDES/CPTAC_downloads")
PANCAN_CPTAC_SAMPLE = dbx.env("PANCAN_CPTAC_SAMPLE", "/mnt/labshare/SLIDES/CPTAC_downloads/LUAD_QUPATH/tfrecords/256px_256um/C3N-02242-22.tfrecords")
PANCAN_CPTAC_RESOLUTION = dbx.env("PANCAN_CPTAC_RESOLUTION", "256px_256um")

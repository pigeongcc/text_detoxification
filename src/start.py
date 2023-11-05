from data import data_downloads, data_preparation
from models import models_downloads


def main():
    data_downloads.download_dataset()   # download data archive
    data_downloads.extract_dataset_zip()    # unzip the archive

    data_preparation.prepare_data()  # process the dataset

    models_downloads.download_model()   # download the model weights archive
    models_downloads.extract_model_zip()    # unzip the archive


if __name__ == '__main__':
    main()
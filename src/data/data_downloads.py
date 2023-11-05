import requests
import zipfile


dataset_path = 'data/raw/'
dataset_zip_filename = "filtered_paranmt.zip"


def download_dataset():
    file_url = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'

    try:
        response = requests.get(file_url)
        
        if response.status_code == 200:
            print("Downloading filtered_paranmt.zip...")
            filepath = dataset_path + dataset_zip_filename
            with open(filepath, 'wb') as file:
                file.write(response.content)
            print(f'File filtered_paranmt.zip downloaded to {filepath}')
        else:
            print(f'Failed to download the file filtered_paranmt.zip. Status code: {response.status_code}')
    except Exception as e:
        print(f'An error occurred: {e}')


def extract_dataset_zip():
    filepath = dataset_path + dataset_zip_filename

    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        
        print(f'filtered_paranmt.zip extracted to {dataset_path}')
    except Exception as e:
        print(f'An error occurred: {e}')

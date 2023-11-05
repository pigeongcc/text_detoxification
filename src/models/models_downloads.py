import requests
import zipfile


models_path = 'models/'
t5_zip_filename = "t5-paraphrase.zip"


def download_model():
    file_url = 'https://drive.usercontent.google.com/download?id=1jZuelfXcRw3recfZWJ6LIBtj9YS5Uogf&export=download&authuser=0&confirm=t&uuid=8f252383-2507-47f1-8591-ae34d2981538&at=APZUnTUv6SxGHFG2tRkEkuTb75x5:1699215050917'

    try:
        response = requests.get(file_url)
        
        if response.status_code == 200:
            print("Downloading t5-paraphrase.zip...")
            filepath = models_path + t5_zip_filename
            with open(filepath, 'wb') as file:
                file.write(response.content)
            print(f'File t5-paraphrase.zip downloaded to {filepath}')
        else:
            print(f'Failed to download the file t5-paraphrase.zip. Status code: {response.status_code}')
    except Exception as e:
        print(f'An error occurred: {e}')


def extract_model_zip():
    filepath = models_path + t5_zip_filename

    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(models_path)
        
        print(f't5-paraphrase.zip extracted to {models_path}')
    except Exception as e:
        print(f'An error occurred: {e}')

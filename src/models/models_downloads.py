import requests
import zipfile


models_path = 'models/'
t5_zip_filename = "t5-paraphrase.zip"


def download_model():
    file_url = 'https://doc-14-58-docs.googleusercontent.com/docs/securesc/aqclo059d265do7m8bo1gcua1goo9f9k/77mp10ip0er8buqgqccbg679d5noa4qj/1699211925000/04388380893538145158/17224040402858289494Z/1jZuelfXcRw3recfZWJ6LIBtj9YS5Uogf?e=download&uuid=023d654e-f7ad-48d5-a9c2-530569252f8a&nonce=7gg79fl71o9ii&user=17224040402858289494Z&hash=h3vkdpa3136nfap0boonhut7dcjcrpqh'

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

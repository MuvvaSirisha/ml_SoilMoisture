from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

import os

# Load .env if present (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==========================================
# AUTH
# ==========================================

gauth = GoogleAuth()

scope = ['https://www.googleapis.com/auth/drive']

_KEY_PATH = os.environ.get('GOOGLE_SERVICE_ACCOUNT_KEY', 'cloud/service_account.json')

gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
    _KEY_PATH,
    scope
)

drive = GoogleDrive(gauth)

print("Authenticated Successfully.\n")

# ==========================================
# GOOGLE DRIVE LITERATURE FOLDER ID
# ==========================================

FOLDER_ID = "1DVxGBgN5FUr_hnEO_kOs0i6KHOcSjjNc"

# ==========================================
# CACHE FOLDER
# ==========================================

CACHE_FOLDER = r"cache\literature"

os.makedirs(CACHE_FOLDER, exist_ok=True)

def sync_literature():

    print("\nSyncing literature from Google Drive...")

    file_list = drive.ListFile({
        'q': f"'{FOLDER_ID}' in parents and trashed=false"
    }).GetList()

    for file in file_list:
        filename = file['title']
        save_path = os.path.join(CACHE_FOLDER, filename)

        if not os.path.exists(save_path):
            print(f"Downloading {filename}...")
            file.GetContentFile(save_path)
            
    return CACHE_FOLDER

if __name__ == "__main__":
    cache_path = sync_literature()
    print(f"All literature files cached successfully in {cache_path}")
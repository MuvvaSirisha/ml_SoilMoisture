from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

import xarray as xr
import os
import zipfile

# Load .env if present (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==========================================
# GOOGLE DRIVE AUTH
# ==========================================

gauth = GoogleAuth()

scope = ['https://www.googleapis.com/auth/drive']

_KEY_PATH = os.environ.get('GOOGLE_SERVICE_ACCOUNT_KEY', 'cloud/service_account.json')

gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
    _KEY_PATH,
    scope
)

drive = GoogleDrive(gauth)

# ==========================================
# GOOGLE DRIVE FOLDER ID
# ==========================================

FOLDER_ID = "1Wz6NvuB12Aa0PM5s6v4IYgEtufnb-YAF"

# ==========================================
# CACHE FOLDER
# ==========================================

CACHE_FOLDER = r"cache\zarr"

os.makedirs(CACHE_FOLDER, exist_ok=True)

# ==========================================
# DOWNLOAD YEAR IF MISSING
# ==========================================

def download_year(year):

    zarr_folder = os.path.join(
        CACHE_FOLDER,
        f"{year}_v2.zarr"
    )

    # already cached
    if os.path.exists(zarr_folder):

        print(f"{year} already cached.")

        return zarr_folder

    target_file = f"{year}_v2.zarr.zip"

    print(f"\nSearching Drive for {target_file}...")

    file_list = drive.ListFile({
        'q': f"'{FOLDER_ID}' in parents and trashed=false"
    }).GetList()

    target = None

    for file in file_list:

        if file['title'] == target_file:
            target = file
            break

    if target is None:
        raise FileNotFoundError(
            f"{target_file} not found in Drive."
        )

    # ==========================================
    # DOWNLOAD ZIP
    # ==========================================

    zip_path = os.path.join(
        CACHE_FOLDER,
        target_file
    )

    print(f"Downloading {target_file}...")

    target.GetContentFile(zip_path)

    print("Download complete.")

    # ==========================================
    # EXTRACT ZIP
    # ==========================================

    print("Extracting ZIP...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(CACHE_FOLDER)

    print("Extraction complete.")

    return zarr_folder

# ==========================================
# LOAD YEAR DATASET
# ==========================================

def load_year_dataset(year):

    zarr_path = download_year(year)

    print(f"\nLoading {year} dataset...")

    ds = xr.open_zarr(
        zarr_path,
        consolidated=True
    )

    print("Dataset loaded successfully.")

    return ds
# ==========================================
# LOAD MULTIPLE YEARS
# ==========================================

def load_multiple_years(start_year, end_year):

    datasets = []

    for year in range(start_year, end_year + 1):

        print(f"\nProcessing year {year}...")

        ds = load_year_dataset(str(year))

        datasets.append(ds)

    print("\nCombining datasets...")

    combined = xr.concat(
        datasets,
        dim="time"
    )

    print("Combined dataset ready.")

    return combined

def get_full_dataset():

    print("\nSyncing datasets from Google Drive...")

    file_list = drive.ListFile({
        'q': f"'{FOLDER_ID}' in parents and trashed=false"
    }).GetList()

    years = []
    for file in file_list:
        title = file['title']
        if title.endswith("_v2.zarr.zip"):
            year_str = title.split("_")[0]
            if year_str.isdigit():
                years.append(int(year_str))
                download_year(year_str)

    if not years:
        raise FileNotFoundError("No Zarr datasets found in Drive.")

    years.sort()
    start_year = years[0]
    end_year = years[-1]

    return load_multiple_years(start_year, end_year)

# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    ds = load_multiple_years(2010,2012)

    print("\n")
    print(ds)
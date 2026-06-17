from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

import geopandas as gpd
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
# GOOGLE DRIVE SHAPEFILE FOLDER ID
# ==========================================

FOLDER_ID = "1Ee12yCV_MloYk6wtna1W4BtXal9Xc4nX"

# ==========================================
# CACHE FOLDER
# ==========================================

CACHE_FOLDER = r"cache\shapefiles"

os.makedirs(CACHE_FOLDER, exist_ok=True)

def get_shapefile_path():

    print("\nSyncing shapefiles from Google Drive...")

    file_list = drive.ListFile({
        'q': f"'{FOLDER_ID}' in parents and trashed=false"
    }).GetList()

    shp_filename = None

    for file in file_list:
        filename = file['title']
        if filename.endswith(".shp"):
            shp_filename = filename

    if shp_filename is None:
        raise FileNotFoundError(".shp file not found in Drive.")

    required_extensions = [
        ".shp", ".shx", ".dbf", ".prj", ".cpg"
    ]

    for file in file_list:
        filename = file['title']
        for ext in required_extensions:
            if filename.endswith(ext):
                save_path = os.path.join(
                    CACHE_FOLDER,
                    filename
                )
                if not os.path.exists(save_path):
                    print(f"Downloading {filename}...")
                    file.GetContentFile(save_path)

    local_shp_path = os.path.join(
        CACHE_FOLDER,
        shp_filename
    )
    
    return local_shp_path

if __name__ == "__main__":
    local_shp_path = get_shapefile_path()
    print("\nLoading shapefile...\n")
    gdf = gpd.read_file(local_shp_path)
    print(gdf.head())
    print("\nCRS:")
    print(gdf.crs)
    print("\nShapefile loaded successfully.")
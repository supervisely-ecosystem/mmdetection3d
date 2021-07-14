
from upload_api import UploadAPI



if __name__ == '__main__':
    up = UploadAPI(project_id=5268, ds_name='pointcloud')
    up.download_dataset(f"supervisely_dataset", "test")

    up = UploadAPI(project_id=5268, ds_name='pointcloud_012')
    up.download_dataset(f"supervisely_dataset", "train")




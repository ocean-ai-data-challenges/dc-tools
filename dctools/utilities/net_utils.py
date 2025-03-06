"""NET Utilities functions."""

from mypy_boto3_s3.client import S3Client
from pathlib import Path
from typing import List
from urllib.parse import urlparse 


def download_s3_file(
    s3_client: S3Client, bucket_name: str, file_name: str, local_file_path: str
) -> None:
    """Download a file from s3 server.

    Args:
        s3_client: (S3Client) boto3 S3 client
        bucket_name(str): name of s3 bucket
        filename(str): file to download from bucket
        outpath(str): path where to save the downloaded file
    """
    try:
        s3_client.download_file(
            Bucket=bucket_name,
            Key=file_name,
            Filename=local_file_path,
        )
    except Exception as error:
            print(f"Error when downloading from s3 : {error}")


def get_file_folders(s3_client, bucket_name, prefix=""):
    """Get files and folders names in a S3 bucket.

    Args:
        s3_client(typing.Type[S3Client]): boto3 S3 client instance
        bucket_name(str): name of S3 bucket
    Return:
        file_names(List[str]): list of filenames
        folders(List[str]): list of folders names
    """
    file_names = []
    folders = []

    default_kwargs = {
        "Bucket": bucket_name,
        "Prefix": prefix
    }
    next_token = ""

    while next_token is not None:
        updated_kwargs = default_kwargs.copy()
        if next_token != "":
            updated_kwargs["ContinuationToken"] = next_token

        response = s3_client.list_objects_v2(**default_kwargs)
        contents = response.get("Contents")

        for result in contents:
            key = result.get("Key")
            if key[-1] == "/":
                folders.append(key)
            else:
                file_names.append(key)

        next_token = response.get("NextContinuationToken")

    return file_names, folders


def download_files(
        s3_client: S3Client,
        bucket_name: str,
        local_path: str,
        file_names: List[str],
        folders: List[Path],
    ):
    """Download all listed files and folders in a S3 bucket.

    Args:
        s3_client(S3Client): boto3 S3 client instance
        bucket_name(str): name of S3 bucket
        local_path(str): local path where to store files
        file_names(List[str]): list of files to download
        folders(List[Path]): list of folders to download
    """
    localpath = Path(local_path)

    for folder in folders:
        folder_path = Path.joinpath(localpath, folder)
        folder_path.mkdir(parents=True, exist_ok=True)

    for file_name in file_names:
        file_path = Path.joinpath(localpath, file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(
            bucket_name,
            file_name,
            str(file_path)
        )


class S3Url(object):
    """Class to manipulate S3 urls."""

    def __init__(self, url):
        """Init.
    
        Args:
            url(str): S3 url
        """
        self._parsed = urlparse(url, allow_fragments=False) 

    @property
    def bucket(self):
        """Get bucket name from url.

        Return:
            (str): bucket name
        """
        return self._parsed.netloc

    @property 
    def key(self):
        """Get file key from url.

        Return:
            (str): file key
        """
        if self._parsed.query: 
            return self._parsed.path.lstrip("/") + "?" + self._parsed.query 
        else: 
            return self._parsed.path.lstrip("/") 

    @property 
    def url(self):
        """Get url.

        Return:
            (str): url
        """
        return self._parsed.geturl()


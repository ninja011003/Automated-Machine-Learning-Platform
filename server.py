
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
import os
import json

def upload_file(file_path):
    # Load credentials from token.json
    creds = load_credentials()

    # Build the Drive service
    service = build('drive', 'v3', credentials=creds)

    # Get the file name from the file path
    file_name = os.path.basename(file_path)

    # Create file metadata
    file_metadata = {'name': file_name,
                     'parents': ['15lW-Io7wQs33OYGqWRI2j1dA2dUmvijq']
                     }

    # Create media file upload
    media = MediaFileUpload(file_path, resumable=True)

    # Upload the file
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print("File uploaded successfully. File ID:", file.get('id'))

def load_credentials():
    # Load credentials from token.json
    with open('token.json', 'r') as token_file:
        creds_data = json.load(token_file)
        return Credentials.from_authorized_user_info(creds_data)


def intialize_user(emailid,password):

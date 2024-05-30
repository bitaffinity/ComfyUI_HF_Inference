"""Session handler"""
import os
import requests
session = requests.Session()
if "HF_AUTH_TOKEN" in os.environ:
    session.headers.update({
        "Authorization": f"Bearer {os.environ['HF_AUTH_TOKEN']}",
    })
else:
    print("No 'HF_AUTH_TOKEN' set.")
"""Session handler"""
import os
import requests
from time import sleep
import logging

session = requests.Session()
if "HF_AUTH_TOKEN" in os.environ:
    session.headers.update({
        "Authorization": f"Bearer {os.environ['HF_AUTH_TOKEN']}",
    })
else:
    logging.warning("No 'HF_AUTH_TOKEN' set.")

def post(url, **kwargs):

    if not url.startswith('http'):
        url = f'https://api-inference.huggingface.co/models/{url}'

    response = session.post(url, **kwargs)

    if response.status_code != 200:
        if 'estimated_time' in response.text:
            estimated_time = response.json()['estimated_time']
            model_path = '/'.join(url.split('/')[-2:])
            logging.info(f'Waiting for {estimated_time/60} minutes to load {model_path}')
            sleep(estimated_time)
            return post(url, **kwargs)
        raise Exception(response.text)
    return response
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "requests",
#     "tqdm",
# ]
# ///

import datetime
import json
import logging
import os
import signal
import sys

import click
import requests
from tqdm import tqdm


def download_file(file_url, file_path, logger, timeout):
    try:
        response = requests.get(file_url, allow_redirects=True, timeout=timeout, stream=True)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                total_size = int(response.headers.get('content-length', 0))
                logger.info(f"Downloading file {file_path} ({total_size / 1e6} Mbytes)")
                with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=file_path, miniters=1, file=sys.stdout) as t:
                    for data in response.iter_content(chunk_size=4096):
                        f.write(data)
                        t.update(len(data))
    except Exception as e:
        logger.error(f"Error downloading {file_url}: {e}")


def make_logger(model_name):
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{model_name.replace('/', '_')}_{current_time}.log"))
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


@click.command()
@click.option("-o", "--output-directory", required=True, type=click.Path(), help="The folder where models will be saved.")
@click.option("--models-file", required=True, type=click.Path(exists=True), help="The JSON file containing the models to download.")
@click.option("--timeout", default=30, show_default=True, help="Download timeout in seconds.")
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
def main(output_directory, models_file, timeout, verbose):
    """Download models from Hugging Face's model hub."""
    signal.signal(signal.SIGINT, lambda sig, frame: (logging.info("Interrupted by user. Exiting..."), sys.exit(0)))

    with open(models_file, 'r') as f:
        models = json.load(f)

    if not models:
        click.echo("No models found.")
        return

    for model in models:
        model_name = model['name']
        dir_name = model_name.replace('/', '_')
        output_dir = os.path.join(output_directory, dir_name)
        logger = make_logger(model_name)

        for branch in model['branches']:
            for file in branch['files']:
                file_url = f"https://huggingface.co/{model_name}/resolve/{branch['name']}/{file}"
                file_path = os.path.join(output_dir, branch['name'], file)
                if os.path.exists(file_path):
                    logger.info(f"File {file_path} already exists. Skipping...")
                    continue
                download_file(file_url, file_path, logger, timeout)


if __name__ == "__main__":
    main()

import logging
import os
from datetime import datetime

from constants import LOGS_DIR


os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(LOGS_DIR, f"{datetime.now().strftime('%d-%m-%Y')}-app.log")),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


def read_urls_from_file(file_path):
    with open(file_path, 'r') as file:
        urls = [line.strip() for line in file.readlines()]
    return urls
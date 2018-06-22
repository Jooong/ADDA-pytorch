from __future__ import print_function

import os
from os.path import dirname, abspath, join, exists
import json
import logging
import numpy as np
from datetime import datetime

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def prepare_logger(config):

    BASE_DIR = dirname(abspath(__file__))
    LOG_DIR = join(BASE_DIR, 'logs')
    if not exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    model_name = "SOURCE" if config.is_train_source else "ADAPT"
        
    log_filename='{model_name}-{datetime}.log'.format(model_name=model_name, datetime=get_time())
    log_filepath = join(LOG_DIR, log_filename)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    fileHandler = logging.FileHandler(log_filepath.format(datetime=datetime.now()))
    streamHandler = logging.StreamHandler(os.sys.stdout)

    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    logger.setLevel(logging.INFO)
    
    return logger

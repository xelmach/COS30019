"""
Logging configuration for TBRGS.
"""

#105106819 Suman Sutparai
# Logger for TBRGS
import logging
import os
from datetime import datetime

def setup_logger(log_dir="logs", log_file=None):
    """Setup and configure logger.
    
    Args:
        log_dir (str): Directory for logs.
        log_file (str): Name of the log file.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if log_file is None:
        log_file = datetime.now().strftime("%Y-%m-%d.log")
    log_path = os.path.join(log_dir, log_file)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger 
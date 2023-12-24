
import logging

'''# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)'''

'''logging.basicConfig(
    filename='app.log', # write to this file
    filemode='w', # open in append mode
    format='%(name)s - %(levelname)s - %(message)s'
    )'''

'''logging.getLogger("urllib3.connectionpool").disabled = True
logging.getLogger("filelock").disabled = True'''

def get_file_handler(file_name):
    file_handler = logging.FileHandler(file_name)
    _log_format = '%(asctime)s %(name)s - %(levelname)s - %(message)s'
    file_handler.setFormatter(logging.Formatter(_log_format))
    return file_handler

log_file_dict = {
    'info_logger': 'info.log',
    'time_logger': 'time.log',
    'transition_logger': 'transition.log'
}

def logger_creating():
    for log_name, log_file_name in log_file_dict.items():
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.INFO)
        logger.setLevel(logging.DEBUG)
        log_file = log_file_name
        logger.addHandler(get_file_handler(log_file))
        logger.propagate = False

#from pathlib import Path
#print(Path('/content/app.log').read_text())
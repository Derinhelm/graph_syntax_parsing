
import logging
logger = logging.getLogger('my_logger')

# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename='app.log', # write to this file
    filemode='w', # open in append mode
    format='%(name)s - %(levelname)s - %(message)s'
    )

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().setLevel(logging.DEBUG)

logging.getLogger("urllib3.connectionpool").disabled = True
logging.getLogger("filelock").disabled = True

logging.warning('This will get logged to a file')
logging.warning('New warning')
logging.debug('New debug')
logging.info('New info')


#from pathlib import Path
#print(Path('/content/app.log').read_text())
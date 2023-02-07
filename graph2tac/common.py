import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level = logging.DEBUG,
    format = '%(name)s:%(levelname)s - %(message)s'
)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger('graph2tac')

logging.addLevelName(15, "VERBOSE")  # between debug and info
logger.verbose = lambda message: logger.log(level=15, msg=message)

logging.addLevelName(25, "SUMMARY")  # between info and warning
logger.summary = lambda message: logger.log(level=25, msg=message)

logger.setLevel(logging.INFO)

import logging

FORMAT = '%(asctime)s:%(process)d:%(levelname)s::%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("visualization")
logger.setLevel(logging.INFO)

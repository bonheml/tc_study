import logging

FORMAT = '%(asctime)s:%(process)d:%(levelname)s::%(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("truncation_experiment")
logger.setLevel(logging.INFO)

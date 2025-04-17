import logging
import os
import json
import sys
from datetime import datetime

def load_config(config_file='config.json'):
    """Load configuration from the given JSON file."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file)
    if not os.path.exists(config_path):
        config_path = os.path.join(os.getcwd(), config_file)
    with open(config_path, 'r') as file:
        return json.load(file)

def setup_logger(config_file='config.json'):
    """Set up the logger to output to stdout and stderr for SLURM based on log level."""
    # Load configuration
    try:
        config = load_config(config_file)
        logger_config = config['logger']
        model_config = config['model']
        task_config = config['task']
    except Exception as e:
        print(f"Erreur lors du chargement de config.json : {e}", file=sys.stderr)
        raise

    # Extract logger settings
    log_level = logger_config.get('log_level', 'INFO').upper()
    model_version = logger_config.get('model_version', 'Not specified')
    model_name = model_config.get('model_name', 'UnknownModel')
    task_type = task_config.get('task_type', 'train')

    # Set up the logger
    logger = logging.getLogger('VideoClassification')
    logger.setLevel(getattr(logging, log_level, logging.INFO))  # Niveau global du logger

    # Supprime les handlers existants
    if logger.handlers:
        logger.handlers.clear()

    # Handler pour stdout (DEBUG et INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)  # Filtre jusqu'à INFO

    # Handler pour stderr (WARNING, ERROR, CRITICAL)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)  # Capture WARNING et plus

    # Formatter pour les messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    # Ajoute les handlers
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    # Messages initiaux
    logger.info("Logging initialized. Normal logs to .out, warnings/errors to .err")
    logger.info(f'Model version: {model_version}')
    
    return logger
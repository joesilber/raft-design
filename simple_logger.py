# -*- coding: utf-8 -*-
"""
Some logging boilerplate code.
"""
# set up a log file
import logging
import time

def start_logger(paths=None):
    '''Initializes a logger which will output log files to one or more paths.
    
    Returns a tuple:
        tuple[0] ... logger instance
        tuple[1] ... file handler associated to logger
        tuple[2] ... stream handler associated to logger
    
    If no paths argument, then logs only to shell.'''
    clear_logger()
    global logger
    if not paths:
        paths = []
    if isinstance(paths, str):
        paths = [paths]
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    [logger.removeHandler(h) for h in logger.handlers]
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S %z')
    formatter.converter = time.gmtime
    for p in paths:
        fh = logging.FileHandler(filename=p, mode='a', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    for path in paths:
        logger.info(f'Logging to {path}')
    return logger, fh, sh

def clear_logger():
    '''Mysteriously (to me) this doesn't seem to work on the second run of script in
    the same ipython/debugger session, but does seem to work on the third... Weird,
    not gonna bother trying to track it down --- just adds duplicate printouts.'''
    if 'logger' not in locals():
        return
    for h in logger.handlers:
        logger.removeHandler(h)
    
def assert2(test, message, show_quit_msg=True):
    '''Like an assert, but cleaner handling of logging.'''
    if not test:
        logger.error(message)
        if show_quit_msg:
            logger.warning('Now quitting, so user can check inputs.')
        assert False  # for ease of jumping back into the error state in ipython debugger
        
def input2(message):
    '''Wrapper for input which will log the interaction.'''
    logger.info(f'PROMPT: {message}')
    user_input = input('>>> ')
    logger.info(f'USER ENTERED >>> {user_input}')
    return user_input
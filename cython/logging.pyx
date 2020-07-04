import logging

class Logger:

    def log(self, msg, level):
        if level == "INFO":
            logging.info(msg)
        elif level == "WARNING":
            logging.warning(msg)
        elif level == "ERROR":
            logging.error(msg)
        elif level == "CRITICAL":
            logging.critical(msg)

import logging


def singleton(cls):
    instances = {}

    def _singleton(*args,**kwargs):
        if cls not in instances:
            instances[cls] = cls(*args,**kwargs)
        return instances[cls]
    return _singleton


@singleton
class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'critical':logging.CRITICAL
    }

    def __init__(self,level='info',when='D',backCount=3,fmt='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'):
        self.logger = logging.getLogger()
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))

        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setFormatter(format_str)
            self.logger.addHandler(sh)

    def info(self,msg):
        self.logger.info(msg=msg)

    def error(self,msg):
        self.logger.error(msg=msg)

    def warning(self,msg):
        self.logger.warning(msg=msg)

    def exception(self,msg,exc_info=True):
        self.logger.exception(msg=msg,exc_info=exc_info)
import logging
import json
from datetime import datetime, timezone  
from google.colab import userdata
import os


token = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = token
os.environ['HF_TOKEN'] = token

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName
        }        
        
        if hasattr(record, 'extra_data'):
            log_obj.update(record.extra_data)
        return json.dumps(log_obj)
  

def setup_logging():
    logger = logging.getLogger('langgraph_app')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger

if __name__ == '__main__':
    logger = setup_logging()
    logger.info("Logging setup complete", extra={'extra_data': {'app': 'langgraph'}})

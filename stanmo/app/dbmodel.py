import os
import logging

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from .config import STANMO_CONFIG_DICT

# http://docs.sqlalchemy.org/en/rel_1_0/orm/tutorial.html#adding-new-objects
SABase = declarative_base()
# Model prediction history (model id, record id, prediction result, feedback result, prediction score, predict date, feedback date)
class PredictionHistory(SABase):
    __tablename__ = 'prediction_history'
    model_name = Column(String, primary_key=True)
    record_id = Column(String, primary_key=True)
    prediction_result = Column(String)
    prediction_date = Column(DateTime, default=func.now())
    prediction_score = Column(Integer) # 0~100.
    feedback_result = Column(String)
    feedback_date = Column(DateTime)
    def __unicode__(self):
        return self.name

database_path = os.path.join(STANMO_CONFIG_DICT['stanmo']['stanmo_home'], 'stanmo_sqlite.db')
engine = create_engine('sqlite:///'+ database_path)  #, echo=True
if not os.path.exists(database_path):
    # SABase.metadata.drop_all(engine)
    SABase.metadata.create_all(engine)
    logging.getLogger('stanmo_logger').info('Database initialization finished.') # will not print anything

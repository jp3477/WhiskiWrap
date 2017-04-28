from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship


class Session(Base):
    __tablename__ = 'session'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime)
    plot_filepath = Column(String)
    mouse_id = Column(Integer, ForeignKey('mouse.id'))
    mouse = relationship('Mouse', back_populates='sessions')

class Mouse(Base):
    __tablename__ = 'mouse'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    sessions = relationship('Session', back_populates='mouse')



# Base.metadata.create_all(engine)
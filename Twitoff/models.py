""" SQLAlchemy models and utility functions for Twitoff Application """

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Creates a 'user' table
class User(db.Model):
    """Twitter User Table that will correspond to tweets - SQLAlchemy syntax""" 
    id = db.Column(db.Integer, primary_key=True) # id as primary key
    name = db.Column(db.String(50), nullable=False) # user name

    def __repr__(self):
        return '<User: {}>'.format(self.name)

# Creates a 'tweet' table
class Tweet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    vect = db.Column(db.PickleType, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('tweets', lazy=True))

    def __repr__(self):
        return '<Tweet: {}>'.format(self.text)
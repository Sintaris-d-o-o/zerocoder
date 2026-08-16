from app import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    vk_api_id = db.Column(db.String(250), nullable=True)
    vk_group_id = db.Column(db.String(20), nullable=True)

    def __repr__(self):
        return f"User('{self.username}')"

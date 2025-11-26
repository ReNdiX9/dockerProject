from flask_login import UserMixin
from datetime import datetime
import json
import os

class User(UserMixin):
    """User model for authentication"""
    
    def __init__(self, id, email, name, oauth_provider=None, two_factor_secret=None):
        self.id = id
        self.email = email
        self.name = name
        self.oauth_provider = oauth_provider
        self.two_factor_secret = two_factor_secret
        self.created_at = datetime.utcnow()
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'oauth_provider': self.oauth_provider,
            'two_factor_secret': self.two_factor_secret,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        }
    
    @staticmethod
    def from_dict(data):
        user = User(
            id=data['id'],
            email=data['email'],
            name=data['name'],
            oauth_provider=data.get('oauth_provider'),
            two_factor_secret=data.get('two_factor_secret')
        )
        if 'created_at' in data:
            user.created_at = datetime.fromisoformat(data['created_at']) if isinstance(data['created_at'], str) else data['created_at']
        return user


class UserStore:
    """Simple file-based user storage (replace with real database in production)"""
    
    def __init__(self, filepath='users.json'):
        self.filepath = filepath
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump({}, f)
    
    def _read_users(self):
        with open(self.filepath, 'r') as f:
            data = json.load(f)
            return {k: User.from_dict(v) for k, v in data.items()}
    
    def _write_users(self, users):
        with open(self.filepath, 'w') as f:
            data = {k: v.to_dict() for k, v in users.items()}
            json.dump(data, f, indent=2)
    
    def get_user(self, user_id):
        users = self._read_users()
        return users.get(str(user_id))
    
    def get_user_by_email(self, email):
        users = self._read_users()
        for user in users.values():
            if user.email == email:
                return user
        return None
    
    def create_user(self, email, name, oauth_provider=None):
        users = self._read_users()
        user_id = str(len(users) + 1)
        user = User(user_id, email, name, oauth_provider)
        users[user_id] = user
        self._write_users(users)
        return user
    
    def update_user(self, user):
        users = self._read_users()
        users[str(user.id)] = user
        self._write_users(users)
        return user
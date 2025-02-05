import sqlite3
import hashlib
import os
from typing import Optional, Dict, Any, List
from datetime import datetime

class Database:
    def __init__(self, db_file: str = "legal_assistant.db"):
        self.db_file = db_file
        self.init_db()
    
    def get_db_connection(self):
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        conn = self.get_db_connection()
        c = conn.cursor()
        
        # Create users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                phone TEXT,
                location TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Create user_cases table for tracking user's legal cases
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                case_type TEXT NOT NULL,
                case_description TEXT,
                court TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create chat_messages table for storing user chat history
        c.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash a password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username: str, email: str, password: str, 
                     full_name: str = None, phone: str = None, location: str = None) -> bool:
        """Register a new user"""
        try:
            conn = self.get_db_connection()
            c = conn.cursor()
            
            # Check if username or email already exists
            c.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
            if c.fetchone() is not None:
                return False
            
            password_hash = self.hash_password(password)
            c.execute('''
                INSERT INTO users (username, email, password_hash, full_name, phone, location)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (username, email, password_hash, full_name, phone, location))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error registering user: {e}")
            return False
        finally:
            conn.close()
    
    def login_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Login a user and return user data if successful"""
        try:
            conn = self.get_db_connection()
            c = conn.cursor()
            
            password_hash = self.hash_password(password)
            c.execute('''
                SELECT id, username, email, full_name, phone, location
                FROM users 
                WHERE username = ? AND password_hash = ?
            ''', (username, password_hash))
            
            user = c.fetchone()
            if user:
                # Update last login time
                c.execute('''
                    UPDATE users 
                    SET last_login = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (user['id'],))
                conn.commit()
                
                return dict(user)
            return None
        except Exception as e:
            print(f"Error logging in user: {e}")
            return None
        finally:
            conn.close()
    
    def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user profile data"""
        try:
            conn = self.get_db_connection()
            c = conn.cursor()
            
            c.execute('''
                SELECT id, username, email, full_name, phone, location, created_at, last_login
                FROM users 
                WHERE id = ?
            ''', (user_id,))
            
            user = c.fetchone()
            if user:
                return dict(user)
            return None
        except Exception as e:
            print(f"Error getting user profile: {e}")
            return None
        finally:
            conn.close()
    
    def update_user_profile(self, user_id: int, updates: Dict[str, Any]) -> bool:
        """Update user profile data"""
        try:
            conn = self.get_db_connection()
            c = conn.cursor()
            
            # Build update query dynamically based on provided fields
            update_fields = []
            values = []
            for key, value in updates.items():
                if key in ['full_name', 'phone', 'location', 'email']:
                    update_fields.append(f"{key} = ?")
                    values.append(value)
            
            if not update_fields:
                return False
            
            values.append(user_id)
            query = f'''
                UPDATE users 
                SET {', '.join(update_fields)}
                WHERE id = ?
            '''
            
            c.execute(query, values)
            conn.commit()
            return True
        except Exception as e:
            print(f"Error updating user profile: {e}")
            return False
        finally:
            conn.close()
    
    def save_chat_message(self, user_id: int, role: str, content: str) -> bool:
        """Save a chat message for a user"""
        try:
            conn = self.get_db_connection()
            c = conn.cursor()
            
            c.execute('''
                INSERT INTO chat_messages (user_id, role, content)
                VALUES (?, ?, ?)
            ''', (user_id, role, content))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving chat message: {e}")
            return False
        finally:
            conn.close()
    
    def get_user_chat_history(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history for a user"""
        try:
            conn = self.get_db_connection()
            c = conn.cursor()
            
            c.execute('''
                SELECT role, content, timestamp
                FROM chat_messages
                WHERE user_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
            ''', (user_id, limit))
            
            messages = []
            for row in c.fetchall():
                messages.append({
                    "role": row['role'],
                    "content": row['content'],
                    "timestamp": row['timestamp']
                })
            
            return messages
        except Exception as e:
            print(f"Error getting chat history: {e}")
            return []
        finally:
            conn.close()
    
    def clear_user_chat_history(self, user_id: int) -> bool:
        """Clear all chat messages for a user"""
        try:
            conn = self.get_db_connection()
            c = conn.cursor()
            
            c.execute('DELETE FROM chat_messages WHERE user_id = ?', (user_id,))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error clearing chat history: {e}")
            return False
        finally:
            conn.close()

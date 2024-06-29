import sqlite3
import PIL.Image as PIL
import numpy as np
import io
from Network.globalVariables import *

class database():
    def __init__(self):
        self.db = 'knownfaces.db'
        self.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, profile_picture BLOB, feature_vector BLOB)')

    def add_user(self, name : str, pic : PIL.Image, feature_vector : np.ndarray):
        if self.user_exists(name):
            print(f"attempt to add existing user {name}")
            return
        with io.BytesIO() as buffer:
            pic.save(buffer, format="jpg")
            image_blob = buffer.getvalue()
        vector_blob = feature_vector.tobytes()
        self.execute("INSERT INTO users (name, profile_picture, feature_vector) VALUES (?, ?, ?)", name, image_blob, vector_blob)
    
    def get_image(self, name : str):
        results = self.execute("SELECT profile_picture FROM users WHERE name = ?", name)
        if results == []:
            raise Exception("No such user in database")
        return PIL.open(io.BytesIO(results[0][0]))
    
    def find_match(self, vectors : list[np.ndarray], tolerance : float=t):
        users = self.execute('SELECT id, feature_vector FROM users')
        min_distance = None
        min_distance_id = None
        for id, fv in users:
            sum_distances = 0
            for v in vectors:
                sum_distances += np.linalg.norm(fv-v)
            avg_distance = sum_distances/len(vectors)
            if min_distance is None or avg_distance < min_distance:
                min_distance = avg_distance
                min_distance_id = id
        min_user = self.execute("SELECT name FROM users WHERE id = ?", min_distance_id)[0][0]
        return min_user if min_distance_id < tolerance else None


    def user_exists(self, name):
        # Check if a user with the given name already exists
        count = self.execute('''
        SELECT COUNT(*) FROM users WHERE name = ?
        ''', name)[0][0]
        return count > 0

    def execute(self, command, *args):
        connection = sqlite3.connect(self.db)
        cursor = connection.cursor()
        cursor.execute(command, args)

        results = None
        if "SELECT" in command: results = cursor.fetchall()

        connection.commit()
        connection.close()

        return results
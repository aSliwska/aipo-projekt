import pickle
import sqlite3

# CODE FROM https://stackoverflow.com/questions/28397847/most-straightforward-way-to-cache-geocoding-data
class Cache(object):
    def __init__(self):
       self.conn = conn = sqlite3.connect('osm_cache.db')
       cur = conn.cursor()
       cur.execute('CREATE TABLE IF NOT EXISTS '
                   'Geo ( '
                   'query STRING PRIMARY KEY, '
                   'location BLOB '
                   ')')
       conn.commit()

    def query_cached(self, query):
        cur = self.conn.cursor()
        cur.execute('SELECT location FROM Geo WHERE query=?', (query,))
        res = cur.fetchone()
        if res is None: return False
        return pickle.loads(res[0])

    def save_to_cache(self, query, location):
        cur = self.conn.cursor()
        cur.execute('INSERT INTO Geo(query, location) VALUES(?, ?)',
                    (query, sqlite3.Binary(pickle.dumps(location, -1))))
        self.conn.commit()
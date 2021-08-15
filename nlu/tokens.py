import sqlite3

conn = sqlite3.connect("tokens.db", check_same_thread=False)
c = conn.cursor()


class Tokens:
    def __init__(self):
        self.true = None
        self.addr = None
        self.token = None
        self.searchable = ""
        self.create_table(True)

    def create_table(self, true):
        self.true = true
        if self.true:
            c.execute("CREATE TABLE IF NOT EXISTS tokens(addr TEXT, token TEXT)")
            conn.commit()
        else:
            pass

    def data_entry(self, addr, token):
        self.question = addr
        self.token = token
        c.execute("INSERT INTO tokens (addr, token) VALUES (?, ?)", (self.addr, self.token))
        conn.commit()

    def verify_token(self, addr):
        self.addr = str(addr)
        c.execute("SELECT * FROM tokens WHERE addr=(?)", (self.addr,))
        for row in c.fetchall():
            if self.addr == row[0]:
                return row[1]
            else:
                return None

    def read_all(self):
        c.execute("SELECT * FROM tokens")
        for row in c.fetchall():
            print(row)


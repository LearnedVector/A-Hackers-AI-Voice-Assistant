import sqlite3

conn = sqlite3.connect("VoiceAssistant/nlu_stable/etc/memory.db")
c = conn.cursor()


class MemoryUnit:
    def __init__(self):
        self.true = None
        self.question = None
        self.answer = None
        self.searchable = None
        self.create_table(True)

    def create_table(self, true):
        self.true = true
        if self.true:
            c.execute("CREATE TABLE IF NOT EXISTS memoryUnit(question TEXT, answer TEXT)")
            conn.commit()
        else:
            pass

    def data_entry(self, question, answer):
        self.question = question
        self.answer = answer
        c.execute("INSERT INTO memoryUnit (question, answer) VALUES (?, ?)", (self.question, self.answer))
        conn.commit()

    def read_from_db(self, searchable):
        self.searchable = searchable
        c.execute("SELECT * FROM memoryUnit")
        for row in c.fetchall():
            if self.searchable in row[0]:
                return row[1]
            else:
                return None

    def read_all(self):
        c.execute("SELECT * FROM memoryUnit")
        for row in c.fetchall():
            print(row)

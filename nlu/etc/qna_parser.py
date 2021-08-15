class Parser:
    def __init__(self):
        self.answer = None
        self.aux_verbs = ["is ", "are ", "was ", "were ", "has ", "have ", "had ", "can ", "could ", "should" , 
        "would ", "shall ", "will ", "might ", "may "]
        self.question_words = ["who", "where", "which", "whom", "how", "what"]
        self.question = None
        self.second_person = {"i ": "you ", "me ": "you ", "my ": "your ", "mine ": "yours ", "myself ": "yourself "}
        self.first_person = ['i ', 'me ', 'my ', 'mine ', 'myself ']

    def parse_question(self, question):
        """
        :param question:
        :return:
        """
        self.question = str(question).lower()
        for i in self.question_words:
            if i in self.question:
                for j in self.aux_verbs:
                    if (i + " " + j) in self.question:
                        return self.question[self.question.index(i + " " + j)-1 + len(i + " " + j + " "):]
                    else:
                        return self.question[self.question.index(i + " ")-1 + len(i + " "):]

    def parse_answer(self, answer):
        self.answer = str(answer).lower()

        for i in self.first_person:
            if i in self.answer:
                return self.answer.replace(i, self.second_person[i])


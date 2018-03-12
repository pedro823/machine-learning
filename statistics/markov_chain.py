import random as r
random = r.SystemRandom()

class SimpleMarkovChain:
    """
        Simple Markov chain with words.
        Only previews next word based on last word seen.
        Created by:
        ~razgrizone (Pedro Pereira)
    """

    class MarkovChainException(Exception):
        """ Common class for errors thrown by MarkovChain. """
        pass

    def __init__(self, text):
        self.chain = dict()
        self.splitted_text = text.split(' ')

        if '_total' in self.splitted_text:
            raise MarkovChainException('\'_total\' is a reserved word. Please'
                                       'do not include it in the text.')

        for idx, word in enumerate(self.splitted_text):
            if self.chain.get(word) is None:
                self.chain[word] = dict()
                self.chain[word]['_total'] = 1
            else:
                # is is the last word?
                self.chain[word]['_total'] += 1

            if idx < len(self.splitted_text) - 1:
                # sees next word and updates
                next_word = self.splitted_text[idx + 1]
                if self.chain[word].get(next_word) is None:
                    self.chain[word][next_word] = 1
                else:
                    self.chain[word][next_word] += 1

    def statistics(self):
        statistics = {
            'chain': self.chain
        }
        return statistics

    def build_chain(self, length, starting_word=None, until_period=False):
        """
            Builds a markov sentence.
            arguments:

            length (int)              -> the amount of words the text
                                         should have.
            starting_word=None (str)  -> The word to start the chain.
            until_period=False (bool) -> Goes on after length until the next
                                         period is found.
                                         WARNING: can make algorithm go to
                                         infinite loop
        """
        if type(length) is not int:
            raise MarkovChainException('length is not of type int')

        if type(until_period) is not bool:
            raise MarkovChainException('until_period is not of type bool')

        if starting_word is None:
            # Finds any word in the splitted text
            starting_word = random.choice(self.splitted_text)
        # starting word exists in text?
        elif self.chain.get(starting_word) is None:
                raise MarkovChainException('Starting word doesn\'t'
                                           'exist in text')
        words = list()
        cur_word = starting_word # current word
        for i in range(length):
            cur_word = self.__insert_word_find_next(words, cur_word)

        if until_period:
            while cur_word[-1] != '.':
                cur_word = self.__insert_word_find_next(words, cur_word)
        return ' '.join(words + [cur_word])

    # private

    def __insert_word_find_next(self, words, current_word):
        words.append(current_word)
        total = self.chain[current_word]['_total']
        # finds next word
        random_int = random.randint(0, total)
        j = 0
        for next_word, freq in self.chain[current_word].items():
            # do not include _total
            if next_word == '_total': continue
            if j + freq >= random_int:
                current_word = next_word
                break
            else:
                j += freq

        return  current_word


if __name__ == '__main__':
    from data import text
    import json
    data = ' '.join(text.strip().split('\n'))
    a = SimpleMarkovChain(data)
    print(json.dumps(a.statistics(), indent=4))
    print(a.build_chain(30, until_period=True))

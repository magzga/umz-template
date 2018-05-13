from emails import Email
from classifier import Bayes

Email.load_emails()
Bayes.train(Email.emails_list)

ham_list = list(Bayes.ham_words_dict.items())
ham_list = sorted(ham_list, key = lambda entry : entry[1])

print(ham_list[-10:])

spam_list = list(Bayes.spam_words_dict.items())
spam_list = sorted(spam_list, key = lambda entry : entry[1])

print(spam_list[-10:])
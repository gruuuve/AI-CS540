import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r', encoding='utf8') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
#Needs modifications DONE
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    f = open(filepath, "r", encoding='utf8')
    for word in f: # go through each line and check if in vocab
        word = word.strip()
        if word in vocab:
            if word not in bow.keys(): # need to create new bow key for word
                bow[word] = 1
            else:
                bow[word] += 1
        else:
            if None not in bow.keys(): # None addition and init
                bow[None] = 1
            else:
                bow[None] += 1  
    return bow

#Needs modifications DONE
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}
    total = 0 # used to store total number of files
    for sec in training_data: # iterate through files
        if sec.get("label") in label_list: # count number of iterations matching labels
            total += 1
            if sec.get("label") not in logprob.keys():
                logprob[sec.get("label")] = 2 # 1 for label found, 1 for final calc
            else: 
                logprob[sec.get("label")] += 1
    # once finished through list, do calculations for log prob
    total = total + 2 # add 2 for add-1 smoothing
    for word in logprob.keys(): # add-1 smoothing calculation
        numerator = math.log(logprob[word])
        denominator = math.log(total)
        logprob[word] = numerator - denominator
    return logprob

#Needs modifications DONE
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}
    # TODO: add your code here
    wc = 0 # used to store the word count
    word_prob = {key: 0 for key in vocab}
    for sec in training_data: # iterate through documents matching label
        if sec.get("label") == label:
            bow = sec.get("bow")
            for word in bow: # iterate through each word
                wc += bow.get(word) # increment word count
                if word in vocab:                       
                    if word not in word_prob.keys() and len(word) > 0:
                        word_prob[word] = 0
                        word_prob[word] = bow.get(word)
                    elif len(word) > 0: 
                        word_prob[word] += bow.get(word)
                else: # word not in vocab
                    if None not in word_prob.keys(): # OOV word
                        word_prob[None] = bow.get(word)
                    else:
                        word_prob[None] += bow.get(word)

    # word_prob contains counts of words in documents
    for key in list(word_prob.keys()):
        numerator = int(word_prob[key]) + smooth * 1 # add-1 smoothing calc
        denominator = wc + smooth * (len(vocab) + 1)
        word_prob[key] = math.log(numerator / denominator)
    return word_prob


##################################################################################
#Needs modifications DONE
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    # create objects to add to retval
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    prior_data = prior(training_data, label_list)
    # add to retval dictionary
    retval['vocabulary'] = vocab
    retval['log prior'] = prior_data
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, training_data, '2020')
    retval['log p(w|y=2016)'] = p_word_given_label(vocab, training_data, '2016')
    return retval

#Needs modifications DONE
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    f = open(filepath, 'r', encoding='utf8')
    for line in f:
        line = line.strip()
        if line in model['vocabulary']:
            model['log prior']['2020'] += model['log p(w|y=2020)'][line]
            model['log prior']['2016'] += model['log p(w|y=2016)'][line]
        else:
            model['log prior']['2020'] += model['log p(w|y=2020)'][None]
            model['log prior']['2016'] += model['log p(w|y=2016)'][None]
    retval['log p(y=2020|x)'] = model['log prior']['2020']
    retval['log p(y=2016|x)'] = model['log prior']['2016']
    if model['log prior']['2020'] > model['log prior']['2016']: # chose larger log prior
        retval['predicted y'] = '2020'
    else:
        retval['predicted y'] = '2016'
    
    return retval

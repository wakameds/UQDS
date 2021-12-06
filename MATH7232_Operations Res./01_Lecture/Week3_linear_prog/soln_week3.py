def find_last(char, strng):
    """ Function that returns the index of the last occurrence
        of char in strng

    Parameters:
        char: The character
        strng: The string
        
    Return:
        int: The index
        
    """

    for i,c in enumerate(strng[::-1]):
        if c == char:
            return len(strng) - i - 1
    return -1

def find_non_alpha(strng):
    """ Function that returns the indexes of all the non-alphabetic
        characters in strng

    Parameters:
        strng: The string

    Return:
        tuple: The indexes
        
    """
    
    indexes = ()

    for i,c in enumerate(strng):
        if c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
            indexes += (i,)

    return indexes

def find_phrases(phrase, string, start):
    """ Function that returns a tuple containing the indexes where
        all the occurrences of ‘phrase’ occur within string,
        starting at index, start

    Parameters:
        phrase: The phrase to look for
        string: The string to search in
        start: The starting index

    Return:
        tuple: The tuple of indexes
    """
    
    pos = start
    indexes = ()
    string_length = len(string)
    phrase_length = len(phrase)

    while pos < string_length:
        if phrase == string[pos:pos+phrase_length]:
            indexes += (pos,)
        pos += 1

    return indexes

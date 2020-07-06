import re
from typing import Set, Dict


class ValueUtils:
    '''
    This class contains some static utility methods to process values.
    '''

    MAX_TOKENS = 6
    
    def __init__(self, params):
        '''
        Constructor
        '''
    
    
    @staticmethod
    def split_values(raw: str) -> list:
        """
        Compute the single values provided for an object by a source
        :param raw: string
        :return: list of strings
        """
        #return re.split(';| and |&|with', re.sub('[(\[].*?[)\]]', "", str(raw).lower()))
        #return re.split(';| and |&|with|/', re.sub('[(\[].*?[)\]]', "", str(raw).lower()))  # e: added '/'
        li = re.split(';|\|| and |&|with|/| und ', re.sub('[(\[].*?[)\]]', "", str(raw).lower()))
        
        # Now manage the commas: try to split every discovered value
        values = []
        # For each value
        for v in li:
            val = v.strip()
            if val.endswith(','):
                val = val[0:len(val)-1]
            val = val.replace(' ,', ',')
            # Count the number of commas
            numCommas = val.count(',')
            split = []
            split.append(val)
            if numCommas > 1:
                splitComma = re.split(',', val)
                # At least two words in the second chunk: "Rossi, Mario Verdi, Pino"
                if ValueUtils.countWordsLongerThanX(splitComma[0].strip(), 1) == 1 and ValueUtils.countWordsLongerThanX(splitComma[1].strip(), 1) > 1 and \
                    (len(splitComma) < 4 or ValueUtils.countWordsLongerThanX(splitComma[3].strip(), 1) > 1):
                        split = re.split('(?<!,)\s', val)  # Split on spaces not preceded by comma
                # "Rossi, Mario, Verdi, Pino"
                elif ValueUtils.countWordsLongerThanX(splitComma[0].strip(), 1) == 1:
                    split = ValueUtils.splitOnEvenCommas(val)
                else:  # "Mario Rossi, Pino Verdi": split on commas
                    split = splitComma
            else:  # If there are at least two words after the single comma, we split on the comma
                if numCommas == 1:
                    beforeComma = val[0:val.find(',')]
                    afterComma = val[val.find(',')+1:len(val)]
                    if ValueUtils.countWordsLongerThanX(beforeComma,1)>1 and ValueUtils.countWordsLongerThanX(afterComma,1)>1:
                        split = re.split(',', val)
            values.extend(split)
        
        # Delete empty strings
        valuesNotEmpty = []
        for v in values:
            vs = v.strip()
            if len(vs)>0:
                valuesNotEmpty.append(vs)
        
        return valuesNotEmpty
    
    
    @staticmethod
    def countWordsLongerThanX(s: str, x: int) -> int:
        """
        Count the number of words (substrings separated by space) in the given string having length greater than or equal to a given threshold.
        The given threshold is used starting from the second word; for the first word a threshold 1 is used.
        :param s: string to be checked
        :param x: maximum allowed length
        :return: integer representing the number of substrings of s with length greater than or equal to x
        """
        
        # Remove repeated spaces
        s = re.sub(' {2,}', ' ', s)
        # Split the string the spaces
        spl = s.split()
        # Return the number of words longer than x
        num = 0
        i = 0
        for word in spl:
            w = word.replace('.', '') # Not counting the dots
            w = re.sub('[^a-zA-Z]+', '', w)
            if i==0:
                if len(w)>=1:
                    num += 1
            else:
                if len(w)>x:
                    num += 1
            i += 1
        return num

        
    # Split on even commas (first is 1)
    @staticmethod
    def splitOnEvenCommas(s: str) -> list:
        """
        Splits the given string on the even commas (considering the first as 1)
        :param s: string to be split.
        :return: list of substrings resulting from the split on even commas
        """
        splitList = []
        sb = ''
        numCommas = 0
        for i in range(len(s)):
            if s[i] == ',':
                numCommas += 1
                if numCommas % 2 == 0:
                    splitList.append(sb)
                    sb = ''
                else:
                    sb = sb + ','
            else:
                sb = sb + s[i]
        if len(sb) > 0:
            splitList.append(sb)
        return splitList
    

    @staticmethod
    def clean_value(raw: str) -> str:
        """
        Clean object value from dirty characters
        :param raw: string to clean
        :return: cleaned string
        """
        # s = re.sub('[(\[].*?[)\]]', "", raw)
        t = raw.replace(',', ' ')  # e: replace comma with space
        # e: Remove numbers (1111 - 1111, 1111-1111, 1111) and ordinals
        t = re.sub('[0-9]*1st','',t)
        t = re.sub('[0-9]*2nd','',t)
        t = re.sub('[0-9]*3rd','',t)
        t = re.sub('[0-9]*[0,4-9]th','',t)
        t = re.sub('[0-9]+ \- [0-9]*', '', t)
        t = re.sub('[0-9]+\-[0-9]*', '', t)
        t = re.sub('[0-9]+', '', t)
        t = re.sub(' +', ' ', t) # e: remove multiple spaces
        t = t.lower().replace('hrsg.', '').replace('ed.', '').replace('(ill)','').replace(' ill.','')   \
            .replace(',', '').replace(';', '').replace('.', ' ').replace(':', '').replace(')', '').replace('(' ,'').replace('[','').replace(']','') \
            .replace('contributing authors', '').replace('illustrator', '').replace('illustrated', '').replace('illustrations', '') \
            .replace('translated', '').replace('translator', '').replace(' by ', ' ').replace('translation', '') \
            .replace('edited', '').replace('editors', '').replace('editor', '').replace('collaborator', '').replace('introduction', '') \
            .replace('copyright', '').replace('paperback', '').replace('collection','').replace('general','').replace('project','').replace('director','') \
            .replace('et al', '').replace('compiler', '').replace('illust','').replace(' ill','') \
            .replace('a conversation with','').replace('actors', '').replace('actresses','').replace('director','') \
            .replace('actor', '').replace('actress','').replace('\\n','').replace('\\t','')
        
        t = re.sub(' +', ' ', t) # e: again multiple spaces
        t = t.strip().split(' ')
        s = set()    
                
        for st in t:   #e: added
            k = st
            if k.startswith("-"):
                k = k[1:]
            
            # The token must end with a letter
            while len(k)>0 and not(k[len(k)-1].isalpha()):
                k = k[0:len(k)-1]
            if len(k)>0 and not (k=='none'):  # Remove also the values 'none'
                s.add(k)
  
        #return str(sorted(s))
        s = ValueUtils.retain_only_values_with_alphabet(s)  # e: remove the tokens not containing any letter
        return ' '.join(sorted(s))
    
    
    # e: added
    @staticmethod
    def retain_only_values_with_alphabet(vals: Set[str]) -> Set[str]:
        """
        Given a set of strings, returns another set containing only the strings from the first one with at least one alphabetic character.
        :param vals: The set of strings.
        :rtype: set
        """
        c2 = set()
        for s in vals:
            if re.search('[a-zA-Z]', s):
                c2.add(s)
        return c2 
    
    # e: added
    @staticmethod
    def retain_only_short_known_values(vals: Set[str]) -> Set[str]:
        """
        Given a set of strings, returns another set containing only the strings from the first one that contains no more than the number of token specified by a threshold.
        :param vals: The set of strings.
        :rtype: set
        """
        c2 = set()
        for s in vals:
            if s.lower()!='unknown' and s.lower()!='various':
                tokens = re.split(' ', s)
                if len(tokens)<=ValueUtils.MAX_TOKENS:
                    c2.add(s)
        return c2
    
    
    @staticmethod
    def split_values_movies(raw: str) -> list:
        """
        Compute the single values provided for an object by a source (movie dataset)
        :param raw: string
        :return: list of strings
        """
        
        # First delete all that is after '</div'
        pos = raw.find('</div')
        if pos>=0:
            raw = raw[0:pos]
        
        # Now delete all that is before the last '>'
        pos = raw.rfind('>')
        if pos>=0:
            raw = raw[pos+1:len(raw)]
        
        li = re.split(';|,', re.sub('[(\[].*?[)\]]', "", str(raw).lower()))
        
        # Now manage and/&: we must consider the situations {"Mario and Luca Rossi", "Rossi Luca and Mario"}
        values = []
        # For each value
        for v in li:
            val = v.strip()
            # Count the number of and/&
            numAnds = val.count(' and ')
            numAnds += val.count(' & ')
            
            li2 = re.split(' and | & ', val)
            if numAnds==1:
                tokens0 = re.split(' ', li2[0])
                tokens1 = re.split(' ', li2[1])
                if len(tokens0)==1 and len(tokens1)==2:  # "Mario and Luca Rossi"
                    li2[0] = li2[0] + ' ' + tokens1[1]
                elif len(tokens0)==2 and len(tokens1)==1:  # "Rossi Luca and Mario"
                    li2[1] = tokens0[0] + ' ' + li2[1] 
            values.extend(li2)                  
        
        # Delete empty strings
        valuesNotEmpty = set()
        for v in values:
            vs = v.strip()
            if len(vs)>0 and vs!='\\n' and vs!='\\t':
                valuesNotEmpty.add(vs)
        
        return list(valuesNotEmpty)
"""
    Generate classic POSTagger
    
    @see config.py
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import regex


class PreProcessText ():
    """
    PreProcessText
    """
    
    msg_language = {
        'x': 'por',
        'd': 'de',
        'gob': 'gobierno',
        '€+': 'euros'
    }
    
    acronyms = {
        'TVE': 'Televisión Española',
        'EEUU': 'Estados Unidos',
        'UE': 'Unión Europa',
        'PAC': 'Política Agraria Común',
        'CCAA': 'Comunidades Autónomas',
        'ERTE': 'Expediente de Regulación de Empleo',
        'MEFP': 'Ministerio de Educación y Formación Profesional',
        'BCE': 'Banco Central Europeo',
        'FMI': 'Fondo Monetario Internacional',
        'OCDE': 'Organización para la Cooperación y el Desarrollo Económicos',
        'MIR': 'Médico Interno Residente',
        'ETA': 'Euskadi Ta Askatasuna',
        'CSIC': 'Consejo Superior de Investigaciones Científicas',
        'LGTB': 'Lesbianas, Gais,​ Bisexuales y Transgénero',
        'LGTB[\w\+]?': 'Lesbianas, Gais,​ Bisexuales y Transgénero',
        'IVA': 'Impuesto al valor agregado',
        'CE': 'Constitución Española',
        'CM': 'Congreso de Ministros',
        'CyL': 'Castilla y León',
        'CAM': 'Comunidad Autónoma de Madrid',
        'G\. Mixto': 'Grupo Mixto',
        'PNV': 'Partido Nacionalista Vasco',
        'PSOE': 'Partido Socialista Obrero Español',
        'PP': 'Partido Popular',
        'UP': 'Unidas Podemos',
    }
    
    
    # Patterns
    whitespace_pattern = r'\s+'
    quotations_pattern = r'["“”\'«»]'
    elogation_pattern = r'(.)\1{2,}'
    
    orphan_dots_pattern = r'[\.\s]{2,}'
    orphan_commas_pattern = r'[\,\s]{2,}'
    dashes_pattern = r' [\-\—] '
    orphan_exclamatory_or_interrogative_pattern = r'\s+([\?\!]) '
    
    url_pattern = regex.compile (r'https?://\S+')
    hashtag_pattern = regex.compile (r'#([\p{L}0-9\_]+)')
    begining_mentions_pattern = regex.compile (r"^(@[A-Za-z0-9\_]+\s?)+")
    middle_mentions_pattern = regex.compile (r'(?<!\b)@([A-Za-z0-9\_]+)\b(?<!user)')
    laughs_pattern = regex.compile (r'/\bj[ja]+a?\b')
    digits_pattern = regex.compile (r"\b\d+[\.,]?\d*\b")
    emoji_pattern = regex.compile ("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags = regex.UNICODE)
                               

    def camel_case_split (identifier):
        """
        camel_case_split
        
        @param identifier String
        
        @link https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python/29920015
        """
        matches = regex.finditer ('.+?(?:(?<=\p{Ll})(?=\p{Lu})|(?<=\p{Lu})(?=\p{Lu}\p{Ll})|$)', identifier)
        return ' '.join ([m.group (0) for m in matches])


    def remove_urls (sentences):
        return sentences.replace (to_replace = self.url_pattern, value = '', regex = True)
    
    def remove_mentions (sentences, placeholder = "@USER"):
        sentences = sentences.replace (to_replace = self.begining_mentions_pattern, value = '', regex = True)
        sentences = sentences.apply (lambda x: regex.sub (self.middle_mentions_pattern, placeholder, x))
        sentences = sentences.apply (lambda x: regex.sub (r'(@USER ){2,}', placeholder + " ", x))

        return sentences

    def expand_hashtags (sentences):
        """
        Expand hashtags into meaningful words
        """
        return sentences.apply (lambda x: regex.sub (self.hashtag_pattern, lambda match: ' ' + self.camel_case_split (match.group (1)), x))

    def remove_digits (sentences):
        """
        remove_digits
        """
        return sentences.replace (to_replace = self.digits_pattern, value = '', regex = True)
        
    def remove_whitespaces (sentences):
        """
        remove_whitespaces
        """
        sentences = sentences.replace (to_replace = self.whitespace_pattern, value=' ', regex=True)    
        sentences = sentences.replace (to_replace = self.orphan_dots_pattern, value='. ', regex=True)
        sentences = sentences.replace (to_replace = self.orphan_commas_pattern, value=', ', regex=True)
        sentences = sentences.replace (to_replace = self.dashes_pattern, value='. ', regex=True)
        sentences = sentences.replace (to_replace = self.orphan_exclamatory_or_interrogative_pattern, value='\1 ', regex=True)
        
        return sentences
        

    def remove_emojis (sentences):
        """
        remove_emojis
        """
        return sentences.apply (lambda x: regex.sub (self.emoji_pattern, '', x))
    
    
    def remove_quotations (sentences):
        """
        remove_quotations
        """
        sentences = sentences.replace (to_replace = self.quotations_pattern, value = '', regex = True)
        
        return sentences


    def remove_elongations (sentences):
        """
        remove_elongations
        """
        
        # Remove exclamatory and interrogative
        for character in ['!', '¡', '?', '¿']:
            pattern = regex.compile ('\\' + character + '{2,}')
            sentences = sentences.apply (lambda x: regex.sub (pattern, character, x))
        
        # Remove letters longer than 2
        sentences = sentences.apply (lambda x: regex.sub (self.elogation_pattern', '\1', x))
        sentences = sentences.apply (lambda x: regex.sub (self.laughs_pattern, 'jajaja', x))
        
        return sentences

    def expand_acronyms (sentences, acronyms):
        """
        expand_acronyms
        
        @param sentences
        @param acronyms
        """
        
        for key, value in acronyms.items ():
            pattern = regex.compile (r'(?i)\b' + key + r'\b')
            sentences = sentences.replace (to_replace = pattern, value = value, regex = True)
            
        return sentences
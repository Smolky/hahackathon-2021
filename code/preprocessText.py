"""
    Generate classic POSTagger
    
    @see config.py
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import regex
import string


class PreProcessText ():
    """
    PreProcessText
    """
    
    msg_language = {
        '€+': 'euros',
        'x': 'por',
        'd': 'de',
        'gob': 'gobierno',
    }
    
    
    english_contractions = {
        "won\'t": "will not",
        "can\'t": "can not",
        "n\'t": " not",
        "\'re": " are",
        "\'s": " is",
        "\'d": " would",
        "\'ll": " will",
        "\'t": " not",
        "\'ve": " have",
        "\'m": " am"
    }
    
    acronyms = {
        'TVE': 'Televisión Española',
        'OMS': 'Organización Mundial de la Salud',
        'EEUU': 'Estados Unidos',
        'SM': 'Su Majestad',
        'GAL': 'Grupos Antiterroristas de Liberación',
        'UE': 'Unión Europa',
        'PAC': 'Política Agraria Común',
        'CCAA': 'Comunidades Autónomas',
        'ERTE': 'Expediente de Regulación de Empleo',
        'ERTES': 'Expedientes de Regulación de Empleo',
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
        'CLM': 'Castilla La Mancha',
        'CyL': 'Castilla y León',
        'CAM': 'Comunidad de Madrid',
        'BCN': 'Barcelona',
        'MWC': 'Mobile World Congress',
        'G. Mixto': 'Grupo Mixto',
        'PGE': 'Presupuestos Generales del Estado',
        'PNV': 'Partido Nacionalista Vasco',
        'PP': 'Partido Popular',
        'PSOE': 'Partido Socialista Obrero Español',
        'UP': 'Unidas Podemos',
    }
    
    # Patterns
    whitespace_pattern = r'\s+'
    quotations_pattern = r'["“”\'«»]'
    elogation_pattern = r'(.)\1{2,}'
    
    gender_contraction_pattern = r'(?i)\b(\p{L}+)@s\b'
    
    orphan_dots_pattern = r'[\.\s]{2,}'
    dashes_pattern = r' [\-\—] '
    orphan_exclamatory_or_interrogative_pattern = r'\s+([\?\!]) '
    
    url_pattern = regex.compile (r'https?://\S+')
    hashtag_pattern = regex.compile (r'#([\p{L}0-9\_]+)')
    begining_mentions_pattern = regex.compile (r"^(@[A-Za-z0-9\_]+\s?)+")
    middle_mentions_pattern = regex.compile (r'(?<!\b)@([A-Za-z0-9\_]+)\b(?<!user)')
    laughs_pattern = regex.compile (r'(?i)/\bj[ja]+a?\b')
    digits_pattern = regex.compile (r"\b\d+[\.,]?\d*\b")
    emoji_pattern = regex.compile (r'[^\x00-\x7F]+')
    

    def camel_case_split (self, identifier):
        """
        camel_case_split
        
        @param identifier String
        
        @link https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python/29920015
        """
        matches = regex.finditer ('.+?(?:(?<=\p{Ll})(?=\p{Lu})|(?<=\p{Lu})(?=\p{Lu}\p{Ll})|$)', identifier)
        return ' '.join ([m.group (0) for m in matches])

    def remove_urls (self, sentences):
        """
        remove_urls
        
        @param sentences
        """
        return sentences.apply (lambda x: regex.sub (self.url_pattern, '', x))
    
    def remove_mentions (self, sentences, placeholder = "@USER"):
        sentences = sentences.apply (lambda x: regex.sub (self.begining_mentions_pattern, '', x))
        sentences = sentences.apply (lambda x: regex.sub (self.middle_mentions_pattern, placeholder, x))
        sentences = sentences.apply (lambda x: regex.sub (r'(@USER ){2,}', placeholder + " ", x))

        return sentences

    def expand_hashtags (self, sentences):
        """
        Expand hashtags into meaningful words
        """
        return sentences.apply (lambda x: regex.sub (self.hashtag_pattern, lambda match: ' ' + self.camel_case_split (match.group (1)), x))

    def remove_digits (self, sentences):
        """
        remove_digits
        """
        return sentences.apply (lambda x: regex.sub (self.digits_pattern, '', x))
    
    
    def remove_whitespaces (self, sentences):
        """
        remove_whitespaces
        """
        sentences = sentences.replace (to_replace = self.whitespace_pattern, value=' ', regex=True)    
        sentences = sentences.replace (to_replace = self.orphan_dots_pattern, value='. ', regex=True)
        sentences = sentences.replace (to_replace = self.dashes_pattern, value='. ', regex=True)
        sentences = sentences.replace (to_replace = self.orphan_exclamatory_or_interrogative_pattern, value='\1 ', regex=True)
        
        return sentences
        

    def remove_emojis (self, sentences):
        """
        remove_emojis
        """
        return sentences.apply (lambda x: regex.sub (self.emoji_pattern, '', x))
    
    
    def remove_quotations (self, sentences):
        """
        remove_quotations
        """
        sentences = sentences.replace (to_replace = self.quotations_pattern, value = '', regex = True)
        
        return sentences


    def remove_elongations (self, sentences):
        """
        remove_elongations
        """
        
        # Remove exclamatory and interrogative
        for character in ['!', '¡', '?', '¿']:
            pattern = regex.compile ('\\' + character + '{2,}')
            sentences = sentences.apply (lambda x: regex.sub (pattern, character, x))
        
        # Remove letters longer than 2
        sentences = sentences.apply (lambda x: regex.sub (self.elogation_pattern, '\1', x))
        sentences = sentences.apply (lambda x: regex.sub (self.laughs_pattern, 'jajaja', x))
        
        return sentences
        
    def to_lower (self, sentences):
        """
        to_lower
        
        @param sentences
        """
        return sentences.str.lower ()
    
    
    def remove_punctuation (self, sentences):
        """
        remove_punctuation
        
        @param sentences
        """
        return sentences.apply (lambda x: x.translate (str.maketrans ('', '', string.punctuation)))
    

    def expand_gender (self, sentences):
        """
        expand_gender
        
        Example: amig@s: amigos y amigas
                 nosotr@s: nosotros y nosotras
        """
        return sentences.replace (to_replace = self.gender_contraction_pattern, value = '\1os y \1as', regex = True)

    def expand_acronyms (self, sentences, acronyms):
        """
        expand_acronyms
        """
    
        for key, value in acronyms.items ():
            pattern = regex.compile (r"(?i)\b" + key + r"\b")
            sentences = sentences.apply (lambda x: regex.sub (pattern, value, x))
            
        return sentences
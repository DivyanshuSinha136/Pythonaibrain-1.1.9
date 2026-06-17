from .main import TTS
from .main import speak, TTSConfig

__all__ = ['TTS', 'speak']
__version__ = '1.0.9'
__author__ = "Divyanshu Sinha"
__doc__ = '''
'windows' ->
        'david',     # Microsoft David (Male, en-US)
        'zira',      # Microsoft Zira (Female, en-US)
        'mark'       # Microsoft Mark (Male, en-US)

'macos' ->
        'alex',      # Alex (Male, en-US)
        'samantha',  # Samantha (Female, en-US)
        'victoria',  # Victoria (Female, en-US)
        'fred',      # Fred (Male, robotic style)
        'daniel',    # Daniel (Male, en-GB)
        'fiona'      # Fiona (Female, en-GB)

'linux' ->
        'english',      # Default espeak voice
        'english-us',   # American English
        'english-uk',   # British English
        'mb-en1',       # MBROLA English 1
        'mb-fr1'        # MBROLA French 1
'''

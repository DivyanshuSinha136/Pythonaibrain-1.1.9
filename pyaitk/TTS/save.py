import pyttsx3

class Text:
    def __init__(self, text: str = 'Hello World', out_file_name: str = 'output', voice: str = 'Male') -> None:
        idx = 0
        if voice.lower() == 'male':
            idx = 0

        elif voice.lower() == 'female':
            idx = 1

        else:
            print(f'Invalid voice {voice}\nChose Either Male of Female')
            idx = 0

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)

        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', self.voices[1].id)

        self.engine.say(text)
        self.engine.save_to_file(text, f'{out_file_name.split(".")[0]}.wav')
        self.engine.runAndWait()

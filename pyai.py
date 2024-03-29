import numpy as np
import whisper


class Basic:
    def KNN(x, y, returnValues = 0):
        distances = []
        for axisX, axisY in zip(x, y):
            distance = axisX - axisY
            absDistance = np.absolute(distance)
            distances.append(absDistance)
            
        sortedDistances = []
        checkDistance = min(distances, key = lambda x:np.absolute(x-i)) 
        sortedDistances.append(checkDistance)
        distances.remove(checkDistance)
            
        if returnValues == 0:
            return sortedDistances[0]
        else:
            return sortedDistances[0:returnValues-1]
        
    def RNN(w, u, b, x):
        yt = 0
        ht = 1 / 1 (w * x + u * yt ** -1 + b) ** -1
        yt = 1 / 1 (w * ht + b) ** -1

        return yt
        
class Audio:
    def __init__(self):
        self.model = whisper.load_model("base")
        
    def getTextFromAudio(self, audio):
        audio = whisper.load_audio("audio.mp3")
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        _, probs = self.model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)

        return result.text

class NLP:
    def __init__(self, text: str):
        self.text = text
        self.sentences = text.split(".")
        self.words = text.split(" ")
        self._past = ["was", "had", "did"]
        self._present = ["is", "has"]
        self._future = ["will", "shall"]

    def setTokensTo(self, letters: bool, *words: bool, **sentences: bool):
        self.tokens = []

        if letters:
            tokens = iter(self.text)
            for t in tokens:
                self.tokens.append(t)
        elif words:
            for t in self.words:
                self.tokens.append(t)
        elif sentences:
            for t in self.sentences:
                self.tokens.append(t)
        else:
            self.tokens.append("ERROR")

    def getTense(self):
        self.past = False
        self.present = False
        self.future = False

        if self.sentences in self._past:
            self.past = True
        elif self.sentences in self._present:
            self.present = True
        elif self.sentences in self._future:
            self.future = True
        else:
            return "ERROR - Tense :: Not Enough Data"

        return self.past, self.present, self.future

    def getWords(self):
        return self.words

    def getSentences(self):
        return self.sentences
    
    def getTokens(self):
        return self.tokens

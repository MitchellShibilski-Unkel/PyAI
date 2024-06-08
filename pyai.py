import spacy
import whisper
import numpy as np
from torch import nn
from torch import Tensor
from sklearn.tree import DecisionTreeRegressor


class PyAI:
    def __init__(self, useGPU: bool):
        self.GPU = useGPU
    
    class Algorithms:
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

        def ReLU(self, x: list, *y: list, **u: list):
            X, Y, U = [Tensor(x2) for x2 in x], [Tensor(y2) for y2 in y], [Tensor(u2) for u2 in u]
            
            if self.GPU:
                relu = nn.ReLU().to("cuda")
            else:
                relu = nn.ReLU().to("cpu")
                
            newX, newY, newU = [relu(x) for x in X], [relu(y) for y in Y], [relu(u) for u in U]
            
            if newU is not None:
                return newX, newY, newU
            elif newY is not None:
                return newX, newY
            else:
                return newX

        def Softmax(self, x):
            if self.GPU:
                tensor = Tensor(x, 1).to("cuda")
                soft = nn.Softmax(dim=1).to("cuda")(x)
            else:
                tensor = Tensor(x, 1).to("cpu")
                soft = nn.Softmax(dim=1).to("cpu")(x)

            return soft

        def decisionTree(self, trainX: list, trainY: list, words: list):
            w = np.array([len(a) for a in words]).reshape(-1, 1)
            tree = DecisionTreeRegressor()
            tree.fit(trainX, trainY)                                                           
            return tree.predict(w).tolist()
            
    class Audio:
        def __init__(self, audio: str):
            self.model = whisper.load_model("base")
            self.audio = audio
            
        def generateTextFromAudio(self) -> str:
            aud = whisper.load_audio(self.audio)
            aud = whisper.pad_or_trim(aud)

            self.mel = whisper.log_mel_spectrogram(aud).to(self.model.device)

            self.model.detect_language(self.mel)

            options = whisper.DecodingOptions()
            result = whisper.decode(self.model, self.mel, options)

            return result.text
        
        def translateText(self, text: str, dataSet: str) -> str:
            with open(dataSet, "r") as d:
                data = d.read()
                
            translation = text.translate(data)
            
            return translation
        
        def getLang(self):
            i, lang = self.model.detect_language(self.mel)
            return max(lang, key=lang.get)

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

        def getPartOfSpeech(self, text: str):
            POS = spacy.load("en_core_web_sm")
            return POS(text)[0].tag_
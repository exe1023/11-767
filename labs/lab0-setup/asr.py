import torch
from transformers  import AutoModelForSequenceClassification
from transformers import AutoTokenizer

# record voice
import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file 

# this part fix the wav file to a desired format
import soundfile
data, samplerate = soundfile.read('output.wav')
soundfile.write('new.wav', data, samplerate, subtype='PCM_16')

# read the correct wav file
print('reading wav...')
import speech_recognition as sr
r = sr.Recognizer()
with sr.AudioFile('new.wav') as source:
    audio = r.record(source)


# recognize speech using Google Speech Recognition
try:
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    print('ASR...')
    asr_result = r.recognize_google(audio)
    print("Google Speech Recognition thinks you said " + asr_result)

    print('loading the model...')    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = AutoModelForSequenceClassification.from_pretrained('/home/ubuntu/hate-model/')
    encoded = torch.tensor([tokenizer.encode(asr_result)])
    predict = model(encoded).logits.argmax(-1)[0]
    print(f"It is {'hateful' if predict == 1 else 'not hateful'}.")

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))


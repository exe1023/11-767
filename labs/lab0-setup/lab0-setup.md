Lab 0: Hardware setup
===
The goal of this lab is for you to become more familiar with the hardware platform you will be working with this semester, and for you to complete basic setup so that everyone in the group should be able to work remotely on the device going forward. By the end of class today, everyone in your group should be able to ssh in to the device, use the camera to take a picture, record audio, run a basic NLP model, and run a basic CV model. 

If you successfully complete all those tasks, then your final task is to write a script that pipes together I/O with a model. For example, you could write a script that uses the camera to capture an image, then runs classification on that image. Or you could capture audio, run speech-to-text, then run sentiment analysis on that text.

Group name:
---
Group members present in lab today:

1: Set up your device.
----
Depending on your hardware, follow the instructions provided in this directory: [Raspberry Pi 4](https://github.com/strubell/11-767/blob/main/labs/lab0-setup/setup-rpi4.md), [Jetson Nano](https://github.com/strubell/11-767/blob/main/labs/lab0-setup/setup-jetson.md), [Google Coral](https://coral.ai/docs/dev-board/get-started/). 
1. What device(s) are you setting up?

- Raspberry Pi.

2. Did you run into any roadblocks following the instructions? What happened, and what did you do to fix the problem?
- When flashing the Ubuntu server disk image to the SD card, BalenaEtcher will give you an error message if you don’t launch it with sudo.
- At first, we can’t connect Raspberry Pi to the screen. It is because, by default, there is only one HDMI port on Raspberry Pi that will work. 
- The IP address assigned to the device is not the address corresponding to the registered CMU-Device hostname. It looks like the IP assigned to hostname is already in use and not updated by the CMU side. We should directly use the IP address to ssh to the device.
- When installing python and pip, it shows the required Linux-libc-dev_5.4.0-81.91_arm64.deb package could not be found by the apt package manager (404 error). We directly download this package to install.
3. Are all group members now able to ssh in to the device from their laptops? If not, why not? How will this be resolved?

Yes, by directly using IP address all group members could ssh into the device.

2: Collaboration / hardware management plan
----
4. What is your group's hardware management plan? For example: Where will the device(s) be stored throughout the semester? What will happen if a device needs physical restart or debugging? What will happen in the case of COVID lockdown?

We will put the devices in our lab office. If we need to restart the device, anyone available will go to our lab office to do it. Considering COVID lockdown, if the LTI building is not closed we will still put it in the lab office. Otherwise, we will bring devices back home.


3: Putting it all together
----
5. Now, you should be able to take a picture, record audio, run a basic computer vision model, and run a basic NLP model. Now, write a script that pipes I/O to models. For example, write a script that takes a picture then runs a detection model on that image, and/or write a script that . Include the script at the end of your lab report.
6. Describe what the script you wrote does (document it.)

a. Record voice with package `soundvoice`.
b. Convert the audio into the format for Google ASR.
c. Get the speech recognition results from Google ASR.
d. Feed the ASR results into a hate speech detection model. The model is a bert-based model (`bert-base-uncased`) trained on part of the Twitter dataset used in [1].

7. Did you have any trouble getting this running? If so, describe what difficulties you ran into, and how you tried to resolve them.

a. Our kernel crash sometimes when we used the script for camera. Therefore, we decided to work on ASR.
b. The audio from `soundvoice` is in the format accepted by Google ASR. We used the package `soundfile` to convert it.
c. Our script is very slow. Both the ASR part and the Bert part are very slow. We may need to solve it in the future.


## References

[1] Zhou, Xuhui, et al. "Challenges in Automated Debiasing for Toxic Language Detection." Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume. 2021.

## Scripts

```py
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

```
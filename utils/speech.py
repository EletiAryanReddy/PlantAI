import pyttsx3
import threading

def speak(text):

    def run_speech():

        try:
            engine = pyttsx3.init()

            engine.setProperty('rate', 150)

            engine.say(text)

            engine.runAndWait()

        except Exception as e:

            print("Speech Error:", e)

    # Run speech in separate thread
    threading.Thread(
        target=run_speech
    ).start()


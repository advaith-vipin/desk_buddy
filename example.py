from pytoon.animator import animate

# Automatically creates a lip-synced animation from an audio file + transcript
animation = animate(
    audio_file=".tmp/speech.mp3",
    transcript="Hello, welcome to my application!",
)
animation.export(path="talking_face.output.mp4")
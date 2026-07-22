from reachy_mini import ReachyMini
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf
import time

# ==========================
# Loading model just one time
# ==========================

print("🧠 Loading Faster Whisper...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

# ==========================
# Conectting to Reachy
# ==========================

with ReachyMini(media_backend="webrtc") as mini:

    print("🤖 Reachy Ready")

    mini.media.start_recording()

    time.sleep(3)

    samplerate = mini.media.get_input_audio_samplerate()

    print("Samplerate:", samplerate)

    while True:

        print("\n🎤 Listening...")

        audio_chunks = []

        start = time.time()

        # Escuchar 2 segundos
        while time.time() - start < 0.75:

            samples = mini.media.get_audio_sample()

            if samples is not None:

                volume = np.max(np.abs(samples))

                if volume > 0.005:
                    audio_chunks.append(samples.copy())

            time.sleep(0.001)

        # No speach detected
        if len(audio_chunks) == 0:
            print("🔇 No speech detected")
            continue

        # ==========================
        # Process audio
        # ==========================

        audio = np.concatenate(audio_chunks, axis=0)

        # Stereo -> Mono
        audio = np.mean(audio, axis=1).astype(np.float32)

        # Normalizar
        max_val = np.max(np.abs(audio))

        if max_val > 0:
            audio = audio / max_val

        sf.write(
            "temp.wav",
            audio,
            samplerate
        )

        # ==========================
        # Transcript
        # ==========================

        t0 = time.time()

        segments, info = model.transcribe(
            "temp.wav",
            language="en",
            beam_size=1
        )

        print(
            "Transcription time:",
            round(time.time() - t0, 2),
            "seconds"
        )

        text = " ".join(
            segment.text
            for segment in segments
        ).strip().lower()

        if not text:
            continue

        print("📝 TEXT:", repr(text))

        words = text.split()

        # ==========================
        # Command HELLO
        # ==========================

        if any(word in words for word in ["hello", "hello."]):

            print("👋 I got it")

            try:

                mini.goto_target(
                    antennas=[0.5, -0.5],
                    duration=0.5
                )

                mini.goto_target(
                    antennas=[-0.5, 0.5],
                    duration=0.5
                )

                mini.goto_target(
                    antennas=[0, 0],
                    duration=0.5
                )

                print("🤖 Antennas moved!")

            except Exception as e:

                print("❌ Movement error:")
                print(e)

        # ==========================
        # Command GOODBYE
        # ==========================

        if "goodbye" in words:

            print("👋 Bye!")

            break

    mini.media.stop_recording()
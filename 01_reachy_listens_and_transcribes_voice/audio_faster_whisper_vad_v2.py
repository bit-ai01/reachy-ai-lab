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
# Connect to Reachy
# ==========================

with ReachyMini(media_backend="webrtc") as mini:

    print("🤖 Reachy Ready")

    mini.media.start_recording()

    time.sleep(3)

    samplerate = mini.media.get_input_audio_samplerate()

    print("Samplerate:", samplerate)

    # ==========================
    # VAD Parameters
    # ==========================

    THRESHOLD = 0.015
    SILENCE_SECONDS = 2

    while True:

        print("\n🎤 Waiting for speech...")

        audio_chunks = []

        # ==========================
        # WAIT FOR VOICE
        # ==========================

        while True:

            samples = mini.media.get_audio_sample()

            if samples is None:
                continue

            volume = np.max(np.abs(samples))

            #print(
            #    "Volume:",
            #    round(volume, 4)
           # )

            if volume > THRESHOLD:

                print("🗣️ Voice detected")

                audio_chunks.append(samples.copy())

                last_voice_time = time.time()

                break

        # ==========================
        # RECORD UNTIL SILENCE
        # ==========================

        while True:

            samples = mini.media.get_audio_sample()

            if samples is None:
                continue

            volume = np.max(np.abs(samples))

            audio_chunks.append(samples.copy())

            silence_time = round(
                time.time() - last_voice_time,
                2
            )

            #print(
            #    "Volume:",
            #    round(volume, 4),
            #    "| Silence:",
            #    silence_time,
            #    "sec"
           # )

            if volume > THRESHOLD:

                last_voice_time = time.time()

            if time.time() - last_voice_time > SILENCE_SECONDS:

                print("🔇 End of speech")

                break

        # ==========================
        # Safety Check
        # ==========================

        if len(audio_chunks) == 0:

            print("🔇 No speech detected")

            continue

        # ==========================
        # Process Audio
        # ==========================

        audio = np.concatenate(
            audio_chunks,
            axis=0
        )

        # Stereo -> Mono

        audio = np.mean(
            audio,
            axis=1
        ).astype(np.float32)

        # Normalize

        max_val = np.max(np.abs(audio))

        if max_val > 0:

            audio = audio / max_val

        sf.write(
            "temp.wav",
            audio,
            samplerate
        )

        # ==========================
        # Transcription
        # ==========================

        t0 = time.time()

        segments, info = model.transcribe(
            "temp.wav",
            language="en",
            beam_size=1,
            vad_filter=True
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
        # HELLO COMMAND
        # ==========================

        if "hello" in words:

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
        # GOODBYE COMMAND
        # ==========================

        if "goodbye" in words:

            print("👋 Bye!")

            break

    mini.media.stop_recording()
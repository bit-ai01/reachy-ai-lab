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

    THRESHOLD = 0.009
    SILENCE_LIMIT = 20

    while True:

        print("\n🎤 Waiting for speech...")

        audio_chunks = []
        silence_counter = 0

        # ==========================
        # WAIT FOR VOICE
        # ==========================

        while True:

            samples = mini.media.get_audio_sample()

            if samples is None:
                continue

            volume = np.max(np.abs(samples))
            #print("Volume:", round(volume, 4))
            if volume > THRESHOLD:

                print("🗣️ Voice detected")

                audio_chunks.append(samples.copy())

                break

        # ==========================
        # RECORD UNTIL SILENCE
        # ==========================

        while True:

            samples = mini.media.get_audio_sample()

            if samples is None:
                continue

            volume = np.max(np.abs(samples))
            #print(
             #       "Volume:",
              #      round(volume, 4),
               #     "| Silence:",
                #    silence_counter
                #)
            audio_chunks.append(samples.copy())

            if volume > THRESHOLD:

                silence_counter = 0

            else:

                silence_counter += 1
            
            #print("silence:", silence_counter)

            if silence_counter > SILENCE_LIMIT:

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
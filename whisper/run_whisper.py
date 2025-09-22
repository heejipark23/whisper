import whisper
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

model = whisper.load_model("large")

audio = whisper.load_audio("data/audio_ko.mp3")
# 길이 고정(기본 30초)
audio = whisper.pad_or_trim(audio)

# Log-Mel Spectrogram 변환
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
# 언어 감지: 가장 확률이 높은 언어 출력
_, probs = model.detect_language(mel)
print(f"Detected Language: {max(probs, key=probs.get)}")

# 디코딩 옵션 설정: translate
options = whisper.DecodingOptions(task="translate")
result = whisper.decode(model, mel, options)

print(result.text)
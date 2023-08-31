## 1. Setup
#### 複製l2arctic 到 espnet
`
rsync -avP l2arctic /path/to/your/espnet/egs2/
`
#### 安裝 Whisperx (conda create --name whisperx python=3.10)
[whiserpx](https://github.com/m-bain/whisperX#setup-%EF%B8%8F)

#### 安裝 YourTTS (conda create --name yourtts python=3.10)
[yourtts](https://github.com/coqui-ai/TTS#installation)


### 2. 打開 `path.sh`
```
# 修改成你的conda路徑 (whisperx && yourtts)
eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
conda activate whisperx
```

### 3. whisper decoding + WER
```
./run.sh --stage 0
```

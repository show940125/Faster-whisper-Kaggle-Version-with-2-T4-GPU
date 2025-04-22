# Faster-whisper-Kaggle-Version-with-2-T4-GPU
媒體、政治工作者的長篇逐字稿救星

<div align="center">
  <a href="#繁體中文" style="margin-right: 20px;">
    <img src="https://img.shields.io/badge/繁體中文-000?style=for-the-badge&logo=translate&logoColor=white" alt="繁體中文">
  </a>
  <a href="#english">
    <img src="https://img.shields.io/badge/English-000?style=for-the-badge&logo=translate&logoColor=white" alt="English">
  </a>
</div>

---
## 繁體中文

### 本專案提供媒體工作者、政治工作者及苦逼工讀生擺脫不效率的**聽打困境**，特別設計一個高效的音頻轉錄的傻瓜操作方案。
### 為何不使用colab？
1. 因為獲取GPU的限制與不穩定性，即便掛載drive可以讓整個程序更容易使用，但轉錄任務的重點仍在GPU。
2. 因此，為了充分利用所有免費資源，使用 Kaggle 平台免費提供的兩張 T4 GPU 資源，可以大幅提升轉錄效率，助您輕鬆完成各類轉錄任務。**[更新]** **新代碼會自動偵測可用的 GPU 數量，並合理分配任務。**
3. kaggle是一個免費的機器學習平台，特點是每個星期提供30個小時的免費GPU，相較於不穩定的colab，絕對足以滿足日常的工作任務。
### 本人已脫離政治工作，也不知道誰會看到本專案(因為通常這類工作的人根本不會打開github)，幫助後輩少走彎路是我對前一份工作的執念，目前這些代碼集成已經接近穩定，後續是否還有優化空間我再問問GPT~ **[更新]** **新代碼整合了批次處理 (Batch Processing) 和更精細的並行控制，效率更高。**
### 的確網路上有比較快的Demo，比如whisperJAX、Whisper web gpu甚至groq api等等，但無法使用客製化模型，也無法使用較大的音檔(通常超過25MB~=30分鐘低音值mp3檔就不太能用)
### ☆贈與有緣人~反正整套目前都不用花錢~
### ~~☆新增：**FW1.1.1 with Batched pipeline full code (2024/11/22)~~ **[更新]** **目前代碼已整合 `BatchedInferencePipeline`，並提供更完善的自動化處理流程 (2025/04/22)。**

## 目錄

- [功能](#功能)
- [技術特色](#技術特色)
- [安裝與設定](#安裝與設定)
- [使用方法](#使用方法)
- [範例代碼](#範例代碼)
- [貢獻](#貢獻)
- [授權條款](#授權條款)
- [致謝](#致謝)
- [聯繫方式](#聯繫方式)

## 功能

1.  **[更新]** **多 GPU 自動化並行處理**
    *   **自動 GPU 偵測與分配**：自動偵測 Kaggle 環境中可用的 T4 GPU 數量 (最多支援4個)，並以輪詢 (Round-Robin) 方式將音檔分配給不同 GPU 處理。
    *   **批次推理 (Batch Inference)**：利用 `BatchedInferencePipeline` 將多個轉錄請求打包處理，大幅提升 GPU 利用率和整體吞吐量。
    *   **單 GPU 並行控制**：可設定每張 GPU 同時處理的最大任務數 (`MAX_CONCURRENCY_PER_GPU`)，避免單一 GPU 過載，確保穩定運行。
    *   **轉錄文本合併與時間戳記接續**：將多個轉錄結果合併為一個文件，並自動接續時間戳記，確保時間軸連續。

2.  **模型本地化與加載優化**
    *   **模型預先下載與本地加載**：提前下載適用 `faster-whisper` 的模型並上傳至 Kaggle 的 Datasets 或 Models，避免每次運行時重新下載，提高預熱速度。
    *   **[新增]** **多 GPU 模型獨立加載**：為每個偵測到的 GPU 獨立加載模型實例，確保並行處理順暢。

3.  **轉錄文本的分段與打印優化**
    *   **固定時間間隔分段**：基於可配置的 `SEGMENT_DURATION` (預設 30 秒) 時間間隔進行文本分段，提升上下文可讀性，方便後續校正。
    *   **即時進度打印**：轉錄過程中，逐段打印帶時間戳的文本到控制台，方便即時監控進度。**[新增]**
    *   **gemini後續校正**：利用Google AI Studio可調用免費gemini 2.5 /proflash進行高精度免費校稿，經多次測試驗證，每次校正約6500token之份量可以兼顧穩定的校正品質以及效率。

4.  **[更新]** **易用性與錯誤處理**
    *   **自動音檔掃描**：自動遞迴掃描指定根目錄 (`AUDIO_ROOT`) 下的所有音檔 (支援 `.wav`, `.flac`, `.mp3`, `.ogg` 等格式)，無需手動指定每個文件。
    *   **自動輸出命名**：根據掃描到的音檔順序，自動生成 `01.txt`, `02.txt`... 等輸出檔名。
    *   **參數集中管理**：將常用參數 (模型路徑、音檔根目錄、批次大小、並行數、替換詞等) 集中在代碼開頭，方便修改。
    *   **錯誤捕捉**：對單個文件的轉錄過程進行錯誤捕捉，避免單一文件失敗導致整個程序中斷。

5.  **時間戳記位置與格式調整**
    *   **時間戳記置於段落開頭**：統一時間戳記格式 (`HH:MM:SS-HH:MM:SS`)，確保每個段落的時間戳記位於文本最前端。

## 技術特色

- **[更新]** **高效批次推理**：利用 `BatchedInferencePipeline` 實現高效批次處理，最大化 GPU 吞吐量。
- **[更新]** **自動化多 GPU 管理**：自動偵測並利用所有可用 GPU (最多4個)，智能分配任務。
- **[更新]** **精細化並行控制**：通過 `threading.Semaphore` 控制單張 GPU 的並行任務數，平衡效率與穩定性。
- **[更新]** **自動化文件處理**：自動掃描音檔、自動命名輸出文件，簡化操作流程。
- **高效的多執行緒處理**：使用 `concurrent.futures.ThreadPoolExecutor` 實現多執行緒調度。~~經測試，多進程效率低於多線程(2024/12/11，FW=1.1.0)~~ **[註]** **新代碼依然使用多線程，結合 Semaphore 控制並行度。**
- **優化的模型加載**：將 `faster-whisper` 適用的模型上傳 Kaggle，避免每次重複下載，減少轉錄任務前的準備時間。
- **靈活的文本分段方式**：支持固定時間間隔 (`SEGMENT_DURATION`) 分段方式，提升轉錄文本的可讀性。
- **自動化的轉錄文本合併**：自動接續時間戳記，確保多個轉錄文件的時間軸連續。

## 安裝與設定

### 環境需求
- **Python 版本**: 3.8+
- **平台**: Kaggle（Kaggle 已經為你預先配置了所需環境）
- **硬體**: 兩張 T4 GPU（由 Kaggle 提供免費資源）

### 運行步驟

#### 1. 安裝必要的 Python 套件

在新開的 Kaggle Notebook 中，執行以下命令來安裝 `faster-whisper` 套件：(約20秒)
#### 2024/10/26**更新**:ctranslate2最新版在cuda相容性上貌似出現問題，目前以退回版本方式處理
#### 2024/12/11**更新**:~FW1.1.0版本中問題似乎已經解決，可以拿掉ctranslate2==4.4.0。~
#### 2025/04/22**更新**:FW1.1.1版本下問題受限於平台環境的依賴版本，目前仍需ctranslate2==4.4.0。 **[註]**:平台環境目前不穩定，建議回滾到去年以前的環境或等更新

```python
!pip install faster-whisper==1.1.1 ctranslate2==4.4.0 -q
# 使用 -q 安靜模式安裝
```

#### 2. 上傳模型至 Kaggle，並在notebook中加載

接下來，將 `faster-whisper` 模型從 Hugging Face 下載，並上傳至 Kaggle。請遵循以下步驟：

1. 前往 Hugging Face 模型頁面：範例[faster-whisper-large-v2-zh-TW](https://huggingface.co/XA9/faster-whisper-large-v2-zh-TW)
2. 點擊頁面中的 **Files and version** 按鈕，將模型的所有檔案下載到本地電腦。
3. 登入 Kaggle，並在頁面中點擊**creat**，選擇 **Create New model**。
4. 在建立新的 model 時，為它取一個名稱，例如：`faster-whisper-large-v2-zh-TW`。
5. 將下載好的所有檔案上傳到該 model 資料夾中。
6. 上傳完成後，點擊 **Create** 按鈕，即可完成模型上傳至 Kaggle。
7. 首次使用項目時，點擊右邊"input"下的"add input"，選"models"以及"your work"正確找到你上傳好的model，之後再啟動notebokk就會自動加載(拍手~)

備註：largeV3模型微調的model在轉錄速度上會比largeV2的快10%~20%，但精度上，基於largeV2微調的模型對於中文的適應性較佳，範例提供的版本已經是精度最佳的版本之一。

####  [更新] 上傳音檔至 Kaggle Dataset


1. 從你的裝置將欲轉錄的音檔切一半，你可以用ffmpeg或是用剪映都可
2. 跟上傳模型步驟類似，只是要選擇dataset，你可以在同一個dataset中上傳複數個音檔
3. 使用notebook的時候"add input"將音檔匯入notebook中，方法跟model一樣
4. 之後代碼對自動捕捉音檔，不用管他


## 使用方法

在新開的kaggle notebook中，根據以下代碼逐步執行，等於傻瓜操作喔~

## 範例代碼
(請按代碼塊複製貼上就行，讓你改你再改)

### 轉錄音頻文件
1. 注意MODEL_PATH = "複製你input的whisper model的路徑(直接點COPY就好)"
2. 其他參數已經經過超過300小時以上的轉錄任務實際驗證，不太需要調整
3. replacements(用Ctrl+f查找)部分能提供固定轉錄任務(比如多次轉錄同一個講者的音檔)較佳的體驗，把固定的錯漏字直接替換成正確的文字，格式為"錯字": "正確字",
4. 本專案(按範例模型)實測3小時音頻文件(WAV檔，同常大小約300MB)的轉錄任務約需9分鐘轉錄時間，準確度平均在90%以上，再透過gemini校正，準確率可達99%，不唬爛。
5. 錄音筆建議預設錄製WAV/flac檔，精度確實優於MP3檔。WAV檔(192K&256K)大概是音質影響精度的極限，再大則無用。

```python
# ---------- import字典 ----------
from faster_whisper import WhisperModel, BatchedInferencePipeline
import datetime, time, os, re, torch, glob
from typing import List, Tuple, Dict
import concurrent.futures, threading

# ---------- 可調參數 ----------
MODEL_PATH = "/kaggle/input/faster-whisper..."
AUDIO_ROOT = "/kaggle/input"            # 只改這裡就能換資料來源
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")   # 允許的音檔副檔名
SEGMENT_DURATION = 30.0                          # 每段最長秒數
BATCH_SIZE = 8
MAX_CONCURRENCY_PER_GPU = 2                     # 同張卡並行上限
REPLACEMENTS: Dict[str, str] = {                # 常見錯字修正表
    "XX": "OO" 
}
INITIAL_PROMPT = "XXX"                      # 給模型的 system prompt（可留空）

# ---------- 自動收集音檔 ----------
def collect_audio_files(root: str, exts=AUDIO_EXTS) -> List[str]:
    """
    遞迴走訪 root 底下所有子目錄，
    只要檔名副檔名（不論大小寫）符合 exts，就收進來。
    """
    exts_lower = {e.lower() for e in exts}
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts_lower:
                files.append(os.path.join(dirpath, fn))
    return sorted(files)

# ---------- 建立 (音檔, 輸出檔, GPU index) 對照表 ----------
def create_job_table(audio_files: List[str], gpu_count: int) -> List[Tuple[str, str, int]]:
    jobs = []
    for idx, path in enumerate(audio_files, start=1):
        out_name = f"{idx:02d}.txt"
        gpu_idx = idx % gpu_count  # round‑robin
        jobs.append((path, out_name, gpu_idx))
    return jobs

# ---------- 取代/清洗工具 ----------
pattern = re.compile("|".join(re.escape(k) for k in REPLACEMENTS.keys()))
def clean_text(txt: str) -> str:
    txt = txt.lstrip("! ")
    return pattern.sub(lambda m: REPLACEMENTS[m.group(0)], txt)

def to_timestamp(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def fmt_chunk(start: float, end: float, txt: str) -> str:
    return f"{to_timestamp(start)}-{to_timestamp(end)} {txt.strip()}\n"

# ---------- 轉錄 + 寫檔 ----------
def process_segments(segments, outfile: str, max_len=SEGMENT_DURATION):
    buf, chunk_start, chunk_txt = "", 0.0, ""
    for seg in segments:
        chunk_txt += " " + clean_text(seg.text)
        if seg.end - chunk_start >= max_len:
            line = fmt_chunk(chunk_start, seg.end, chunk_txt)
            print(line, end="", flush=True)
            buf += line
            chunk_start, chunk_txt = seg.end, ""
    if chunk_txt:
        line = fmt_chunk(chunk_start, seg.end, chunk_txt)
        print(line, end="", flush=True)
        buf += line
    with open(outfile, "w", encoding="utf‑8") as fh:
        fh.write(buf)
    print(f" ✔ 已寫入 {outfile}")

def transcribe_single(job, pipelines, semaphores):
    in_path, out_path, gpu_idx = job
    sem = semaphores[gpu_idx]
    with sem:  # 限制同一張卡的並行數
        try:
            segments, _info = pipelines[gpu_idx].transcribe(
                in_path,
                batch_size=BATCH_SIZE,
                word_timestamps=True,
                hallucination_silence_threshold=3,
                initial_prompt=INITIAL_PROMPT or None,
                beam_size=5,
                temperature=0,
                patience=1.5,
                language="zh",
                max_new_tokens=256,
                condition_on_previous_text=False,
                no_repeat_ngram_size=3,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 250, "speech_pad_ms": 600},
                log_progress=True,
            )
            process_segments(segments, out_path)
        except Exception as exc:
            print(f" ✘ 轉錄失敗: {in_path} ({exc})")

# ---------- 主流程 ----------
def main():
    # 1. 檢查 GPU
    gpu_count = torch.cuda.device_count() or 1   # 沒 GPU 時 fallback CPU
    if gpu_count > 4:                            # Kaggle 通常 1 張卡；這裡只是保險
        gpu_count = 4
    print(f"偵測到 GPU 數量：{gpu_count}")
    
    # 2. 掃描音檔
    audio_files = collect_audio_files(AUDIO_ROOT)
    if not audio_files:
        raise RuntimeError(f"找不到任何音檔於 {AUDIO_ROOT}")
    job_table = create_job_table(audio_files, gpu_count)
    
    # 3. 建立每張卡各自的模型與 pipeline
    pipelines = {}
    for idx in range(gpu_count):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pipelines[idx] = BatchedInferencePipeline(
            WhisperModel(MODEL_PATH, device=dev, device_index=idx, compute_type="float16")
        )
        print(f"GPU {idx} 模型初始化完成")
    
    # 4. 為每張卡準備 Semaphore，控制並行度
    semaphores = {idx: threading.Semaphore(MAX_CONCURRENCY_PER_GPU) for idx in range(gpu_count)}
    
    # 5. 多執行緒並行轉錄
    workers = gpu_count * MAX_CONCURRENCY_PER_GPU
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(transcribe_single, job, pipelines, semaphores) for job in job_table]
        for f in concurrent.futures.as_completed(futures):
            pass  # 錯誤已在 transcribe_single 內捕捉
    
    print("🎉 所有轉錄任務完成")

# Entry
if __name__ == "__main__":
    audio_list = collect_audio_files(AUDIO_ROOT)
    print(f"共找到 {len(audio_list)} 個音檔，前 10 筆：")
    for p in audio_list[:10]:
        print("  ", p)
    tic = time.time()
    main()
    print(f"總耗時：{time.time() - tic:.1f} 秒")
```

### 合併轉錄文本
承上代碼，預命名規則為01.txt->02.txt，如需合併請使用以下代碼，確定最底下的組合是否正確，即可按需合併囉~(檔名預設為merged_output.txt)
```python
def merge_transcriptions(file1, file2, output_file):
    def parse_timestamp(timestamp_str):
        """將 XX:XX:XX 格式的時間戳記轉換為秒數"""
        h, m, s = map(int, timestamp_str.split(':'))
        return h * 3600 + m * 60 + s

    def format_timestamp(seconds):
        """將秒數轉換為 XX:XX:XX 格式的時間戳記"""
        return format_to_custom_timestamp(seconds)
    
    merged_content = ""
    total_duration = 0

    # 讀取第一個文件的內容
    with open(file1, 'r', encoding="utf-8") as f1:
        for line in f1:
            # 假設時間戳記在行的開頭，並且格式為 XX:XX:XX-XX:XX:XX
            time_range, text = line.split(' ', 1)
            start_time_str, end_time_str = time_range.split('-')
            
            # 將時間戳記轉換為秒數
            start_time = parse_timestamp(start_time_str)
            end_time = parse_timestamp(end_time_str)
            
            # 更新時間戳記以包括累積的總時間
            new_start_time = format_timestamp(start_time + total_duration)
            new_end_time = format_timestamp(end_time + total_duration)
            
            # 合併更新後的行
            merged_content += f"{new_start_time}-{new_end_time} {text}"
        
        # 更新累積的總時間
        total_duration = parse_timestamp(new_end_time)
    
    # 讀取第二個文件的內容並合併
    with open(file2, 'r', encoding="utf-8") as f2:
        for line in f2:
            time_range, text = line.split(' ', 1)
            start_time_str, end_time_str = time_range.split('-')
            
            # 將時間戳記轉換為秒數
            start_time = parse_timestamp(start_time_str)
            end_time = parse_timestamp(end_time_str)
            
            # 更新時間戳記以包括累積的總時間
            new_start_time = format_timestamp(start_time + total_duration)
            new_end_time = format_timestamp(end_time + total_duration)
            
            # 合併更新後的行
            merged_content += f"{new_start_time}-{new_end_time} {text}"
    
    # 將合併後的內容保存到輸出文件
    with open(output_file, 'w', encoding="utf-8") as out_file:
        out_file.write(merged_content)

    print(f"Transcriptions merged and saved to {output_file}")

# 假設已經成功生成了 01.txt 和 02.txt
merge_transcriptions("01.txt", "02.txt", "merged_output.txt")
```

## 貢獻

歡迎任何形式的貢獻！不會或要做調整自己問GPT喔~我也是慢慢問出來的~

## 授權條款

本專案採用 [MIT License](LICENSE) 授權，詳情請參閱 [LICENSE](LICENSE) 文件。


## 第三方資源授權

- **faster-whisper**：本專案使用 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 庫，該庫採用 MIT 授權。
- **faster-whisper-large-v2-zh-TW 模型**：本專案範例模型使用來自 [Hugging Face](https://huggingface.co/XA9/faster-whisper-large-v2-zh-TW) 的模型，請遵守其授權條款。

## 致謝

感謝以下項目和資源對本專案的支持：

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 開發團隊
- [Hugging Face](https://huggingface.co/XA9/faster-whisper-large-v2-zh-TW) 提供的優秀模型資源
- [Kaggle](https://www.kaggle.com/) 提供的免費 GPU 資源

## 聯繫方式

如有任何問題或建議，請通過以下方式聯繫：

- **電子郵件**：a0953041880@gmail.com
- **GitHub Issues**：略

感謝您的關注與支持！醬~下次見

## English

### Welcome to **show940125**'s GitHub repository! This project offers media professionals, political workers, and diligent student assistants a solution to overcome the inefficiency of **transcription challenges**. It is a well-designed and user-friendly audio transcription method.

### Why Not Use Colab?

1. **GPU Availability and Stability**: Colab imposes limitations and instability in accessing GPUs. Even though mounting Google Drive can simplify the workflow, the core of transcription tasks relies heavily on GPU performance.
2. **Maximizing Free Resources**: To fully utilize all available free resources, this project leverages Kaggle's two free T4 GPUs, significantly enhancing transcription efficiency and enabling you to effortlessly complete various transcription tasks.
3. **Reliable Free GPU Access**: Kaggle is a free machine learning platform that offers 30 hours of free GPU usage each week. Compared to the unreliable nature of Colab, Kaggle's offerings are more than sufficient to meet daily work requirements.

### Personal Note

Having moved away from a career in political work, I understand that individuals in such roles might not frequently visit GitHub. My dedication to helping future generations avoid the pitfalls of my previous job is the driving force behind this project. The current codebase is nearing stability, and any potential optimizations will be explored with the help of GPT in the future.

### Existing Online Demos

There are faster demos available online, such as whisperJAX, Whisper Web GPU, and even Groq API. However, these solutions lack the ability to use customized models and cannot handle larger audio files (typically over 25MB or approximately 30 minutes of low-bitrate MP3 files) easily.

### ☆ A Gift for the Worthy~

This entire setup is currently free of charge, and that is the point.

## Table of Contents

- [Features](#features)
- [Technical Highlights](#technical-highlights)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Code Examples](#code-examples)
- [Contributing](#contributing)
- [License](#license)
- [Third-Party Licensing](#third-party-licensing)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Features

1. **Dual GPU Parallel Processing and Merging Functionality Expansion**
   - **Dual GPU Parallel Processing**: Supports simultaneous processing of two audio files using dual T4 GPUs, significantly enhancing transcription efficiency.
   - **Transcription Text Merging and Timestamp Continuity**: Merges multiple transcription results into a single file with automatic timestamp continuation, ensuring a continuous timeline.

2. **Model Localization and Loading Optimization**
   - **Pre-downloaded Model and Local Loading**: Pre-download models compatible with `faster-whisper` and upload them to Kaggle's module to avoid re-downloading each run, improving preheat speed.

3. **Text Segmentation and Printing Optimization**
   - **Fixed Time Interval Segmentation**: Segments text based on fixed 30-second intervals to enhance readability.
   - **Hybrid Natural and Fixed Segmentation**: Combines natural sentence segmentation with fixed time intervals to maintain sentence coherence and paragraph readability.

4. **Resolution of Repeated Sentences Issue**
   - **Corrected Paragraph Accumulation and Clearing Logic**: Prevents the occurrence of repeated sentences, ensuring clean text output.

5. **Timestamp Position and Format Adjustment**
   - **Timestamps at the Beginning of Paragraphs**: Standardizes timestamp format, ensuring each paragraph's timestamp is at the very beginning of the text.

## Technical Highlights

- **Dual GPU Parallel Processing**: Fully utilizes Kaggle’s two T4 GPUs, doubling transcription efficiency.
- **Efficient Multithreading**: Implements multithreading using `concurrent.futures.ThreadPoolExecutor` to maximize GPU resource utilization.
- **Optimized Model Loading**: Localizes `faster-whisper` compatible models on Kaggle, avoiding repeated downloads and reducing preparation time to under 1 minute.
- **Flexible Text Segmentation**: Supports both fixed time interval and natural sentence segmentation methods to improve transcription text readability.
- **Automated Transcription Merging**: Automatically continues timestamps to ensure a continuous timeline across multiple transcription files.

## Installation and Setup

### Environment Requirements
- **Python Version**: 3.8+
- **Platform**: Kaggle (pre-configured environment)
- **Hardware**: Two T4 GPUs (provided free by Kaggle)

### Running Steps

#### 1. Install Required Python Packages

In a new Kaggle Notebook, execute the following command to install the `faster-whisper` package (approximately 20 seconds):

```python
%%time
pip install faster-whisper==1.1.1
```

#### 2. Upload Model to Kaggle and Load in Notebook

Next, download the `faster-whisper` model from Hugging Face and upload it to Kaggle. Follow these steps:

1. **Visit the Hugging Face Model Page**: Example [faster-whisper-large-v2-zh-TW](https://huggingface.co/XA9/faster-whisper-large-v2-zh-TW)
2. **Download Model Files**: Click the **Files and versions** button on the page to download all model files to your local computer.
3. **Upload to Kaggle**:
   - Login to Kaggle and click **Create** > **Create New Model**.
   - Name the model, for example, `faster-whisper-large-v2-zh-TW`.
   - Upload all downloaded files to the model's directory.
   - After uploading, click **Create** to complete the model upload to Kaggle.
4. **Add Model to Notebook**:
   - When first using the project, click **Add Input** under the "Input" section, select **Models**, and locate your uploaded model under "your work".
   - Once added, the notebook will automatically load the model. (Applause~)

**Note**: The `largeV3` model is 10%–20% faster than `largeV2` in transcription speed, but the `largeV2` model fine-tuned for Chinese offers better accuracy. The provided example uses one of the most accurate versions available.

#### 3. Upload Audio Files to Kaggle and Load in Notebook

1. **Split Audio Files**: Divide your audio files in half using tools like `ffmpeg` or any audio editing software.
2. **Upload to Dataset**: Similar to uploading the model, but select **Dataset**. You can upload multiple audio files within the same dataset.
3. **Import Audio Files into Notebook**:
   - In the notebook, click **Add Input** to import the audio files, following the same method as uploading the model.
4. **Copy Audio File Paths**: **Important**: In the subsequent code, copy (Ctrl + C) the paths of the audio files to the designated locations.

## Usage

In a new Kaggle Notebook, follow the steps below by executing the provided code blocks.

## Code Examples
(Copy and paste the code blocks, make adjustments as needed)

### Transcribing Audio Files

1. **Set `MODEL_PATH`**: Copy the path of your input Whisper model (simply click COPY).
2. **Other Parameters**: These have been validated through over 300 hours of transcription tasks and generally do not require adjustments.
3. **Replacements**: The `replacements` section (use Ctrl + F to find) provides a better experience for fixed transcription tasks (e.g., transcribing the same speaker multiple times). Directly replace fixed misspelled words with correct ones in the format `"incorrect_word": "correct_word"`.
4. **Audio Path Settings**: Set the audio paths in the two paths under `def main():`, with the first path at the top and the second at the bottom. The resulting two TXT files are preset as `04.txt` and `05.txt`. Due to the need for subsequent merging, renaming is not recommended; otherwise, rename synchronously during merging.
5. **Performance**: This project (using the example model) has been tested with a 3-hour audio file (WAV format, approximately 300MB). The transcription task takes about 9 minutes, with an average accuracy above 95% in chinese. Through replacement rules, accuracy can reach 99%, no exaggeration.
6. **Recording Recommendations**: It is recommended to use WAV files for transcrbing, as they offer better accuracy than MP3 files. WAV files (192K & 256K) are approximately the limit of audio quality affecting accuracy; larger files are ineffective.

```python
from faster_whisper import WhisperModel
import datetime
import os
import logging
import concurrent.futures
from typing import List, Tuple

# Configure Logging
logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

import concurrent.futures
from typing import List, Tuple

# Set constants
MODEL_PATH = "/kaggle/input/faster-whisper-large-v2-zh-tw/transformers/default/1"
SEGMENT_DURATION = 30.0
MAX_WORKERS = 2

# Define multiple audio files
files: List[Tuple[str, str, int]] = [
    ("/kaggle/input/ccl07-08/CLL08_part1.wav", "04.txt", 0),
    ("/kaggle/input/ccl07-08/CLL08_part2.wav", "05.txt", 1)
]

def transcribe_audio(input_file: str, output_file: str, device_index: int, model, segment_duration: float = SEGMENT_DURATION) -> None:
    # Use the provided model to transcribe
    segments, info = model.transcribe(
        input_file, 
        word_timestamps=True, 
        initial_prompt=None,
        beam_size=4, 
        language="zh", 
        max_new_tokens=192, 
        condition_on_previous_text=False,
        vad_filter=True, 
        vad_parameters=dict(min_silence_duration_ms=300)
    )
    
    process_segments(segments, output_file, segment_duration)

def process_segments(segments, output_file: str, segment_duration: float) -> None:
    # Process transcription segments
    txt_content = ""
    current_segment_start = 0.0
    current_segment_text = ""

    for segment in segments:
        start_time, end_time = segment.start, segment.end
        text = replace_special_chars(segment.text)
        
        current_segment_text += " " + text

        if float(end_time) - float(current_segment_start) >= segment_duration:
            formatted_segment = format_segment(current_segment_start, end_time, current_segment_text)
            print(formatted_segment)
            txt_content += formatted_segment
            
            current_segment_start = float(end_time)
            current_segment_text = ""
    
    # Handle the last segment
    if current_segment_text:
        formatted_segment = format_segment(current_segment_start, end_time, current_segment_text)
        print(formatted_segment)
        txt_content += formatted_segment
    
    # Write the result to file
    with open(output_file, 'w', encoding="utf-8") as txt_file:
        txt_file.write(txt_content)
    
    print(f"Saved: {os.path.abspath(output_file)}")

def format_segment(start_time: float, end_time: float, text: str) -> str:
    # Format a single segment
    start_time_str = format_to_custom_timestamp(start_time)
    end_time_str = format_to_custom_timestamp(end_time)
    return f"{start_time_str}-{end_time_str} {text.strip()}\n"

def replace_special_chars(text: str) -> str:
    # Replace special characters and correct common errors
    if text.startswith(("! ", " ")):
        text = text.lstrip("! ")
    
    replacements = {
        "XX": "OO", 
    }
    
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    
    return text

def format_to_custom_timestamp(seconds: float) -> str:
    # Convert seconds to custom timestamp format
    dt = datetime.datetime(1, 1, 1) + datetime.timedelta(seconds=seconds)
    return f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"

def main():
        
    # Preload models for each device
    device_indices = set([device_index for _, _, device_index in files])
    models = {}
    for device_index in device_indices:
        models[device_index] = WhisperModel(
            MODEL_PATH, device="cuda", device_index=device_index, compute_type="float16"
        )

    # Use thread pool to process audio files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Pass the models here
        futures = [executor.submit(transcribe_audio, input_file, output_file, device_index, models[device_index]) 
                   for input_file, output_file, device_index in files]

        for future in concurrent.futures.as_completed(futures):
            future.result()

    print("All transcription tasks are completed.")

if __name__ == "__main__":
    main()
```

### Merging Transcription Texts

Simply copy and execute to merge the two transcribed documents directly (default filename is `merged_output.txt`). Renaming is not recommended to avoid issues during merging.

```python
def merge_transcriptions(file1, file2, output_file):
    def parse_timestamp(timestamp_str):
        """Convert timestamp in XX:XX:XX format to seconds"""
        h, m, s = map(int, timestamp_str.split(':'))
        return h * 3600 + m * 60 + s

    def format_timestamp(seconds):
        """Convert seconds to XX:XX:XX timestamp format"""
        return format_to_custom_timestamp(seconds)
    
    merged_content = ""
    total_duration = 0

    # Read the first file's content
    with open(file1, 'r', encoding="utf-8") as f1:
        for line in f1:
            # Assume timestamp is at the beginning of the line in XX:XX:XX-XX:XX:XX format
            time_range, text = line.split(' ', 1)
            start_time_str, end_time_str = time_range.split('-')
            
            # Convert timestamp to seconds
            start_time = parse_timestamp(start_time_str)
            end_time = parse_timestamp(end_time_str)
            
            # Update timestamps to include the cumulative total time
            new_start_time = format_timestamp(start_time + total_duration)
            new_end_time = format_timestamp(end_time + total_duration)
            
            # Merge the updated line
            merged_content += f"{new_start_time}-{new_end_time} {text}"
        
        # Update the cumulative total time
        total_duration = parse_timestamp(new_end_time)
    
    # Read the second file's content and merge
    with open(file2, 'r', encoding="utf-8") as f2:
        for line in f2:
            time_range, text = line.split(' ', 1)
            start_time_str, end_time_str = time_range.split('-')
            
            # Convert timestamp to seconds
            start_time = parse_timestamp(start_time_str)
            end_time = parse_timestamp(end_time_str)
            
            # Update timestamps to include the cumulative total time
            new_start_time = format_timestamp(start_time + total_duration)
            new_end_time = format_timestamp(end_time + total_duration)
            
            # Merge the updated line
            merged_content += f"{new_start_time}-{new_end_time} {text}"
    
    # Save the merged content to the output file
    with open(output_file, 'w', encoding="utf-8") as out_file:
        out_file.write(merged_content)

    print(f"Transcriptions merged and saved to {output_file}")

# Assume 04.txt and 05.txt have been successfully generated
merge_transcriptions("04.txt", "05.txt", "merged_output.txt")
```

## Contributing

Contributions are welcome in any form! If you need assistance or wish to make adjustments, feel free to consult GPT. I also developed this project gradually with its help~

## License

This project is licensed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for details.

## Third-Party Licensing

- **faster-whisper**: This project uses the [faster-whisper](https://github.com/SYSTRAN/faster-whisper) library, which is licensed under the MIT License.
- **faster-whisper-large-v2-zh-TW Model**: The example model used in this project is sourced from [Hugging Face](https://huggingface.co/XA9/faster-whisper-large-v2-zh-TW). Please adhere to its [licensing terms](https://huggingface.co/XA9/faster-whisper-large-v2-zh-TW).

## Acknowledgements

Special thanks to the following projects and resources for supporting this project:

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) development team
- [Hugging Face](https://huggingface.co/XA9/faster-whisper-large-v2-zh-TW) for providing excellent model resources
- [Kaggle](https://www.kaggle.com/) for providing free GPU resources

## Contact

If you have any questions or suggestions, please reach out through the following methods:

- **Email**: a0953041880@gmail.com
- **GitHub Issues**: Not available

Thank you for your attention and support! See you next time~

---

> **Note**: This project makes full use of the free GPU resources provided by Kaggle. Please comply with Kaggle's terms of use, use resources responsibly, and avoid misuse.

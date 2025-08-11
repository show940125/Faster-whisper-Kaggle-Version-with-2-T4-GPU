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
### ~~☆新增：**FW1.1.0 with Batched pipeline full code (2024/11/22)~~ **[更新]** **目前代碼已整合 `BatchedInferencePipeline`，並提供更完善的自動化處理流程 (2025/04/22)。**

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
    *   **首段起點修正**：以第一個語音片段的 start 作為段落起點，避免前置靜音造成時間戳誤差。
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
#### 2025/04/22**更新**:~FW1.2.0版本下問題受限於平台環境的依賴版本，目前仍需ctranslate2==4.4.0。 **[註]**:平台環境目前不穩定，建議回滾到去年以前的環境或等更新~
#### 2025/06/30**更新**:`ctranslate2` 已升級，並隨 `faster-whisper==1.1.1` 一併安裝，相關 CUDA 相容性問題已解決，無需再額外安裝 `ctranslate2==4.4.0`。
#### 2025/08/11**更新**:更新版本faster-whisper==1.2.0

```python
# 安裝 faster-whisper (已內含 ctranslate2)
!pip install faster-whisper==1.2.0 -q
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
4. 本專案(按範例模型)實測3小時音頻文件(WAV檔，256K同常大小約300MB)的轉錄任務約需3至4鐘轉錄時間，準確度平均在90%以上，再透過gemini校正，準確率可達99%，不唬爛。
5. 錄音筆建議預設錄製WAV/flac檔，精度確實優於MP3檔。WAV檔(192K&256K)大概是音質影響精度的極限，再大則無用。

```python
%%time
from faster_whisper import WhisperModel, BatchedInferencePipeline
import datetime, time, os, re, torch, glob
from typing import List, Tuple, Dict
import concurrent.futures, threading, multiprocessing as mp

%%time
# ================================================================
# 1. 參數區（可調）
# ================================================================
# [1.0] 路徑設定
MODEL_PATH = "/kaggle/working/1"
AUDIO_ROOT = "/kaggle/input"            # 更換資料夾只改這裡
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")

# [1.2] 轉錄與分段
SEGMENT_DURATION = 30.0                 # 一行最多覆蓋的「牆鐘時間」秒數
BATCH_SIZE = 24
MAX_CONCURRENCY_PER_GPU = 2

# [1.3] 文本清理與初始提示
REPLACEMENTS: Dict[str, str] = {"課題": "客體", "緝罪": "既遂"}
INITIAL_PROMPT = "法律"

# ================================================================
# 2. 公用小工具
# ================================================================
# [2.1] 收集音檔清單
def collect_audio_files(root: str, exts=AUDIO_EXTS) -> List[str]:
    exts_lower = {e.lower() for e in exts}
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts_lower:
                files.append(os.path.join(dirpath, fn))
    return sorted(files)

# [2.2] 秒 → 時:分:秒
def to_timestamp(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# [2.3] 格式化一行輸出
def fmt_chunk(start: float, end: float, txt: str) -> str:
    return f"{to_timestamp(start)}-{to_timestamp(end)} {txt.strip()}\n"

# [2.4] 文本清理（關鍵詞替換 + 去前綴符號）
_pattern = re.compile("|".join(re.escape(k) for k in REPLACEMENTS)) if REPLACEMENTS else None
def clean_text(txt: str) -> str:
    txt = txt.lstrip("! ").strip()
    if _pattern:
        txt = _pattern.sub(lambda m: REPLACEMENTS[m.group(0)], txt)
    return txt

# [2.5] 段落輸出（修正版）
def process_segments(segments, outfile: str, max_len=SEGMENT_DURATION):
    """
    [2.5] 段落輸出（修正版）
    目的：修正首段起點硬從 0:00 算的問題；改成「以第一個 seg.start 為起點」。
         這樣不會在音檔前置靜音時，把時間區間算得過長。
    規則：
      (1) 一行最多涵蓋 max_len 秒（用牆鐘時間：最後一個 seg.end - 該行第一個 seg.start）。
      (2) 換行後，下一行的起點 = 下一個實際 seg.start（不壓縮中間靜音）。
    註：若你偏好「壓縮靜音」（下一行起點改接前一行的 last_end），
        可把 `chunk_start = None` 的邏輯，改為 `chunk_start = last_end`。
    """
    buf_lines = []
    chunk_start = None
    chunk_text_parts = []
    last_end = None

    for seg in segments:
        if chunk_start is None:
            chunk_start = float(getattr(seg, "start", 0.0))
            chunk_text_parts = []

        chunk_text_parts.append(clean_text(seg.text))
        last_end = float(getattr(seg, "end", chunk_start))

        if (last_end - chunk_start) >= max_len:
            line_txt = " ".join(chunk_text_parts).strip()
            line = fmt_chunk(chunk_start, last_end, line_txt)
            print(line, end="", flush=True)
            buf_lines.append(line)

            chunk_start = None
            chunk_text_parts = []
            last_end = None

    if chunk_text_parts:
        end_time = last_end if last_end is not None else chunk_start
        line_txt = " ".join(chunk_text_parts).strip()
        line = fmt_chunk(chunk_start, end_time, line_txt)
        print(line, end="", flush=True)
        buf_lines.append(line)

    with open(outfile, "w", encoding="utf-8") as fh:
        fh.write("".join(buf_lines))
    print(f" ✔ 已寫入 {outfile}")

# ================================================================
# 3. 子進程執行邏輯（每張 GPU 一支）
# ================================================================
def gpu_worker(gpu_idx: int,
               jobs: List[Tuple[str, str]],
               semaphore_size: int,
               result_queue: mp.Queue):
    """
    [3.1] 子進程只看得見自己的 GPU（透過 CUDA_VISIBLE_DEVICES）
    [3.2] 載入 WhisperModel → 建 Pipeline（嚴禁在父進程先初始化 CUDA）
    [3.3] 用 ThreadPoolExecutor + Semaphore 控單卡併發
    [3.4] 把每個 job 的結果（成功/失敗）寫進 multiprocessing.Queue
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GPU{gpu_idx}] 開始初始化模型 ...")
    tic = time.time()

    model = WhisperModel(MODEL_PATH,
                         device=dev,
                         device_index=0,
                         compute_type="float16")
    pipeline = BatchedInferencePipeline(model=model)
    print(f"[GPU{gpu_idx}] 模型初始化完成，用時 {time.time()-tic:.1f}s")

    semaphore = threading.Semaphore(semaphore_size)

    def transcribe_one(in_path: str, out_path: str):
        with semaphore:
            try:
                segments, _ = pipeline.transcribe(
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
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 250,
                        "speech_pad_ms": 600
                    },
                    log_progress=True,
                )
                process_segments(segments, out_path)
                result_queue.put((in_path, "✓"))
            except Exception as exc:
                print(f"[GPU{gpu_idx}] ✘ 轉錄失敗: {in_path} ({exc})")
                result_queue.put((in_path, "✘"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=semaphore_size) as pool:
        pool.map(lambda pair: transcribe_one(*pair), jobs)

    result_queue.put(("__DONE__", gpu_idx))

# ================================================================
# 4. 主流程
# ================================================================
def main():
    audio_files = collect_audio_files(AUDIO_ROOT)
    if not audio_files:
        raise RuntimeError(f"找不到任何音檔於 {AUDIO_ROOT}")
    print(f"共找到 {len(audio_files)} 個音檔")

    gpu_count = min(torch.cuda.device_count() or 1, 4)
    print(f"偵測到 GPU 數量：{gpu_count}")
    jobs_by_gpu: Dict[int, List[Tuple[str, str]]] = {i: [] for i in range(gpu_count)}
    for idx, path in enumerate(audio_files):
        base = os.path.splitext(os.path.basename(path))[0]
        jobs_by_gpu[idx % gpu_count].append((path, f"{base}.txt"))

    ctx = mp.get_context("fork")
    result_queue = ctx.Queue()
    processes = []
    for gpu_idx in range(gpu_count):
        p = ctx.Process(target=gpu_worker,
                        args=(gpu_idx,
                              jobs_by_gpu[gpu_idx],
                              MAX_CONCURRENCY_PER_GPU,
                              result_queue))
        p.start()
        processes.append(p)

    finished_gpu = set()
    while len(finished_gpu) < gpu_count:
        item = result_queue.get()
        if item[0] == "__DONE__":
            finished_gpu.add(item[1])
            continue
        in_path, status = item
        print(f"[主控] {status} {in_path}")

    for p in processes:
        p.join()
    print("🎉 所有轉錄任務完成！")

# ================================================================
# 5. Entry point
# ================================================================
if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"總耗時：{time.time() - t0:.1f} 秒")
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

### A lifesaver for long transcriptions for media professionals, political staff, and hardworking interns.
### This project provides an efficient, user-friendly solution designed to overcome the inefficiencies of manual **audio transcription**.

### Why Not Use Colab?
1.  **GPU Limitations and Instability:** While mounting Google Drive can simplify usage, the core bottleneck for transcription tasks remains GPU availability and stability, which can be inconsistent on Colab.
2.  **Leveraging Free Resources:** This project utilizes the two free T4 GPUs provided by the Kaggle platform, significantly boosting transcription efficiency for various tasks. **[Update]** **The new code automatically detects the number of available GPUs and distributes tasks accordingly.**
3.  **Kaggle's Advantage:** Kaggle is a free machine learning platform offering 30 hours of free GPU time per week. Compared to the unpredictable nature of Colab, this is generally sufficient for routine transcription work.

### Having moved on from political work, I'm unsure who might find this project (as people in such roles rarely browse GitHub). However, driven by a desire to help successors avoid pitfalls from my previous job, I've refined this codebase to a near-stable state. I might consult GPT for further optimization possibilities~ **[Update]** **The new code integrates Batch Processing and finer-grained parallel control for higher efficiency.**

### While faster demos exist online (e.g., WhisperJAX, Whisper WebGPU, even Groq API), they often lack support for custom models or large audio files (typically struggling with files over 25MB, roughly equivalent to a 30-minute low-bitrate MP3).

### ☆ A gift to those destined to find it ~ The entire setup is currently free ~
### ~~☆ Added: **FW1.2.0 with Batched pipeline full code (2024/11/22)~~ **[Update]** **The current code integrates `BatchedInferencePipeline` and offers a more complete automated processing workflow (2025/04/22).**

## Table of Contents

- [Features](#features)
- [Technical Highlights](#technical-highlights)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Example Code](#example-code)
- [Contributing](#contributing)
- [License](#license)
- [Third-Party Licenses](#third-party-licenses)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Features

1.  **[Update]** **Automated Multi-GPU Parallel Processing**
    *   **Automatic GPU Detection and Assignment**: Automatically detects the number of available T4 GPUs in the Kaggle environment (supports up to 4) and assigns audio files to different GPUs using a Round-Robin strategy.
    *   **Batch Inference**: Utilizes `BatchedInferencePipeline` to process multiple transcription requests in batches, significantly improving GPU utilization and overall throughput.
    *   **Per-GPU Concurrency Control**: Allows setting the maximum number of concurrent tasks per GPU (`MAX_CONCURRENCY_PER_GPU`) to prevent overloading individual GPUs and ensure stable operation.
    *   **Transcript Merging and Continuous Timestamps**: Merges results from multiple transcriptions into a single file, automatically adjusting timestamps to ensure a continuous timeline.

2.  **Model Localization and Loading Optimization**
    *   **Pre-downloading Models and Local Loading**: Download `faster-whisper` compatible models beforehand and upload them to Kaggle Datasets or Models. This avoids re-downloading during each run, speeding up initialization.
    *   **[New]** **Independent Model Loading per GPU**: Loads a separate model instance for each detected GPU, ensuring smooth parallel processing.

3.  **Transcript Segmentation and Printing Optimization**
    *   **Fixed Time Interval Segmentation**: Segments the transcript based on a configurable `SEGMENT_DURATION` (default: 30 seconds) interval, enhancing context readability and facilitating subsequent proofreading.
    *   **Real-time Progress Printing**: Prints timestamped text segments to the console during transcription for real-time progress monitoring. **[New]**
    *   **First Segment Start Fix**: The first line now begins at the first audio segment's start time, avoiding inflated timestamps when there is leading silence.
    *   **Post-correction with Gemini**: Leverage Google AI Studio's free Gemini 1.5 Flash/Pro models for high-accuracy proofreading. Multiple tests have verified that correcting chunks of approximately 6500 tokens balances stable quality and efficiency.

4.  **[Update]** **Usability and Error Handling**
    *   **Automatic Audio File Scanning**: Automatically scans recursively for all audio files (supports `.wav`, `.flac`, `.mp3`, `.ogg`, etc.) within the specified root directory (`AUDIO_ROOT`), eliminating the need to manually specify each file.
    *   **Automatic Output Naming**: Automatically generates output filenames like `01.txt`, `02.txt`, etc., based on the order of scanned audio files.
    *   **Centralized Parameter Management**: Consolidates frequently used parameters (model path, audio root directory, batch size, concurrency level, replacements, etc.) at the beginning of the code for easy modification.
    *   **Error Handling for Individual Files**: Implements error catching for the transcription process of single files, preventing the entire program from crashing due to one failed file.

5.  **Timestamp Position and Format Adjustment**
    *   **Timestamp at the Beginning of Segment**: Standardizes the timestamp format (`HH:MM:SS-HH:MM:SS`) and places it at the very beginning of each text segment.

## Technical Highlights

- **[Update]** **Efficient Batch Inference**: Leverages `BatchedInferencePipeline` for high-throughput batch processing, maximizing GPU utilization.
- **[Update]** **Automated Multi-GPU Management**: Automatically detects and utilizes all available GPUs (up to 4), intelligently distributing tasks.
- **[Update]** **Fine-grained Concurrency Control**: Uses `threading.Semaphore` to manage the number of concurrent tasks per GPU, balancing efficiency and stability.
- **[Update]** **Automated File Handling**: Automatically scans audio files and names output files, simplifying the workflow.
- **Efficient Multi-threading**: Employs `concurrent.futures.ThreadPoolExecutor` for effective multi-threaded scheduling. ~~Testing showed multi-processing was less efficient than multi-threading (2024/12/11, FW=1.1.0).~~ **[Note]** **The new code still uses multi-threading, combined with Semaphores for concurrency control.**
- **Optimized Model Loading**: Uploading `faster-whisper` compatible models to Kaggle avoids repeated downloads, reducing the preparation time before transcription tasks.
- **Flexible Text Segmentation**: Supports segmentation by a fixed time interval (`SEGMENT_DURATION`), improving the readability of the transcribed text.
- **Automated Transcript Merging**: Automatically adjusts and continues timestamps, ensuring a coherent timeline across merged transcription files.

## Installation and Setup

### Environment Requirements
- **Python Version**: 3.8+
- **Platform**: Kaggle (Kaggle typically pre-configures the necessary environment)
- **Hardware**: Two T4 GPUs (Provided for free by Kaggle)

### Running Steps

#### 1. Install Necessary Python Packages

In a new Kaggle Notebook, execute the following command to install the `faster-whisper` package (takes about 20 seconds):
#### 2024/10/26 **Update**: The latest version of `ctranslate2` seems to have CUDA compatibility issues. Currently handled by reverting to an older version.
#### 2024/12/11 **Update**: ~~Issue seems resolved in FW1.1.0, `ctranslate2==4.4.0` might be removable.~~
#### 2025/04/22 **Update**: ~~Under FW1.2.0, due to platform environment dependency versions, `ctranslate2==4.4.0` is still required. **[Note]**: The platform environment is currently unstable; reverting to an environment from last year or waiting for updates is recommended.~~
#### 2025/06/30 **Update**: `ctranslate2` has been upgraded and is bundled with `faster-whisper==1.2.0`. The previous compatibility issue has been resolved, so you no longer need to install `ctranslate2==4.4.0` separately.
#### 2025/08/11 **Update**: faster-whisper version updated(1.2.0)

```python
# Install faster-whisper (includes ctranslate2)
!pip install faster-whisper==1.2.0 -q
```

#### 2. Upload the Model to Kaggle and Load it in the Notebook

Next, download the `faster-whisper` model from Hugging Face and upload it to Kaggle. Follow these steps:

1.  Go to the Hugging Face model page, for example: [faster-whisper-large-v2-zh-TW](https://huggingface.co/XA9/faster-whisper-large-v2-zh-TW)
2.  Click the **Files and versions** button on the page and download all model files to your local computer.
3.  Log in to Kaggle, click **Create** in the header, and select **Create New Model**.
4.  When creating the new model, give it a name, e.g., `faster-whisper-large-v2-zh-TW`.
5.  Upload all the downloaded files into this model's directory.
6.  Once uploaded, click the **Create** button to finish uploading the model to Kaggle.
7.  When using the project for the first time, click "Add input" under the "Input" section on the right panel. Select "Models" and then "Your Work" to find the model you just uploaded. Subsequently, starting the notebook will automatically load it (Applause!).

Note: Models fine-tuned from large-v3 might offer 10-20% faster transcription speeds than large-v2 based models. However, models fine-tuned from large-v2 generally adapt better to Chinese. The example model provided is one of the best in terms of accuracy.

#### 3. [Update] Upload Audio Files to Kaggle Dataset

1.  Optionally, split your audio files into manageable parts using tools like ffmpeg or CapCut (剪映). This is useful if you have very long recordings, although the current script handles full files well. The original text suggested splitting in half, likely for the older 2-GPU manual setup, but with auto-distribution, this might be less critical unless files are extremely large or you want finer control.
2.  Similar to uploading the model, create a new Kaggle **Dataset**. You can upload multiple audio files into the same dataset.
3.  When using the notebook, click "Add input" and import the dataset containing your audio files, just like you did for the model.
4.  The code will automatically detect the audio files within the specified input path; no manual path specification per file is needed.

## Usage

In a new Kaggle notebook, execute the following code blocks step-by-step. It's designed to be straightforward!

## Example Code
(Just copy and paste the code blocks. Only modify where indicated.)

### Transcribe Audio Files
1.  **Important:** Set `MODEL_PATH = "copy the path of your input whisper model here (just click COPY path)"`.
2.  Other parameters have been validated through over 300 hours of actual transcription tasks and generally don't need adjustment.
3.  The `replacements` dictionary (find using Ctrl+F) is useful for consistent tasks (e.g., transcribing the same speaker multiple times) by automatically correcting common misrecognitions. Format is `"incorrect word": "correct word",`.
4.  Based on internal testing (using the example model), transcribing a 3-hour audio file (WAV format, 256K typically around 300MB) takes approximately 3 to 4 minutes, with an average accuracy above 90%. Post-correction with Gemini can increase accuracy to 99% – no exaggeration.
5.  It's recommended to record audio in WAV or FLAC format by default, as their precision is indeed superior to MP3. WAV files at 192kbps or 256kbps seem to hit the sweet spot for quality influencing accuracy; higher bitrates yield diminishing returns.

```python
%%time
from faster_whisper import WhisperModel, BatchedInferencePipeline
import datetime, time, os, re, torch, glob
from typing import List, Tuple, Dict
import concurrent.futures, threading, multiprocessing as mp

%%time
# ================================================================
# 1. Adjustable Parameters
# ================================================================
MODEL_PATH = "/kaggle/working/1"
AUDIO_ROOT = "/kaggle/input"            # Change here to switch audio source
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")

# [1.2] Transcription & segmentation
SEGMENT_DURATION = 30.0                 # Max wall-clock seconds per line
BATCH_SIZE = 24
MAX_CONCURRENCY_PER_GPU = 2

# [1.3] Text cleanup & initial prompt
REPLACEMENTS: Dict[str, str] = {"課題": "客體", "緝罪": "既遂"}
INITIAL_PROMPT = "法律"

# ================================================================
# 2. Utility Functions
# ================================================================
# [2.1] Gather audio file list

def collect_audio_files(root: str, exts=AUDIO_EXTS) -> List[str]:
    exts_lower = {e.lower() for e in exts}
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts_lower:
                files.append(os.path.join(dirpath, fn))
    return sorted(files)

# [2.2] Seconds -> HH:MM:SS

def to_timestamp(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# [2.3] Format a line

def fmt_chunk(start: float, end: float, txt: str) -> str:
    return f"{to_timestamp(start)}-{to_timestamp(end)} {txt.strip()}\n"

# [2.4] Text cleanup (keyword replacement + trim prefixes)

_pattern = re.compile("|".join(re.escape(k) for k in REPLACEMENTS)) if REPLACEMENTS else None

def clean_text(txt: str) -> str:
    txt = txt.lstrip("! ").strip()
    if _pattern:
        txt = _pattern.sub(lambda m: REPLACEMENTS[m.group(0)], txt)
    return txt

# [2.5] Segment output (revised)

def process_segments(segments, outfile: str, max_len=SEGMENT_DURATION):
    """
    [2.5] Segment output (revised)
    Purpose: fix the issue of always starting at 0:00 by using the first seg.start instead.
             This avoids inflated timestamps when there is leading silence.
    Rules:
      (1) Each line covers at most max_len seconds of wall-clock time (last seg.end - first seg.start).
      (2) After a line break, the next line starts at the next actual seg.start (no compression of silence).
    Note: If you prefer compressing silence (next line start = last_end),
          change `chunk_start = None` logic to `chunk_start = last_end`.
    """
    buf_lines = []
    chunk_start = None
    chunk_text_parts = []
    last_end = None

    for seg in segments:
        if chunk_start is None:
            chunk_start = float(getattr(seg, "start", 0.0))
            chunk_text_parts = []

        chunk_text_parts.append(clean_text(seg.text))
        last_end = float(getattr(seg, "end", chunk_start))

        if (last_end - chunk_start) >= max_len:
            line_txt = " ".join(chunk_text_parts).strip()
            line = fmt_chunk(chunk_start, last_end, line_txt)
            print(line, end="", flush=True)
            buf_lines.append(line)

            chunk_start = None
            chunk_text_parts = []
            last_end = None

    if chunk_text_parts:
        end_time = last_end if last_end is not None else chunk_start
        line_txt = " ".join(chunk_text_parts).strip()
        line = fmt_chunk(chunk_start, end_time, line_txt)
        print(line, end="", flush=True)
        buf_lines.append(line)

    with open(outfile, "w", encoding="utf-8") as fh:
        fh.write("".join(buf_lines))
    print(f" ✔ Wrote {outfile}")

# ================================================================
# 3. Worker process (one per GPU)
# ================================================================

def gpu_worker(gpu_idx: int,
               jobs: List[Tuple[str, str]],
               semaphore_size: int,
               result_queue: mp.Queue):
    """
    [3.1] Each subprocess sees only its GPU via CUDA_VISIBLE_DEVICES.
    [3.2] Load WhisperModel and build pipeline inside subprocess (avoid CUDA init in parent).
    [3.3] Use ThreadPoolExecutor + Semaphore to control concurrency on a single GPU.
    [3.4] Put each job result (success/fail) into multiprocessing.Queue.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GPU{gpu_idx}] Initializing model ...")
    tic = time.time()

    model = WhisperModel(MODEL_PATH,
                         device=dev,
                         device_index=0,     # subprocess sees a single GPU
                         compute_type="float16")
    pipeline = BatchedInferencePipeline(model=model)
    print(f"[GPU{gpu_idx}] Model ready in {time.time()-tic:.1f}s")

    semaphore = threading.Semaphore(semaphore_size)

    def transcribe_one(in_path: str, out_path: str):
        with semaphore:
            try:
                segments, _ = pipeline.transcribe(
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
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 250,
                        "speech_pad_ms": 600
                    },
                    log_progress=True,
                )
                process_segments(segments, out_path)
                result_queue.put((in_path, "✓"))
            except Exception as exc:
                print(f"[GPU{gpu_idx}] ✘ Transcription failed: {in_path} ({exc})")
                result_queue.put((in_path, "✘"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=semaphore_size) as pool:
        pool.map(lambda pair: transcribe_one(*pair), jobs)

    result_queue.put(("__DONE__", gpu_idx))

# ================================================================
# 4. Main flow
# ================================================================

def main():
    audio_files = collect_audio_files(AUDIO_ROOT)
    if not audio_files:
        raise RuntimeError(f"No audio files found in {AUDIO_ROOT}")
    print(f"Found {len(audio_files)} audio files")

    gpu_count = min(torch.cuda.device_count() or 1, 4)  # Kaggle usually <=2
    print(f"Detected GPU count: {gpu_count}")
    jobs_by_gpu: Dict[int, List[Tuple[str, str]]] = {i: [] for i in range(gpu_count)}
    for idx, path in enumerate(audio_files):
        base = os.path.splitext(os.path.basename(path))[0]
        jobs_by_gpu[idx % gpu_count].append((path, f"{base}.txt"))

    ctx = mp.get_context("fork")
    result_queue = ctx.Queue()
    processes = []
    for gpu_idx in range(gpu_count):
        p = ctx.Process(target=gpu_worker,
                        args=(gpu_idx,
                              jobs_by_gpu[gpu_idx],
                              MAX_CONCURRENCY_PER_GPU,
                              result_queue))
        p.start()
        processes.append(p)

    finished_gpu = set()
    while len(finished_gpu) < gpu_count:
        item = result_queue.get()
        if item[0] == "__DONE__":
            finished_gpu.add(item[1])
            continue
        in_path, status = item
        print(f"[Main] {status} {in_path}")

    for p in processes:
        p.join()
    print("🎉 All transcriptions completed!")

# ================================================================
# 5. Entry point
# ================================================================

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total time: {time.time() - t0:.1f} s")
```


### Merge Transcribed Texts
After running the code above, files like `01.txt`, `02.txt` etc., will be generated according to the naming convention. Use the following code to merge them if needed. Ensure the file list at the bottom (`files_to_merge`) is correct for your needs. The output filename defaults to `merged_output.txt`.

```python
import os
import re
from datetime import timedelta

def parse_timestamp_str(timestamp_str: str) -> float:
    """Parses an HH:MM:SS timestamp string into total seconds."""
    try:
        h, m, s = map(int, timestamp_str.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        print(f"Warning: Could not parse timestamp '{timestamp_str}'. Using 0 seconds.")
        return 0.0

def format_seconds_to_timestamp(seconds: float) -> str:
    """Formats total seconds into an HH:MM:SS timestamp string."""
    # Ensure seconds are non-negative
    seconds = max(0, seconds)
    # Use timedelta for robust handling of time calculations
    td = timedelta(seconds=round(seconds)) # Round to nearest second
    # Format as HH:MM:SS
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def merge_transcriptions(input_files: list, output_file: str):
    """
    Merges multiple transcription files, adjusting timestamps sequentially.

    Args:
        input_files (list): A list of paths to the transcription files to merge,
                            in the desired sequential order.
        output_file (str): The path to save the merged transcription file.
    """
    merged_content = ""
    cumulative_duration = 0.0  # Keep track of the total duration from previous files

    print(f"[*] Starting merge process for files: {input_files}")

    for i, file_path in enumerate(input_files):
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}. Skipping.")
            continue

        print(f"    Processing file {i+1}/{len(input_files)}: {file_path}")
        last_end_time_in_file = 0.0 # Track the end time within the current file

        try:
            with open(file_path, 'r', encoding="utf-8") as f_in:
                for line_num, line in enumerate(f_in):
                    line = line.strip()
                    if not line: continue # Skip empty lines

                    # Regex to capture HH:MM:SS-HH:MM:SS timestamp and the text
                    match = re.match(r"(\d{2}:\d{2}:\d{2})-(\d{2}:\d{2}:\d{2})\s+(.*)", line)
                    if not match:
                        print(f"Warning: Skipping malformed line {line_num+1} in {file_path}: '{line}'")
                        continue

                    start_time_str, end_time_str, text_content = match.groups()

                    # Parse original timestamps to seconds
                    original_start_sec = parse_timestamp_str(start_time_str)
                    original_end_sec = parse_timestamp_str(end_time_str)

                    # Calculate new timestamps by adding the cumulative duration
                    new_start_sec = original_start_sec + cumulative_duration
                    new_end_sec = original_end_sec + cumulative_duration

                    # Format new timestamps back to HH:MM:SS strings
                    new_start_time_fmt = format_seconds_to_timestamp(new_start_sec)
                    new_end_time_fmt = format_seconds_to_timestamp(new_end_sec)

                    # Append the adjusted line to the merged content
                    merged_content += f"{new_start_time_fmt}-{new_end_time_fmt} {text_content}\n"

                    # Update the latest end time encountered in this file
                    last_end_time_in_file = max(last_end_time_in_file, original_end_sec)

        except Exception as e:
            print(f"Error reading or processing file {file_path}: {e}")
            # Decide if you want to stop merging or continue with the next file
            # For robustness, we'll continue here but print the error.
            continue # Skip to the next file

        # After processing a file, add its duration to the cumulative total
        # Use the last original end time found in that file
        print(f"    Finished processing {file_path}. Duration added: {last_end_time_in_file:.2f} seconds.")
        cumulative_duration += last_end_time_in_file

    # Write the final merged content to the output file
    try:
        with open(output_file, 'w', encoding="utf-8") as f_out:
            f_out.write(merged_content)
        print(f"✔ Transcriptions successfully merged and saved to {output_file}")
        print(f"[*] Total merged duration (approx): {format_seconds_to_timestamp(cumulative_duration)}")
    except Exception as e:
        print(f"Error writing merged content to {output_file}: {e}")

# --- Example Usage ---
# List the files you want to merge in the correct order
# Ensure these files exist after running the transcription script
files_to_merge = ["01.txt", "02.txt"] # Add more files if needed, e.g., "03.txt"
output_merge_file = "merged_output.txt"

# Check if the input files exist before attempting merge
existing_files = [f for f in files_to_merge if os.path.exists(f)]
if len(existing_files) != len(files_to_merge):
     print(f"[!] Warning: Not all specified files found. Merging only existing files: {existing_files}")

if existing_files:
    merge_transcriptions(existing_files, output_merge_file)
else:
    print("[!] No input files found for merging. Skipping merge operation.")
```

## Contributing

Contributions of any kind are welcome! If you're unsure how to contribute or need adjustments, feel free to ask GPT – I learned much of this by asking questions iteratively myself!

## License

This project is licensed under the [MIT License](LICENSE). Please see the [LICENSE](LICENSE) file for details.

## Third-Party Licenses

- **faster-whisper**: This project uses the [faster-whisper](https://github.com/SYSTRAN/faster-whisper) library, which is licensed under the MIT License.
- **faster-whisper-large-v2-zh-TW Model**: The example model used in this project comes from [Hugging Face](https://huggingface.co/XA9/faster-whisper-large-v2-zh-TW). Please adhere to its specific license terms.

## Acknowledgements

Special thanks to the following projects and resources for their support:

- The [faster-whisper](https://github.com/SYSTRAN/faster-whisper) development team
- [Hugging Face](https://huggingface.co/XA9/faster-whisper-large-v2-zh-TW) for providing excellent model resources
- [Kaggle](https://www.kaggle.com/) for providing free GPU resources

## Contact

If you have any questions or suggestions, please feel free to reach out:

- **Email**: a0953041880@gmail.com
- **GitHub Issues**: Please use the repository's "Issues" tab for bug reports or feature requests.

Thank you for your interest and support! Cheers~ See you next time.

---

> **Note**: This project makes full use of the free GPU resources provided by Kaggle. Please comply with Kaggle's terms of use, use resources responsibly, and avoid misuse.

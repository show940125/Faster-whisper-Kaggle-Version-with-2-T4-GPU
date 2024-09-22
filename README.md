# Faster-whisper-Kaggle-Version-with-2-T4-GPU-
媒體、政治工作者的長篇逐字稿救星

#### 歡迎來到 **show940125** 的 GitHub 倉庫！本專案提供媒體工作者、政治工作者及苦逼工讀生擺脫不效率的**聽打困境**，特別設計一個高效的音頻轉錄的傻瓜操作方案。
#### 我們將充分利用所有免費資源，包含 Kaggle 提供的兩張免費 T4 GPU 資源，可以大幅提升轉錄效率，助您輕鬆完成各類轉錄任務。
#### kaggle是一個免費的機器學習平台，特點是每個星期提供30個小時的免費GPU，可以充分滿足日常的工作任務。
#### 本人已脫離政治工作，也不知道誰會看到本專案(因為通常這類工作的人根本不會打開github)，幫助後輩少走彎路是我對前一份工作的執念，目前這些代碼集成已經接近穩定，後續是否還有優化空間我再問問GPT~
#### ☆贈與有緣人~反正整套目前都不用花錢~

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

1. **雙 GPU 並行處理與合併功能擴展**
   - **雙 GPU 並行處理**：支持雙 T4 GPU 同時處理兩個音頻文件，顯著提升轉錄效率。
   - **轉錄文本合併與時間戳記接續**：將多個轉錄結果合併為一個文件，並自動接續時間戳記，確保時間軸連續。

2. **模型本地化與加載優化**
   - **模型預先下載與本地加載**：提前下載適用 `faster-whisper` 的模型並上傳至 Kaggle的module，避免每次運行時重新下載，提高預熱速度。

3. **轉錄文本的分段與打印優化**
   - **固定時間間隔分段**：基於固定的 30 秒時間間隔進行文本分段，提升可讀性。
   - **自然分段結合固定分段**：結合自然語句分段與固定時間分段，保留語句連貫性與段落可讀性。

4. **重複語句的問題解決**
   - **修正段落累積與清空邏輯**：避免重複語句的出現，確保文本整潔。

5. **時間戳記位置與格式調整**
   - **時間戳記置於段落開頭**：統一時間戳記格式，確保每個段落的時間戳記位於文本最前端。

## 技術特色

- **雙 GPU 並行處理**：充分利用 Kaggle 提供的兩張 T4 GPU 資源，提升轉錄效率100%。
- **高效的多執行緒處理**：使用 `concurrent.futures.ThreadPoolExecutor` 實現多執行緒處理，最大化 GPU 資源利用。
- **優化的模型加載**：將 `faster-whisper` 適用的模型kaggle化，避免每次重複下載，使得轉錄任務前的準備時間可以壓縮至1分30秒以內。
- **靈活的文本分段方式**：支持固定時間間隔與自然語句結合的分段方式，提升轉錄文本的可讀性。
- **自動化的轉錄文本合併**：自動接續時間戳記，確保多個轉錄文件的時間軸連續。

## 安裝與設定

### 環境需求
- **Python 版本**: 3.8+
- **平台**: Kaggle（Kaggle 已經為你預先配置了所需環境）
- **硬體**: 兩張 T4 GPU（由 Kaggle 提供免費資源）

### 運行步驟

#### 1. 安裝必要的 Python 套件

在新開的 Kaggle Notebook 中，執行以下命令來安裝 `faster-whisper` 套件：(約20秒)

```python
%%time
pip install faster-whisper==1.0.3
```

#### 2. 上傳模型至 Kaggle，並在notebook中加載

接下來，將 `faster-whisper` 模型從 Hugging Face 下載，並上傳至 Kaggle。請遵循以下步驟：

1. 前往 Hugging Face 模型頁面：範例[faster-whisper-large-v2-zh-TW](https://huggingface.co/XA9/faster-whisper-large-v2-zh-TW)
2. 點擊頁面中的 **Files and version** 按鈕，將模型的所有檔案下載到本地電腦。
3. 登入 Kaggle，並前往 **Datasets** 頁面，點擊 **Create New Dataset**。
4. 在建立新的 Dataset 時，為它取一個名稱，例如：`faster-whisper-large-v2-zh-TW`。
5. 將下載好的所有檔案上傳到該 Dataset 資料夾中。
6. 上傳完成後，點擊 **Create** 按鈕，即可完成模型上傳至 Kaggle。
7. 首次使用項目時，點擊右邊"input"下的"add input"，選"models"以及"your work"正確找到你上傳好的model，之後再啟動notebokk就會自動加載(拍手~)

備註：largeV3模型微調的model在轉錄速度上會比largeV2的快10%~20%，但精度上，基於largeV2微調的模型對於中文的適應性較佳，範例提供的版本已經是精度最佳的版本之一。

#### 3. 上傳音檔至 Kaggle，並在notebook中加載

1. 從你的裝置將欲轉錄的音檔切一半，你可以用ffempeg或是用剪映都可
2. 跟上傳模型步驟類似，只是要選擇dataset，你可以在同一個dataset中上傳複數個音檔
3. 使用notebook的時候"add input"將音檔匯入notebook中，方法跟model一樣
4. **注意在之後的代碼中複製(點copy)音檔路徑到指定位置**

## 使用方法

在新開的kaggle notebook中，根據以下代碼逐步執行，等於傻瓜操作喔~

## 範例代碼
(請按代碼塊複製貼上就行，讓你改你再改)

### 轉錄音頻文件
1. 注意MODEL_PATH = "複製你input的whisper model的路徑(直接點COPY就好)"
2. 其他參數已經經過超過300小時以上的轉錄任務實際驗證，不太需要調整
3. replacements(用Ctrl+f查找)部分能提供固定轉錄任務(比如多次轉錄同一個講者的音檔)較佳的體驗，把固定的錯漏字直接替換成正確的文字，格式為"錯字": "正確字",
4. 音頻路徑設定在def main():向下的兩個路徑當中，前面的在上，後面在下。轉錄完畢的兩個TXT檔預設為04&05，由於需配合後續合併，不建議更名，不然合併時也要同步改名~
5. 本專案(按範例模型)實測3小時音頻文件(WAV檔，同常大小約300MB)的轉錄任務約需9分鐘轉錄時間，準確度平均在95%以上，透過替代規則，準確率可達99%，不唬爛。

```python
from faster_whisper import WhisperModel
import datetime
import os
import logging

# 配置 Logging
logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

import concurrent.futures
from typing import List, Tuple

# 設定常量
MODEL_PATH = "/kaggle/input/faster-whisper-large-v2-zh-tw/transformers/default/1"
SEGMENT_DURATION = 30.0
MAX_WORKERS = 2

def transcribe_audio(input_file: str, output_file: str, device_index: int, model, segment_duration: float = SEGMENT_DURATION) -> None:
    # 使用傳入的模型進行轉錄
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
    # 處理轉錄的段落
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
    
    # 處理最後一個段落
    if current_segment_text:
        formatted_segment = format_segment(current_segment_start, end_time, current_segment_text)
        print(formatted_segment)
        txt_content += formatted_segment
    
    # 將結果寫入文件
    with open(output_file, 'w', encoding="utf-8") as txt_file:
        txt_file.write(txt_content)
    
    print(f"已保存: {os.path.abspath(output_file)}")

def format_segment(start_time: float, end_time: float, text: str) -> str:
    # 格式化單個段落
    start_time_str = format_to_custom_timestamp(start_time)
    end_time_str = format_to_custom_timestamp(end_time)
    return f"{start_time_str}-{end_time_str} {text.strip()}\n"

def replace_special_chars(text: str) -> str:
    # 替換特殊字符和糾正常見錯誤
    if text.startswith(("! ", " ")):
        text = text.lstrip("! ")
    
    replacements = {
        "XX": "OO", 
    }
    
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    
    return text

def format_to_custom_timestamp(seconds: float) -> str:
    # 將秒數轉換為自定義時間戳格式
    dt = datetime.datetime(1, 1, 1) + datetime.timedelta(seconds=seconds)
    return f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"

def main():
    # 定義多個音頻文件
    files: List[Tuple[str, str, int]] = [
        ("/kaggle/input/ccl07-08/CLL08_part1.wav", "04.txt", 0),
        ("/kaggle/input/ccl07-08/CLL08_part2.wav", "05.txt", 1)
    ]
    
    # 預加載每個設備的模型
    device_indices = set([device_index for _, _, device_index in files])
    models = {}
    for device_index in device_indices:
        models[device_index] = WhisperModel(
            MODEL_PATH, device="cuda", device_index=device_index, compute_type="float16"
        )

    # 使用線程池並行處理音頻文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 這裡傳入模型
        futures = [executor.submit(transcribe_audio, input_file, output_file, device_index, models[device_index]) 
                   for input_file, output_file, device_index in files]

        for future in concurrent.futures.as_completed(futures):
            future.result()

    print("所有轉錄任務已完成。")

if __name__ == "__main__":
    main()
```

### 合併轉錄文本
直接複製貼上執行，以上轉錄好的兩個文檔就會直接合併囉~(檔名預設為merged_output.txt)不爽自己改~
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

# 假設已經成功生成了 04.txt 和 05.txt
merge_transcriptions("04.txt", "05.txt", "merged_output.txt")
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

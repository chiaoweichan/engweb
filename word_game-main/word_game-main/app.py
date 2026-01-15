import os
import json
import requests
from flask import Flask, render_template, request, jsonify

# 初始化 Flask 應用
app = Flask(__name__)

# --- Gemini API 配置 (Configuration) ---

# 從環境變數中獲取 API Key
API_KEY = os.getenv("GEMINI_API_KEY") 

# 實際用於 API 呼叫的模型名稱
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-09-2025" 

# API 基礎 URL 和完整的 generateContent 端點 URL
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
GEMINI_API_URL = f"{GEMINI_API_BASE}{GEMINI_MODEL_NAME}:generateContent"

# 本地遊戲數據文件路徑
EASY_MODE_JSON_PATH = "static/data/easy_mode.json"

def call_gemini_api(prompt: str, system_instruction: str) -> str:
    """
    呼叫 Gemini API，接受不同的系統指令 (system_instruction) 並返回生成的文字回饋。
    """
    # 檢查 API Key 是否存在，這是最常見的失敗原因
    if not API_KEY:
        print("致命錯誤：GEMINI_API_KEY 環境變數未設定。")
        return "回饋失敗：AI 服務未配置 (API Key 缺失)。"

    headers = {
        "Content-Type": "application/json",
    }
    
    # 建構 API 請求的 payload
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        # 🎯 更改：從參數接收系統指令，以適應不同的 AI 任務
        "systemInstruction": {
            "parts": [{ "text": system_instruction }],
        },
        "generationConfig": {
            # 保留 temperature = 0.5 以確保提示的穩定性和精確度。
            "temperature": 0.5
        }
    }

    try:
        # 向 Gemini API 發出 POST 請求
        response = requests.post(
            f"{GEMINI_API_URL}?key={API_KEY}", 
            headers=headers, 
            json=payload,
            # 設置較短的超時時間防止請求阻塞
            timeout=10 
        )
        response.raise_for_status() # 對於非 2xx 的狀態碼拋出異常

        result = response.json()
        
        # 從回應結構中解析生成的文本
        candidate = result.get('candidates', [{}])[0]
        generated_text = candidate.get('content', {}).get('parts', [{}])[0].get('text')
        
        if generated_text:
            return generated_text.strip() # 移除 AI 生成文字可能帶有的前後空白
        else:
            # AI 成功回應，但內容不符合預期（例如被安全過濾器攔截）
            print(f"Gemini API 返回結果結構異常或內容為空。Reason: {candidate.get('finishReason')}")
            return "回饋失敗：AI 服務暫時無法提供內容。"
            
    except requests.exceptions.HTTPError as e:
        # 處理 4xx/5xx 錯誤
        print(f"呼叫 Gemini API 發生 HTTP 錯誤 {e.response.status_code}: {e.response.text}")
        return f"回饋失敗：API 服務錯誤 (代碼: {e.response.status_code})。請檢查 API Key 是否有效或是否有使用限制。"
    except requests.exceptions.RequestException as e:
        # 處理連線錯誤或超時
        print(f"呼叫 Gemini API 發生連線或超時錯誤: {e}")
        return f"回饋失敗：網路連線錯誤或超時。"
    except Exception as e:
        # 處理其他內部錯誤
        print(f"處理 AI 回應時發生意外錯誤: {e}")
        return "回饋失敗：內部處理錯誤。"

# --- AI 輔助功能 ---

def generate_word_hints_text(level_info, missing_words, incorrect_words, user_sentence, correct_words, sentence_prompt) -> str:
    """
    🎯 情況二：單字未完全正確時的回饋。
    生成單字提示的回饋文字，並整合句子評論，以滿足固定格式要求：
    您造的句子：... \n\n 單字提示 \n\n 句子評論
    """
    
    # 1. 準備單字提示的參數
    hints_system_instruction = (
        "你是一位親切且專業的英文老師，正在為圖片單字解謎遊戲提供輔助提示。你的任務是根據學生錯過的單字，"
        "提供簡短、精確的中文描述提示，幫助他們推理出正確答案。請以鼓勵和友善的語氣回覆。單字的提示必須符合圖片的意境。"
    )

    tips = level_info.get('tip', [])
    correct_answers = level_info.get('answer', [])
    
    missing_words_prompts = []
    for i, word in enumerate(correct_answers):
        if word in missing_words:
            tip_type = tips[i] if i < len(tips) else '物件'
            missing_words_prompts.append(f"單字: {word} (類別: {tip_type})")

    # 1.1. 建構給 Gemini 模型的單字提示詳細提示
    prompt_hints = (
        "遊戲情境：學生正在玩圖片解謎遊戲，需要根據圖片內容猜出單字。圖片中還有學生猜錯的單字： "
        f"{', '.join(incorrect_words) if incorrect_words else '無' }。你的任務是提供輔助。 "
        "請針對以下『遺漏的正確單字』提供**簡短的中文描述提示**，幫助他猜出正確答案。 "
        "請勿透露單字本身。回覆內容必須是純文字，不需要標題，不需要額外的教學或解釋，僅提供提示內容。 "
        "每個單字的提示**不超過 30 個中文字**。如果有多個單字，請務必使用**中文分號「；」**連接所有提示。 "
        f"需要提示的遺漏單字列表：{', '.join(missing_words_prompts)}。"
    )

    ai_hints = call_gemini_api(prompt_hints, hints_system_instruction)
    
    # 2. 進行句子評論的 AI 呼叫 (即使單字未猜對，仍給予造句修正和建議)
    critique_system_instruction = (
        "你是一位嚴謹的英文老師。你的任務是分析學生的英文造句，根據句型要求和應使用的單字，"
        "提供具體的修正建議。回饋必須親切、鼓勵，並且以中文書寫。"
    )

    critique_prompt = (
        "請分析以下學生造的英文句子：\n\n"
        f"**使用者句子 (User Sentence):** 『{user_sentence}』\n"
        f"**本關卡要求的單字 (Required Words, total 3):** {', '.join(correct_words)}\n"
        f"**句型提示 (Sentence Prompt):** 『{sentence_prompt}』\n\n"
        "請根據以下優先順序給予修正與建議 (作為『造句回饋』):\n"
        "1. 句子是否符合句型提示的要求？若不符，請指示修正。\n"
        "2. 句子中是否有明顯的文法或拼寫錯誤？若有，請修正。\n"
        "3. 提醒學生還沒猜對所有單字，鼓勵他們嘗試使用已猜出的單字造句。\n"
        "回覆格式：請直接輸出文法建議和鼓勵，總長度限制在 50 到 100 個中文字之間。"
    )
    ai_critique = call_gemini_api(critique_prompt, critique_system_instruction)

    # 3. 🎯 格式化輸出：使用 \n\n 確保每個區塊間有兩行空行（視覺上為一個空行段落）
    return f"您造的句子是：{user_sentence}\n\n{ai_hints}\n\n{ai_critique}"


def analyze_user_sentence_text(user_sentence: str, correct_words: list, sentence_prompt: str) -> str:
    """
    🎯 情況一：所有單字都猜對時的回饋。
    生成句子分析的回饋文字，要求必須檢查單字使用和句型合規性。
    """
    # 系統指令：充當英文寫作老師
    system_instruction = (
        "你是一位嚴謹的英文寫作與文法老師。你的任務是分析學生的英文造句，提供具體的文法修正和句型使用建議。回饋必須親切、鼓勵，並且以中文書寫。"
    )

    # 建構給 Gemini 模型的詳細提示 (強制檢查單字和句型)
    prompt = (
        "請分析以下學生造的英文句子，進行嚴格的檢查和回饋：\n\n"
        f"1. **使用者句子 (User Sentence):** 『{user_sentence}』\n"
        f"2. **必須使用的三個單字 (Required Words):** {', '.join(correct_words)}\n"
        f"3. **句型提示 (Sentence Prompt):** 『{sentence_prompt}』\n\n"
        "請確認：\n"
        "a) **【強制檢查】** 句子是否完整且準確地使用了所有『必須使用的三個單字』。\n"
        "b) **【強制檢查】** 句子是否完全符合句型提示的要求。\n"
        "c) 句子中是否有任何文法、詞彙或拼寫錯誤。\n\n"
        "🎯 關鍵要求：回覆格式必須以『恭喜你完全答對了！』開頭，接著是文法修正建議和鼓勵。如果句子在單字使用、句型合規和文法上**完全正確**，則給予高度讚揚。總長度限制在 50 到 100 個中文字之間。請直接輸出回饋內容，不包含額外標題。"
    )

    ai_critique = call_gemini_api(prompt, system_instruction)
    
    # 🎯 格式化輸出：您造的句子 + 兩行空行 + 句子分析 (包含恭喜)。
    # 此情況單字已全對，故沒有單字提示部分。
    return f"您造的句子是：{user_sentence}\n\n{ai_critique}"

# --- Flask 路由 (Routes) ---

@app.route("/")
def home():
    """主頁面路由。"""
    return render_template("index.html")

@app.route("/easy")
def easy_mode():
    """簡單模式遊戲頁面路由。"""
    return render_template("easy_mode.html")

@app.route("/hard")
def hard_mode():
    """困難模式遊戲頁面路由。"""
    return render_template("hard_mode.html")

@app.route("/api/ai_feedback", methods=["POST"])
def get_ai_feedback():
    """
    API 端點，接收遊戲狀態、使用者造句和提示，並返回 AI 生成的組合回饋。
    """
    try:
        data = request.get_json()
        current_level = data.get('level', 1)
        missing_words = data.get('missing_words', [])
        incorrect_words = data.get('incorrect_words', [])
        # 🎯 新增接收造句相關資料
        user_sentence = data.get('user_sentence', '').strip()
        sentence_prompt = data.get('sentence_prompt', '').strip()
        correct_words = data.get('correct_words', []) # 使用者選的三個單字
        
        # 1. 從 JSON 載入遊戲數據
        try:
            with open(EASY_MODE_JSON_PATH, "r", encoding="utf-8") as f:
                all_levels_data = json.load(f)
            
            level_info = next((item for item in all_levels_data if item['level'] == current_level), None)
            if not level_info:
                return jsonify({"feedback": f"系統錯誤：找不到關卡 {current_level} 的資料。請檢查 easy_mode.json。"})

        except (FileNotFoundError, json.JSONDecodeError) as e:
            error_msg = "回饋失敗：後端數據文件 (easy_mode.json) 遺失或格式錯誤。"
            return jsonify({"feedback": error_msg})


        # 2. 根據是否猜對所有單字來決定 AI 任務
        if not missing_words:
            # 情況一：所有單字都猜對了 -> 執行句子分析 (analyze_user_sentence_text)
            if not user_sentence:
                 # 雖然單字答對，但如果沒造句，則提醒
                 return jsonify({"feedback": "恭喜你完全答對了！\n\n請先輸入您的英文造句，以便 AI 進行回饋分析。"})

            feedback = analyze_user_sentence_text(user_sentence, correct_words, sentence_prompt)
        else:
            # 情況二：有單字遺漏或猜錯 -> 執行單字提示 and Sentence Critique (generate_word_hints_text)
            # 傳遞所有必要的參數給 generate_word_hints_text
            feedback = generate_word_hints_text(level_info, missing_words, incorrect_words, user_sentence, correct_words, sentence_prompt)
            
        # 3. 返回 AI 老師的最終組合文字
        return jsonify({"feedback": feedback})

    except Exception as e:
        # 捕捉所有未預期的伺服器錯誤
        print(f"伺服器發生意外錯誤: {e}")
        return jsonify({"feedback": "伺服器處理錯誤，請檢查後端控制台。"}), 500

if __name__ == "__main__":
    # 運行應用程式，使用 0.0.0.0 以兼容容器環境
    app.run(debug=True, host='0.0.0.0')
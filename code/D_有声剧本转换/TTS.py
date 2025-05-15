import os
from docx import Document
from aip import AipSpeech
from playsound import playsound  # 播放库

# 百度API信息，替换成你的
APP_ID = '<需替换 https://ai.baidu.com/tech/speech/tts>'
API_KEY = '<需替换 https://ai.baidu.com/tech/speech/tts>'
SECRET_KEY = '<需替换 https://ai.baidu.com/tech/speech/tts>'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

def read_word_paragraphs(file_path):
    doc = Document(file_path)
    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]

def synthesize_to_file(text, filename):
    result = client.synthesis(text, 'zh', 1, {
        'vol': 5,
        'spd': 8,
        'pit': 3,
        'per': 4
    })

    if not isinstance(result, dict):
        with open(filename, 'wb') as f:
            f.write(result)
        print(f"保存音频: {filename}")
        return True
    else:
        print("语音合成失败:", result)
        return False

def main(word_file):
    paragraphs = read_word_paragraphs(word_file)

    # 创建临时目录存音频
    if not os.path.exists('temp_audio'):
        os.mkdir('temp_audio')

    for i, para in enumerate(paragraphs):
        filename = f'temp_audio/para_{i}.mp3'
        success = synthesize_to_file(para, filename)
        if success:
            print(f"正在播放：第{i+1}段")
            playsound(filename)

if __name__ == '__main__':
    main('../C_剧本生成/剧本输出.docx')

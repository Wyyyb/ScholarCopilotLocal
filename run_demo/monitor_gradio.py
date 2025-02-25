import subprocess
import time
import re
import os
from huggingface_hub import HfApi
import tempfile
import psutil

# Hugging Face配置
HF_TOKEN = "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"  # 替换为你的HF token
SPACE_ID = "TIGER-Lab/ScholarCopilot"


def is_process_running(process):
    """检查进程是否还在运行"""
    try:
        return process.poll() is None
    except:
        return False


def get_gradio_url(process):
    """从程序输出中提取gradio URL，同时显示所有输出"""
    url = None
    while True:
        line = process.stdout.readline()
        if not line:
            break

        # 实时打印每一行输出
        print(f"[Gradio Output] {line.strip()}")

        # 匹配gradio URL
        match = re.search(r'Running on public URL: (https://.*?\.gradio\.live)', line)
        if match:
            url = match.group(1)

    return url


def update_html_file(url):
    """更新index.html文件中的URL"""
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scholar Copilot</title>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            var gradioURL = "{url}"; // Your variable URL
            var iframe = document.getElementById("gradioIframe");
            var link = document.getElementById("gradioLink");
            if (iframe) iframe.src = gradioURL;
            if (link) link.href = gradioURL;
        }});
    </script>
</head>
<body>
    <iframe id="gradioIframe" width="100%" height="100%" style="border:none;">
        Your browser does not support iframes. Please click this <a id="gradioLink">url</a>. 
    </iframe>
</body>
</html>'''

    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as temp_file:
        temp_file.write(html_content)
        temp_path = temp_file.name

    return temp_path


def push_to_hf(file_path):
    """推送文件到Hugging Face space"""
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo="index.html",
        repo_id=SPACE_ID,
        repo_type="space"
    )


def kill_process_and_children(process):
    """终止进程及其子进程"""
    try:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except:
        pass


def start_gradio_process():
    """启动gradio进程并返回进程对象"""
    return subprocess.Popen(
        ['python', 'scholar_copilot_gradio.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


def main():
    process = None
    url = None

    while True:
        try:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # 如果进程不存在或已经停止，启动新进程
            if process is None or not is_process_running(process):
                print(f"[{current_time}] 启动新的进程...")

                # 如果有旧进程，确保完全终止
                if process is not None:
                    kill_process_and_children(process)

                # 启动新进程
                process = start_gradio_process()
                # 等待1分钟让程序完全启动
                time.sleep(60)

                # 获取新的URL
                url = get_gradio_url(process)
                if url:
                    print(f"[{current_time}] 获取到新的URL: {url}")

                    # 更新HTML文件
                    temp_path = update_html_file(url)

                    # 推送到Hugging Face
                    push_to_hf(temp_path)
                    print(f"[{current_time}] 成功更新HF space")

                    # 删除临时文件
                    os.unlink(temp_path)
            else:
                print(f"[{current_time}] 程序正在运行中...")

            # 每分钟检查一次
            time.sleep(60)

        except Exception as e:
            print(f"[{current_time}] 发生错误: {str(e)}")
            if process is not None:
                kill_process_and_children(process)
                process = None
            time.sleep(60)  # 发生错误时等待1分钟后重试


if __name__ == "__main__":
    main()
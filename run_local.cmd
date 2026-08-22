@echo off
chcp 65001 >nul
rem ===== ebook2audiobook 本地启动器 =====
rem ffmpeg/ffprobe : D:\ai_tools\ffmpeg_shared\bin (BtbN shared build, 含 avcodec-*.dll, 应用 register_dlls 需要)
rem ffmpeg 备用    : DownKyi 自带 6.1.2 (静态版, 无 DLL)
rem calibre       : D:\ai_tools\calibre (msiexec /a 解包的便携版, 提供 ebook-convert)
rem ffmpeg 8.1 shared (BtbN n8.1): CLI + avcodec-62 等 DLL, torchcodec 需要, 应用 register_dlls 也认
set "PATH=D:\ai_tools\ffmpeg81\bin;D:\ai_tools\ffmpeg_shared\bin;D:\Program Files (x86)\DownKyi-1.0.23-1.win-x64\ffmpeg;D:\ai_tools\calibre\PFiles64\Calibre2;%PATH%"
rem cu129 索引缺 torchaudio 2.9.0, 官方矩阵中 cu130 (torch 2.11.0) 才完整
set "DEVICE_TAG=cu130"
rem HF 下载: 走系统代理(127.0.0.1:7897)直连 huggingface.co; 禁用 hf_xet(不走代理会失败)
set "HF_HUB_DISABLE_XET=1"
cd /d "%~dp0"
".venv\Scripts\python.exe" app.py %*
pause

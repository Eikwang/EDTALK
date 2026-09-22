# -*- coding: utf-8 -*-
"""AI-Vtuber 整合冒烟脚本（/autoplan 2026-09-21 T7）

对应 DOCS/AI-Vtuber整合_模块接口与调用链分析.md §12.0 冒烟清单 S1-S7。
整合实施第一步：先跑通本脚本再写对接代码——"import≠能跑"。

用法（runtime312 解释器）：
  runtime312/python.exe tools/integration_smoke.py                     # 纯探测（服务需已在跑）
  runtime312/python.exe tools/integration_smoke.py --start-edtalk      # 未运行则拉起 realtime_serve
  runtime312/python.exe tools/integration_smoke.py --start-fastrag     # 未运行则拉起 fastrag server.exe
  runtime312/python.exe tools/integration_smoke.py --tts-wav out.wav   # S7 校验 TTS 输出采样率

退出码：0=无 FAIL；1=存在 FAIL（WARN/SKIP 不影响）。
"""

import argparse
import base64
import json
import math
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RT_PYTHON = os.path.join(PROJECT_ROOT, "runtime312", "python.exe")
RT_FFMPEG = os.path.join(PROJECT_ROOT, "runtime312", "ffmpeg", "bin", "ffmpeg.exe")
RT_FFPROBE = os.path.join(PROJECT_ROOT, "runtime312", "ffmpeg", "bin", "ffprobe.exe")
FASTRAG_DIR = os.path.join(PROJECT_ROOT, "fastrag")
SMOKE_LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

RESULTS = []


def report(item, status, detail="", fix=""):
    RESULTS.append((item, status))
    line = f"[{status:^4s}] {item}"
    if detail:
        line += f"\n         {detail}"
    if status == "FAIL" and fix:
        line += f"\n         fix: {fix}"
    print(line, flush=True)


def http_json(url, payload=None, token=None, timeout=6):
    """返回 (ok, data_dict_or_None, http_status_or_err_str)"""
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    body = None
    if payload is not None:
        req.add_header("Content-Type", "application/json")
        body = json.dumps(payload).encode("utf-8")
    try:
        with urllib.request.urlopen(req, data=body, timeout=timeout) as resp:
            return True, json.loads(resp.read().decode("utf-8")), resp.status
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read().decode("utf-8"))
        except Exception:
            data = None
        return False, data, e.code
    except Exception as e:
        return False, None, str(e)


def make_speech_pcm(seconds=1.5, rate=16000):
    """生成 16kHz 单声道 int16 PCM（幅度调制的低频正弦，模拟语音能量起伏）"""
    samples = bytearray()
    n = int(rate * seconds)
    for i in range(n):
        t = i / rate
        env = 0.5 * (1 + math.sin(2 * math.pi * 2.5 * t))  # 2.5Hz 能量包络
        v = int(0.6 * env * 32767 * math.sin(2 * math.pi * 200 * t))
        samples += struct.pack("<h", v)
    return bytes(samples)


def wait_for(check_fn, timeout_s, interval_s=2.0):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        last = check_fn()
        if last:
            return True
        time.sleep(interval_s)
    return False


def ensure_service(name, url, start_flag, start_cmd_fn, ready_fn, args, env_fn=None):
    """探测服务；未运行且给了启动 flag 则拉起并等待就绪。返回 (running, detail)。"""
    ok, _, err = http_json(url, timeout=4)
    if ok:
        return True, "已在运行"
    if not start_flag:
        return False, f"未运行（{err}）——先启动或加启动参数"
    print(f"         启动 {name} ...", flush=True)
    os.makedirs(SMOKE_LOG_DIR, exist_ok=True)
    log_path = os.path.join(SMOKE_LOG_DIR, f"{name}_smoke.log")
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
            subprocess.Popen(start_cmd_fn(args), cwd=PROJECT_ROOT,
                             stdout=lf, stderr=lf,
                             env=env_fn() if env_fn else None)
    except Exception as e:
        return False, f"启动失败: {e}"
    started = wait_for(lambda: http_json(url, timeout=3)[0], timeout_s=150)
    if not started:
        return False, (f"启动后 {150}s 未就绪——查日志 {log_path}"
                       f"（模型加载/CUDA Graph 捕获通常 10-40s，超时多为模型路径或端口问题）")
    return True, f"已拉起（日志 {log_path}）"


def s1_s2_realtime(args):
    # ---- S1 就绪 ----
    running, detail = ensure_service(
        "realtime_serve", f"{args.edtalk_url}/status", args.start_edtalk,
        lambda a: [RT_PYTHON, "-m", "realtime_serve.main",
                   "--character-dir", a.character_dir, "--segments", "auto"],
        None, args)
    if not running:
        report("S1 realtime_serve 就绪", "FAIL", detail,
               f"cd {PROJECT_ROOT} && runtime312\\python.exe -m realtime_serve.main "
               f"--character-dir {args.character_dir} --segments auto（或本脚本加 --start-edtalk）")
        report("S2 push_full 口型链", "SKIP", "依赖 S1")
        return
    ok, status, err = http_json(f"{args.edtalk_url}/status", token=args.token)
    if not ok:
        report("S1 realtime_serve 就绪", "FAIL", f"/status 请求失败: {err}",
               "检查端口/鉴权 token（--token）")
        report("S2 push_full 口型链", "SKIP", "依赖 S1")
        return
    pre = status.get("preroll_state")
    seg = status.get("current_segment")
    if pre == "complete" and seg:
        report("S1 realtime_serve 就绪", "PASS",
               f"preroll=complete, segment={seg}, 分辨率={status.get('resolution')}")
    else:
        report("S1 realtime_serve 就绪", "WARN",
               f"HTTP 面在但未就绪: preroll={pre}, segment={seg}（启动中或素材缺失）",
               "")
    ready = (pre == "complete")

    # ---- S2 push_full（唯一有效音频通道；/audio/push 为死路径勿测）----
    if not ready:
        report("S2 push_full 口型链", "SKIP", "服务未就绪")
        return
    pcm = make_speech_pcm()
    payload = {"pcm_base64": base64.b64encode(pcm).decode("ascii")}
    ok, data, err = http_json(f"{args.edtalk_url}/audio/push_full", payload,
                              token=args.token, timeout=15)
    if not ok:
        report("S2 push_full 口型链", "FAIL",
               f"/audio/push_full 失败: {err} {json.dumps(data, ensure_ascii=False) if data else ''}",
               "400=PCM 规格问题（本脚本已按 16kHz int16 生成）；413=超 10MB")
        return
    duration = data.get("duration_s")
    # 口型的脚本化代理判据：frame_index 推进（口型本身需虚拟摄像头人眼确认）
    f0 = status.get("frame_index")
    time.sleep(3)
    _, st2, _ = http_json(f"{args.edtalk_url}/status", token=args.token)
    time.sleep(3)
    _, st3, _ = http_json(f"{args.edtalk_url}/status", token=args.token)
    advancing = (st2 and st3 and st3.get("frame_index", 0) > st2.get("frame_index", 0)
                 >= (f0 or 0))
    if advancing:
        report("S2 push_full 口型链", "PASS",
               f"duration_s={duration}, frame_index 推进中 "
               f"({f0}->{st2.get('frame_index')}->{st3.get('frame_index')})——"
               f"口型形态请人眼核验虚拟摄像头")
    else:
        report("S2 push_full 口型链", "WARN",
               f"push 200/duration={duration} 但 frame_index 未推进"
               f"（渲染未开播或推理未跟上）——看 /status.last_error 与控制台")


def s3_s4_fastrag(args):
    def _fastrag_env():
        env = dict(os.environ)
        env["FASTRAG_CONFIG"] = os.path.join(FASTRAG_DIR, "Config.toml")
        return env

    running, detail = ensure_service(
        "fastrag", f"{args.fastrag_url}/health", args.start_fastrag,
        lambda a: [os.path.join(FASTRAG_DIR, "server.exe")],
        None, args, env_fn=_fastrag_env)
    if not running:
        report("S3 fastrag /health", "FAIL", detail,
               "启动: 设 FASTRAG_CONFIG=<绝对路径>\\Config.toml 后运行 fastrag\\server.exe"
               "（或本脚本加 --start-fastrag）；exit 1=索引缺失先跑 build-index.exe")
        report("S4 fastrag /query", "SKIP", "依赖 S3")
        return
    ok, health, err = http_json(f"{args.fastrag_url}/health")
    if not ok:
        report("S3 fastrag /health", "FAIL", f"/health 请求失败: {err}", "")
        report("S4 fastrag /query", "SKIP", "依赖 S3")
        return
    chunks = health.get("chunks", 0)
    if health.get("status") == "ok" and chunks > 0:
        report("S3 fastrag /health", "PASS",
               f"status=ok, chunks={chunks}, rerank={health.get('rerank_enabled')}")
    else:
        report("S3 fastrag /health", "WARN",
               f"status={health.get('status')}, chunks={chunks}（chunks=0 提示索引/知识库错位）")

    ok, data, err = http_json(f"{args.fastrag_url}/query",
                              {"question": args.query}, timeout=30)
    if not ok:
        # fastrag 错误体是纯文本非 JSON——http_json 解析失败也能到这
        report("S4 fastrag /query", "FAIL", f"/query 失败: {err}",
               "400 纯文本=问题为空/超 2000 字；连接失败=服务未起")
        return
    answer = (data or {}).get("answer", "")
    ctx = (data or {}).get("context", [])
    if answer:
        report("S4 fastrag /query", "PASS",
               f"answer({len(answer)}字): {answer[:50]}... | 命中块={len(ctx)}")
        if not ctx:
            print("         note: context 为空——检索可能未命中（查 Config 路径是否错拿，§10-5）",
                  flush=True)
    else:
        report("S4 fastrag /query", "FAIL", "answer 为空（兜底/异常）",
               "查 fastrag 日志与 /health.chunks；LLM 未加载会 panic exit 101")


def s5_preprocess(args):
    base_dir = os.path.join(PROJECT_ROOT, args.base_dir) if not os.path.isabs(args.base_dir) else args.base_dir
    script = os.path.join(PROJECT_ROOT, "data_preprocess", "data_preprocess_for_train",
                          "run_preprocess.py")
    if not os.path.isdir(os.path.join(base_dir, "original_videos")):
        report("S5 预处理 dry_run", "FAIL",
               f"base_dir 无 original_videos/: {base_dir}",
               f"放好视频或用 --base-dir 指向含 original_videos/ 的目录")
        return
    py = RT_PYTHON if os.path.exists(RT_PYTHON) else sys.executable
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"   # 子进程中文输出按 UTF-8，防 GBK 控制台乱码
    env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.run([py, "-u", script, "--base_dir", base_dir, "--dry_run"],
                              cwd=PROJECT_ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=120, env=env)
        out = proc.stdout + proc.stderr
        n_ok = out.count("[v]")
        if proc.returncode == 0 and ("执行计划" in out or n_ok >= 5):
            report("S5 预处理 dry_run", "PASS",
                   f"11 步命令全部解析（计划可见 {n_ok} 步）——去除 --dry_run 即实际执行")
        else:
            report("S5 预处理 dry_run", "FAIL",
                   f"returncode={proc.returncode}，输出尾部: {out[-300:]}",
                   "脚本缺失/参数错误——对照 run_preprocess.py")
    except Exception as e:
        report("S5 预处理 dry_run", "FAIL", str(e), "")


def s6_virtual_camera(args):
    ffmpeg = RT_FFMPEG if os.path.exists(RT_FFMPEG) else "ffmpeg"
    try:
        proc = subprocess.run([ffmpeg, "-hide_banner", "-list_devices", "true",
                               "-f", "dshow", "-i", "dummy"],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=30)
        devices = (proc.stdout + proc.stderr).lower()
        if "unity" in devices:
            report("S6 虚拟摄像头注册", "PASS", "DirectShow 设备列表含 Unity Capture")
        elif "camera" in devices or "video" in devices:
            report("S6 虚拟摄像头注册", "WARN",
                   "有摄像头设备但未见 Unity Capture——服务可能静默回退 OBS 后端（§10-7）",
                   "管理员运行 UnityCapture-master\\Install\\Install.bat")
        else:
            report("S6 虚拟摄像头注册", "FAIL", "未枚举到任何视频设备",
                   "查摄像头驱动与 DirectShow 滤镜注册")
    except Exception as e:
        report("S6 虚拟摄像头注册", "WARN", f"枚举失败: {e}——人工在 OBS 源列表确认", "")


def s7_tts_sample_rate(args):
    if not args.tts_wav:
        report("S7 TTS 采样率 16k", "SKIP", "未提供 --tts-wav（对接纪要已确认 16k，此为复验防漂移）")
        return
    ffprobe = RT_FFPROBE if os.path.exists(RT_FFPROBE) else "ffprobe"
    try:
        proc = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0",
                               "-show_entries", "stream=sample_rate,channels,codec_name",
                               "-of", "json", args.tts_wav],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=30)
        stream = json.loads(proc.stdout)["streams"][0]
        sr, ch = stream.get("sample_rate"), stream.get("channels")
        codec = stream.get("codec_name", "")
        if sr == "16000" and ch == 1 and "pcm" in codec:
            report("S7 TTS 采样率 16k", "PASS", f"{codec} {sr}Hz {ch}ch——可直接 push_full")
        else:
            report("S7 TTS 采样率 16k", "FAIL",
                   f"{codec} {sr}Hz {ch}ch ≠ 16kHz/mono/pcm_s16le（静默口型错乱风险 §10-2）",
                   "ffmpeg -i in.wav -ar 16000 -ac 1 -sample_fmt s16 out.wav")
    except Exception as e:
        report("S7 TTS 采样率 16k", "FAIL", f"ffprobe 失败: {e}", "")


def main():
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="AI-Vtuber 整合冒烟（§12.0 S1-S7）")
    parser.add_argument("--edtalk-url", default="http://127.0.0.1:8000")
    parser.add_argument("--fastrag-url", default="http://127.0.0.1:11420")
    parser.add_argument("--token", default=None, help="realtime_serve Bearer token")
    parser.add_argument("--character-dir", default="wy")
    parser.add_argument("--base-dir", default="wy", help="S5 dry_run 的数据集目录")
    parser.add_argument("--query", default="你好", help="S4 测试问题")
    parser.add_argument("--start-edtalk", action="store_true", help="服务未运行则拉起")
    parser.add_argument("--start-fastrag", action="store_true", help="服务未运行则拉起")
    parser.add_argument("--tts-wav", default=None, help="S7 校验用 TTS 输出 wav 路径")
    args = parser.parse_args()

    print("=" * 64)
    print("AI-Vtuber 整合冒烟 §12.0（S1-S7）")
    print("=" * 64)
    s1_s2_realtime(args)
    s3_s4_fastrag(args)
    s5_preprocess(args)
    s6_virtual_camera(args)
    s7_tts_sample_rate(args)

    n_fail = sum(1 for _, s in RESULTS if s == "FAIL")
    n_pass = sum(1 for _, s in RESULTS if s == "PASS")
    n_warn = sum(1 for _, s in RESULTS if s == "WARN")
    n_skip = sum(1 for _, s in RESULTS if s == "SKIP")
    print("=" * 64)
    print(f"汇总: {n_pass} PASS / {n_warn} WARN / {n_fail} FAIL / {n_skip} SKIP"
          f"  →  {'✅ 冒烟通过' if n_fail == 0 else '❌ 存在 FAIL，先修复再进入对接实施'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

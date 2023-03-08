from ctypes.wintypes import HACCEL
import numpy as np
import cv2
import time
from PIL import ImageGrab
from PIL import Image
import pyocr
import pyocr.builders
import os
import matplotlib
import matplotlib.pyplot as plt

path_tesseract = "C:\Program Files\Tesseract-OCR"
if path_tesseract not in os.environ["PATH"].split(os.pathsep):
    os.environ["PATH"] += os.pathsep + path_tesseract

tools = pyocr.get_available_tools()
tool = tools[0]

# BGR配列を２値化する関数
def nichika(IMG):
    H_IMG, W_IMG = IMG.shape[:2]
    FLAG = 0
    COLOR = 0

    # COLOR = 1, 緑文字用
    COLOR += 1
    for i in range(H_IMG):
        for j in range(W_IMG):
            if IMG[i][j][0] < 100 and IMG[i][j][1] < 100 and IMG[i][j][2] < 100:
                FLAG += 1
                IMG[i][j][0] = 0
                IMG[i][j][1] = 0
                IMG[i][j][2] = 0

    if FLAG < 10:
        # COLOR = 2, 緑文字用
        COLOR += 1
        for i in range(H_IMG):
            for j in range(W_IMG):
                if IMG[i][j][0] < 150 and IMG[i][j][1] > 100 and IMG[i][j][2] < 100:
                    FLAG += 1
                    IMG[i][j][0] = 0
                    IMG[i][j][1] = 0
                    IMG[i][j][2] = 0

    if FLAG < 10:
        # COLOR = 3, 赤文字用
        COLOR += 1
        for i in range(H_IMG):
            for j in range(W_IMG):
                if IMG[i][j][0] < 100 and IMG[i][j][1] < 90 and IMG[i][j][2] > 220:
                    FLAG += 1
                    IMG[i][j][0] = 0
                    IMG[i][j][1] = 0
                    IMG[i][j][2] = 0

    if FLAG < 10:
        # COLOR = 4, 橙文字用
        COLOR += 1
        for i in range(H_IMG):
            for j in range(W_IMG):
                if IMG[i][j][0] > 0 and IMG[i][j][1] > 100 and IMG[i][j][2] > 215:
                    FLAG += 1
                    IMG[i][j][0] = 0
                    IMG[i][j][1] = 0
                    IMG[i][j][2] = 0

    # 文字以外を白で埋め尽くす
    for i in range(H_IMG):
        for j in range(W_IMG):
            if IMG[i][j][0] != 0 and IMG[i][j][1] != 0 and IMG[i][j][2] != 0:
                IMG[i][j][0] = 255
                IMG[i][j][1] = 255
                IMG[i][j][2] = 255

    return IMG, COLOR

# OCRで画像から文字列を抽出する関数
def OCR(IMG_NAME):
    IMG_OCR = Image.open(IMG_NAME)
    builder = pyocr.builders.TextBuilder()
    result = tool.image_to_string(IMG_OCR, lang="eng", builder=pyocr.builders.LineBoxBuilder(tesseract_layout=1))
    if len(result) == 0:
        return "???"
    else:
        OCR_STR = result[0].content
        return OCR_STR

# 現在のモニター画面から必要なデータを取得する関数
def information_getter(TIME_CHECK, DISTANCE_CHECK):
    ERROR = 0
    left_time = -1
    left_distance = -1
    door_close = 0

    # メインモニターのスクショをBGR配列に変換
    UTC_time = time.time()
    tmp_img = ImageGrab.grab()
    img = np.array(tmp_img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # モニターの画面サイズを取得
    H, W = img.shape[:2]

    # ドア閉めランプの点灯確認
    pixel_SUM = 0
    B_SUM = 0
    G_SUM = 0
    R_SUM = 0
    for i in range(885, 895):
        for j in range(990, 1000):
            pixel_SUM += 1
            B_SUM += img[int(i*H/1080)][int(j*W/1920)][0]
            G_SUM += img[int(i*H/1080)][int(j*W/1920)][1]
            R_SUM += img[int(i*H/1080)][int(j*W/1920)][2]
    B_SUM //= pixel_SUM
    G_SUM //= pixel_SUM
    R_SUM //= pixel_SUM
    if abs(B_SUM-21)+abs(G_SUM-255)+abs(R_SUM-254) < 10:
        door_close = 1

    if TIME_CHECK == 1:
        # 画面から目標停止時刻までの残り時間の表示部分を選択し画像で保存
        img_cut = img[int(72*H/1080) : int(102*H/1080), int(1760*W/1920) : int(1880*W/1920)]
        img_cut, time_color = nichika(img_cut)
        cv2.imwrite("left_time.jpg", img_cut)

        # OCRで得られた残り時間の文字列を数値に変換
        ocr_time = OCR("left_time.jpg")
        left_time_str = ""
        for i in range(len(ocr_time)):
            if ocr_time[i].isdecimal() == True:
                left_time_str += ocr_time[i]
            elif ocr_time[i] == "O":
                left_time_str += "0"
        if len(left_time_str) != 4:
            ERROR = 1
        else:
            left_time = int(left_time_str[:2])*60 + int(left_time_str[2:])
            if time_color == 4:
                left_time *= -1

    if DISTANCE_CHECK == 1:
        # 画面から目標停止位置までの残り距離の表示部分を選択し画像で保存
        img_cut = img[int(146*H/1080) : int(178*H/1080), int(1760*W/1920) : int(1880*W/1920)]
        img_cut, distance_color = nichika(img_cut)
        cv2.imwrite("left_distance.jpg", img_cut)

        # OCRで得られた残り距離の文字列を数値に変換
        ocr_distance = OCR("left_distance.jpg")
        left_distance_str = ""
        num_else_check = 0
        for i in range(len(ocr_distance)):
            if ocr_distance[i].isdecimal() == True:
                left_distance_str += ocr_distance[i]
                if num_else_check != 0:
                    ERROR = 1
            elif ocr_distance[i] == "-" and distance_color != 1 and left_distance_str == "":
                left_distance_str += "-"
                if num_else_check != 0:
                    ERROR = 1
            elif ocr_distance[i] == "=" and distance_color != 1 and left_distance_str == "":
                left_distance_str += "-"
                if num_else_check != 0:
                    ERROR = 1
            elif ocr_distance[i] == "_" and distance_color != 1 and left_distance_str == "":
                left_distance_str += "-"
                if num_else_check != 0:
                    ERROR = 1
            elif ocr_distance[i] == "O":
                left_distance_str += "0"
                if num_else_check != 0:
                    ERROR = 1
            elif ocr_distance[i] == "A":
                left_distance_str += "4"
                if num_else_check != 0:
                    ERROR = 1
            elif ocr_distance[i] == " ":
                pass
            else:
                num_else_check = 1
        if len(left_distance_str) == 0 or left_distance_str == "-":
            ERROR = 1
        else:
            left_distance = int(left_distance_str)
            if distance_color == 2:
                left_distance *= 0.01
    
    # 情報を標準出力として表示
    if ERROR == 0:
        pass
    if ERROR != 0:
        print(OCR("left_time.jpg"), "=>", "???")
        print(OCR("left_distance.jpg"), "=>", "???")
        print(" ")
    
    return ERROR, left_time, left_distance, door_close, UTC_time

#print(information_getter(1,1))

# グラフをリアルタイムで出力するための初期設定
matplotlib.use('TkAgg')
fig, ax = plt.subplots(1, 1)
plt.get_current_fig_manager().window.wm_geometry("+0+0")
#plt.get_current_fig_manager().window.wm_geometry("+0-1600")
plt.axvline(x=0, linestyle = "--", color = "black")
plt.axhline(y=0, linestyle = "--", color = "black")
ax.set_xlabel("Time [s]", fontsize=20)
ax.set_ylabel("Length [m]", fontsize=20)

Tmat = []
Dmat = []
target_time = 0
mode = 0 # 0:停車中, 1:出発可能, 2:加速中, 3:減速中
while True:
    if mode == 0:
        Tmat = []
        Dmat = []
        error, clock_time, now_distance, now_door, utc_time = information_getter(1,0)
        if error == 0:
            target_time = utc_time + clock_time
            if now_door == 1 and target_time > utc_time:
                mode = 1

    elif mode == 1:
        error, clock_time, now_distance, now_door, utc_time = information_getter(0,1)
        if now_door == 0:
            mode = 0
        elif now_door == 1 and now_distance > 100:
            Tmat.append(utc_time-target_time)
            Dmat.append(now_distance)
            mode = 2

    elif mode == 2:
        error, clock_time, now_distance, now_door, utc_time = information_getter(0,1)
        if now_door == 0:
            mode = 0
        elif now_distance > Dmat[-1]:
            error = 1
        elif now_distance < Dmat[-1]*0.5 and now_distance < Dmat[-1]-10:
            error = 1
        elif error == 0:
            Tmat.append(utc_time-target_time)
            Dmat.append(now_distance)

    # 時間と距離のグラフを出力
    if len(Tmat) == 0:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        ax.set_xlim(Tmat[0], max(10, Tmat[-1]+10))
        ax.set_ylim(-100, Dmat[0]+200)

    line1, = ax.plot(Tmat, Dmat, color='deepskyblue')
    if mode == 0:
        text1 = ax.text(0.5, 0.5, "STOPPED\n\nDOORS OPENED", fontsize=20, va='center', ha='center')
    if mode == 1:
        text1 = ax.text(0.5, 0.5, "STOPPED\n\nDOORS CLOSEED", fontsize=20, va='center', ha='center')
    if mode >= 2:
        text1 = ax.text(Tmat[-1], Dmat[-1], str(int(Dmat[-1]))+" m\n  "+str(int(Tmat[-1]))+" s", fontsize=16)
        point1, = ax.plot(Tmat[-1], Dmat[-1],"o", color='deepskyblue')
    plt.pause(0.01)
    line1.remove()
    text1.remove()
    if mode >= 2:
        point1.remove()

# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from tensorflow import keras
import torch
import sys
import dill
import subprocess

app = Flask(__name__)

''' Main page '''
@app.route("/", methods=["GET","POST"])
def index():
    if request.method =='GET':
        return render_template('index.html')

''' Bert4Rec result '''
@app.route("/result_bert4rec", methods=["POST"])
def result_bert4rec():
    if request.method =='POST':

        spec1 = request.form['spec1']
        spec2 = request.form['spec2']
        spec3 = request.form['spec3']
        wish = request.form['wish']

        # Bert4Rec 모델이 결과를 출력합니다.
        configs = {}
        configs["templates"] = "train_bert"
        configs["mode"] = "tobigs_test"
        configs["pretrained_weights"] = "dummy_experiments/play/play20210115_128_final/models/best_model.pth"
        configs["max_len"] = "42"
        configs["hidden_units"] = "128"
        configs["input_middle_seq1"] = spec1
        configs["input_middle_seq2"] = spec2
        configs["input_middle_seq3"] = spec3
        configs["input_middle_num"] = "5"
        configs["input_middle_target"] = wish

        custom_args = ["python", "run.py", "--templates", configs["templates"], "--mode", configs["mode"], "--pretrained_weights", configs["pretrained_weights"], "--max_len", configs["max_len"], "--hidden_units", configs["hidden_units"], "--input_middle_seq", configs["input_middle_seq1"], configs["input_middle_seq2"], configs["input_middle_seq3"], "--input_middle_num", configs["input_middle_num"], "--input_middle_target", configs["input_middle_target"]]

        # commands
        subprocess.run(custom_args)
        
        f = open('pred_middle_name.txt', 'r')
        output1 = f.readline()
        output2 = f.readline()
        output3 = f.readline()
        output4 = f.readline()
        output5 = f.readline()

        f.close()

        # 입력에 따른 출력을 웹 사이트로 보여줍니다.
        return render_template('result_bert4rec.html', recommendation=wish, field1=output1, field2=output2, field3=output3, field4=output4, field5=output5)

''' DeepFM result '''
@app.route("/result_deepfm", methods=["POST"])
def result_deepfm():
    if request.method == 'POST':

        # 모델을 로드합니다.
        sys.path.append('.')
        with open('./src/model/deepfm_model.pkl', 'rb') as f:
            deepfm = dill.load(f)

        # 파라미터를 전달 받습니다.
        career_turn = request.form['career_turn'] # 이직 횟수 1
        career_sum_period = request.form['career_sum_period'] # 총 재직 기간 44.0
        certificate_name = request.form['certificate_name'] # 최근 취득 자격증 OPIC
        school_name = request.form['school_name'] # 학교명 한국외국어대학교
        school_major_name = request.form['school_major_name'] # 전공명 영어영문학부
        school_major_state = request.form['school_major_state'] # [전공, 부전공, 복수전공, 연합전공] 중 하나 > 전공
        school_major_level = request.form['school_major_level'] # ['학사', '석사', '전문학사', '박사', '수료'] 중 하나 > 학사
        school_state = request.form['school_state'] # ['졸업', '재학', '중퇴', '휴학', '교환학생', '수료'] 중 하나 > 졸업
        skill = request.form['skill'] # 본인 능력 , 연결로 입력
        # 언어 능력 ['중상급(업무상 원활한 의사소통)', '고급(자유자재의 의사소통)', '초급', '중급(업무상 의사소통 가능)', '원어민 수준']
        기타 = request.form['기타']
        독일어 = request.form['독일어']
        러시아어 = request.form['러시아어']
        베트남어 = request.form['베트남어']
        에스파냐어 = request.form['에스파냐어']
        영어 = request.form['영어']
        인도네시아어 = request.form['인도네시아어']
        일본어 = request.form['일본어']
        중국어 = request.form['중국어']
        프랑스어 = request.form['프랑스어']
        accum_count = request.form['accum_count'] # 수상 개수 1

        # DeepFM 모델이 결과를 출력합니다.
        output1, output2 = deepfm.test(career_turn=career_turn, career_sum_period=career_sum_period, certificate_name=certificate_name, school_name=school_name, school_major_name=school_major_name, school_major_state=school_major_state, school_major_level=school_major_level, school_state=school_state, skill=skill, 기타=기타, 독일어=독일어, 러시아어=러시아어, 베트남어=베트남어, 에스파냐어=에스파냐어, 영어=영어, 인도네시아어=인도네시아어, 일본어=일본어, 중국어=중국어, 프랑스어=프랑스어, accum_count=accum_count)

        # 입력에 따른 출력을 웹 사이트로 보여줍니다.
        return render_template('result_deepfm.html', field1=output2, field2=output1)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)

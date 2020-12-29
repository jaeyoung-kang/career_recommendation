import pandas as pd
import numpy as np
import re
import datetime as dt


def strip(series):
    series = series.str.strip('\r\n\s ')
    return series


def split(series):
    series = series.str.split('\r?\n,?\s+')  # 공백문자 제거
    return series


def explode(series, cols):
    df = pd.DataFrame(series.explode().dropna())
    df.columns = cols
    df = df.reset_index().rename({'index': 'id'}, axis=1)
    return df


def clean_str_df(df):
    oject_columns = df.select_dtypes('object').columns
    for col in oject_columns:
        df[col] = df[col].str.strip()
    return df
        

def get_top_list(series, th):
    value_counts = series.value_counts()
    return value_counts[value_counts > th].index


def basic(data):
    data = data.drop(['Unnamed: 0', 'name'], axis=1)
    data = data.drop_duplicates()
    data = data[data.isna().sum(axis=1) < data.shape[1]]
    data = data.fillna('')
    data = data.reset_index(drop=True)
    data['id'] = data.index
    return data


def career(df):
    df = df.drop(['school','project','language','award','certificate'],axis=1) # 다른 전처리요소는 제거하기
    data = pd.DataFrame()
    for k in range(len(df['career'])):
        cr = list(map(lambda x : x.replace(' ',''), df['career'][k].split('\n'))) 
        crd= []
        for i in range(2,len(cr)):
            if len(cr[i]) > 0:
                crd.append(cr[i])
        cd = pd.DataFrame(crd,columns=['career'])
        cdd = cd.loc[cd[cd['career'].str.contains('년','월',na=False)][cd[cd['career'].str.contains('년','월',na=False)].index > 2].index-1]
        cddd = cd.loc[cd[cd['career'].str.contains('년','월',na=False)][cd[cd['career'].str.contains('년','월',na=False)].index > 2].index]
        cdddd = cd.loc[cd[cd['career'].str.contains('년','월',na=False)][cd[cd['career'].str.contains('년','월',na=False)].index > 2].index-2] # 직무인덱스
        unna = pd.DataFrame(np.full((len(cd.loc[cd[cd['career'].str.contains('년','월',na=False)][cd[cd['career'].str.contains('년','월',na=False)].index > 2].index-1]), 1), df['id'][k]),columns=['id'])
        cd = pd.concat([cdd.reset_index(drop=True),unna,cddd.reset_index(drop=True),cdddd.reset_index(drop=True)],axis=1)
        data = data.append(cd)
    data.columns = ['career','ID','time','task']
 

   
    # 데이터에서 특수문자들 제거하기
    data =data.reset_index(drop=True)
    data['career'] = pd.DataFrame(map(lambda x : x.replace('\r',''),data['career']))
    data['time']=pd.DataFrame(map(lambda x : x.replace('\r',''),data['time']))
    data['task']=pd.DataFrame(map(lambda x : x.replace('\r',''),data['task']))
    data['career'] =pd.DataFrame(map(lambda x : x.replace('=-',''),data['career']))
    # time을 2020년6월-현재	4개월 이렇게 언제부터언제까지 근무했는지랑 얼마나 근무했는지로 나누자
    data = pd.concat([data,pd.DataFrame(map(lambda x : x.split('|'),data['time']))],axis=1)
    
    data_1 = data.copy()
    data_1 = data_1.drop([2,3,4],axis=1)
    data_1.columns = ['career', 'ID', 'time_2', 'task','time','period']
    data_1 = data_1.sort_index(axis=1)
    
    # career 에서 len이 지나치게 긴것들을 제거했음 (60을 기준으로) 실제 확인해보니까 4개 빼고 전부다 이상치
    # career 에서 len이 지나치게 긴것들을 제거했음 (60을 기준으로) 실제 확인해보니까 4개 빼고 전부다 이상치
    idx = []
    check =[]
    for k in range(len(data_1['career'])):
        if type(data_1['career'][k]) == str:
            if len(data_1['career'][k]) >60:
                check.append(k)
            else:
                idx.append(k)
    data_1 = data_1.drop(check)
    data_1.reset_index(drop=True,inplace=True)
    
    # len 18을 기준으로 time 이상치 제거했음 ( 이것또한 check에 담아서 눈으로 확인할수있게 함 )
    idx = []
    check= []
    for k in range(len(data_1['time'])):
        if type(data_1['time'][k]) == str:
            if len(data_1['time'][k]) >18:
                check.append(k)
            else:
                idx.append(k)
    data_1 = data_1.drop(check)
    data_1.reset_index(drop=True,inplace=True)
    
    data_1 = data_1.dropna(subset=['period']) # period에 결측치 있는거 삭제
    

    # 이제 datetime으로 바꾸기 위해서 '-' 을 기준으로 시작날짜 끝날짜 만들자
    aa = pd.DataFrame(map(lambda x : x.split('-'),data_1['time']))
    aa.columns= ['start','end']
    # datetime 형식에 맞게 replace
    aa['start'] =pd.DataFrame(map(lambda x: x.replace('년','-'),aa['start']))
    aa['start'] =pd.DataFrame(map(lambda x: x.replace('월','-01'),aa['start']))
    
    # datetime에서  len < 9보다 크면 이상치로 간주했음 ! 근데 없었음 ! 이미 이전에 많이 제거해서 필요없는듯 ~!
    idx = []
    check= []
    for k in range(len(aa['start'])):
        if type(aa['start'][k]) == str:
            if len(aa['start'][k]) <9:
                idx.append(k)
        else:
            check.append(k)
    aa = aa.drop(idx).reset_index(drop=True)
    
    # datetime으로 바꿨음 ~!
    st =pd.DataFrame(map(lambda x : dt.datetime.strptime(x,'%Y-%m-%d'),aa['start']))
    
    # 끝나는 날도 똑같이 ~! 현재는 11월 1일로 했는데 수정가능함
    aa['end'] =pd.DataFrame(map(lambda x: x.replace('년','-'),aa['end']))
    aa['end'] =pd.DataFrame(map(lambda x: x.replace('월','-01'),aa['end']))
    aa['end'] =pd.DataFrame(map(lambda x: x.replace('현재','2020-11-01'),aa['end']))
    
    ed=pd.DataFrame(map(lambda x : dt.datetime.strptime(x,'%Y-%m-%d'),aa['end']))
    time =pd.concat([st,ed],axis=1)
    time.columns = ['start','end']
    
    # 그리고 다시 datetime으로 바꾼걸 데이터프레임에 붙여줍니다
    data_3 =pd.concat([data_1.drop(idx).reset_index(drop=True),time],axis=1)
    # 이제 필요없게된 time 제거
    data_3=data_3.drop(['time_2','time'],axis=1)
     # period를 전부다 개월수로 변환하자
    bb = data_3[['period','ID']].copy()
    bb['period']=pd.DataFrame(map(lambda x: x.replace('-',''),bb['period']))
    bb.columns=['period','ID']
    # 왜 개월이랑 개월 아닌걸로 나누냐면 그냥 1년 이렇게 되어있는 데이터를 구분하려고 ! 왜 구분하냐면 아래에서 쓸 eval 함수 때문에 ! 개월수가 있는건 *12+ 하고
    # 없으면 *12 만 함 ! 
    idx =bb[bb['period'].str.endswith('년')].index
    idx2 = bb[bb['period'].str.endswith('개월')].index
    idx =idx.append(idx2)
    data_3 =data_3.iloc[idx]
    data_3 =data_3.reset_index(drop=True)
     # period를 전부다 개월수로 변환하자
    bb = data_3[['period','ID']].copy()
    bb['period']=pd.DataFrame(map(lambda x: x.replace('-',''),bb['period']))
    bb.columns=['period','ID']
    idx =bb[bb['period'].str.endswith('년')].index
    
    for i in range(len(bb['period'])):
        if i in idx:
            bb['period'][i] =bb['period'][i].replace('년','*12')
        else :
            bb['period'][i] =bb['period'][i].replace('년','*12+')
    bb['period']=pd.DataFrame(map(lambda x: x.replace('개월',''),bb['period']))
    bb['period']=pd.DataFrame(map(lambda x: eval(x),bb['period']))
    data_4 = pd.concat([data_3.reset_index(drop=True),bb.drop(['ID'],axis=1).reset_index(drop=True)],axis=1)
    
    # turn(이직횟수) 구하는 과정 ! ID로 인덱싱해서 뽑은다음에 거꾸로 정렬해서 그걸 변수로 뽑기
    id =data_4['ID'].unique()
    b= pd.DataFrame(columns =['ID', 'career', 'period', 'task', 'start', 'end', 'period','turn'])
    for i in id:
        a = pd.concat([data_4[data_4['ID'] == i].sort_values(by='start',ascending=True).reset_index(drop=True),pd.DataFrame(data_4[data_4['ID'] == i].sort_values(by='start',ascending=True).reset_index(drop=True).index,columns=['turn'])],axis=1)
        b = b.append(a)
    # 필요없게된 period2랑 turnover 제거 !
    data_4 = b.copy()
    data_4.columns = ['ID', 'career', 'period2', 'task', 'start', 'end', 'period','turn']
    data_4 = data_4.drop(['period2'],axis=1)
    data_4 = data_4.reset_index(drop=True)
    data_4['sum_peri']=0
    # 누적근무기간 구하기
    data_4['period'] = pd.to_numeric(data_4['period'])
    data_4['sum_peri'] = data_4.groupby('ID')['period'].cumsum()
    data_4.sort_values(by='ID').reset_index(drop=True,inplace=True)
    return data_4


def school(data):
    school = data['school']
    school = school.str.replace('학력', '')

    school = split_school_list(school)
    school = one_school_df(school)

    school = keep_useful_school(school)
    school = delete_strange_school(school)

    school_df = split_school_info(school)
    school_df = split_school_major(school_df)
    school_df = split_school_date(school_df)
    school_df = clean_str_df(school_df)
    return school_df


def split_school_list(school):
    def split_join_school(school_info):
        return '\n'.join(school_info.split('\n')[:3])
    # 공백원소 제거
    school = strip(school)
    school = school.str.split('(\n +){3,}')
    school = school.apply(lambda x: list(filter(str.strip, x)))
    # 3번째 new_line 이후 정보 제거
    school = school.apply(lambda x: list(map(split_join_school, x)))
    return school


def one_school_df(school):
    school = pd.DataFrame(school.explode(), columns=['school'])
    school = school.dropna()
    school = school.reset_index().rename({'index': 'id'}, axis=1)
    return school


def keep_useful_school(school):
    # 졸업, 휴학, 중퇴, 재학이 있는 학력만 남김
    def check_state(shool_info):
        states = ['졸업', '휴학', '중퇴', '재학']
        for state in states:
            if state in shool_info:
                return True
        return False
    state = school['school'].apply(check_state)
    return school[state].reset_index(drop=True)
    

def delete_strange_school(school):
    # 고등학교 학력 제거
    # 프로젝트명에 "졸업"이 자주 들어감, 프로젝트 들어간 부분 제거
    def check_state(shool_info):
        strange_states = ["고등", "젝트"]
        for state in strange_states:
            if state in shool_info:
                return False
        return True
    state = school['school'].apply(check_state)
    return school[state].reset_index(drop=True)


def split_school_info(school):
    school_df = pd.DataFrame(school['school'].str.split('\n').to_list())
    school_df.columns=['school_name', 'school_major', 'school_date']
    school_df = pd.concat([school['id'], school_df], axis=1)
    
    school_df['school_major'] = school_df['school_major'].str.split(',')
    school_df = school_df.explode('school_major')
    return school_df.reset_index(drop=True)


def split_school_major(school):
    school = school.copy()
    col_name = 'school_major'
    school[col_name] = school[col_name].fillna('')
    school[col_name] = school[col_name].str.split()
    
    school['school_major_name'] = school[col_name].apply(lambda x: ' '.join(x[:-1]))
    school['school_major_state'] = school[col_name].apply(lambda x: x[-1] if len(x) > 0 else None)

    school['school_major_state'] = school['school_major_state'].str.replace(')', '').str.split('(')

    school['school_major_level'] = school['school_major_state'].apply(lambda x: x[1] if isinstance(x,list) and len(x)> 1 else None)
    school['school_major_state'] = school['school_major_state'].apply(lambda x: x[0] if isinstance(x,list) else None)
    return school.drop(col_name, axis=1)


def split_school_date(school):
    school = school.copy()
    school['school_date'] = school['school_date'].fillna('')
    school['school_date'] = school['school_date'].str.replace('-', '').str.split()
    
    school_date = pd.DataFrame(school['school_date'].to_list()).iloc[:, :3]
    school_date.columns = ['school_start', 'school_end', 'school_state']

    school_date['school_start'] = school_date['school_start'].str.extract('(\d+)')
    school_date['school_end'] = school_date['school_end'].str.extract('(\d+)')
    return pd.concat([school.drop('school_date', axis=1), school_date], axis=1)


def skill(data, top_th=10):
    def get_skills(projects):
        if isinstance(projects, float):
            return []
        all_skill = []
        for project in projects:
            skills = project.split('\n\n')[0].split('\n, ')
            all_skill += skills if len(skills) > 1 else []
        return all_skill

    project = data['project']
    project = project.str.replace('프로젝트', '')
    project = strip(project)

    project_seperator = '    \n      \n        \n          '
    project = project.str.split(project_seperator)
    skill = project.apply(get_skills)

    skill = explode(skill, cols=['skill'])

    skill['project_time'] = skill['skill'].str.findall('\d\d\d\d년 \d월')
    skill['project_time'] = skill['project_time'].apply(lambda x: x[-1] if len(x) > 0 else None)
    skill = clean_str_df(skill)

    top_skills = get_top_list(skill['skill'], th=top_th)
    skill.loc[~skill['skill'].isin(top_skills), 'skill'] = None
    return skill.dropna(thresh=2)


def language(data, top_th=5):
    language = data['language']
    language = language.str.replace('언어', '')
    language = split(strip(language))

    language = language.apply(lambda x: [x[i * 2: i * 2 + 2] for i in range(len(x)//2)])
    
    language = explode(language, cols=['language'])

    lan = pd.DataFrame(language['language'].tolist())
    lan.columns = ['language_name', 'language_lavel']
    language = pd.concat([language['id'], lan], axis=1)
    language = clean_str_df(language)

    top_language = get_top_list(language['language_name'], th=top_th)
    language = language[language['language_name'].isin(top_language)]
    return language


def award(data):
    award = data['award']
    regex = re.compile('[0-9]{4}년[0-9]{1,2}월')
    award = award.str.replace('\n','').str.replace(' ','')
    award = award.apply(regex.findall)

    award = explode(award, cols=['award'])
    return award


def certificate(data, top_th=10):
    def get_certificates(items):
        if isinstance(items, float):
            return []
        certificates = []
        for item in items:
            if len(item) > 0:
                item = item.strip()
                certificates += [item.split('\n')]
        return certificates

    certificate = data['certificate']
    certificate = certificate.str.replace('자격증', '')
    certificate = strip(certificate)

    certificate_seperator = '\n\n\n\n'
    certificate = certificate.str.replace(' ', '').str.split(certificate_seperator)
    certificate = certificate.apply(get_certificates)

    certificate = explode(certificate, cols=['certificate'])
    certificate['certificate'] = certificate['certificate'].apply(lambda x: list(filter(str.strip, x)))

    cer = pd.DataFrame(certificate['certificate'].tolist()).iloc[:, :4]
    cer.columns = ['certificate_name', 'certificate_agency', 'certificate_time', 'certificate_etc']
    cer['certificate_time'] = cer['certificate_time'].fillna('').str.findall('\d\d\d\d년\d월')
    cer['certificate_time']  = cer['certificate_time'] .apply(lambda x: x[-1] if len(x) > 0 else None)
    certificate = pd.concat([certificate[['id']], cer], axis=1)
    certificate = clean_str_df(certificate)

    top_certificate = get_top_list(certificate['certificate_name'], th=top_th)
    certificate = certificate[certificate['certificate_name'].isin(top_certificate)]
    return certificate.drop(['certificate_agency', 'certificate_etc'], axis=1)

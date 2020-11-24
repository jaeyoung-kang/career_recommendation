import pandas as pd
import numpy as np
import re


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


def career(data):
    def get_career_time(items):
        if isinstance(items, float):
            return []
        career_list = []
        for i, item in enumerate(items):
            if re.search('[0-9]{4}년 [0-9]{1,2}월', item):
                career_list += [(items[i-1], item)]
        return career_list

    career = data['career']
    career = career.str.replace('경력', '')
    career = split(strip(career))
    career = career.apply(get_career_time)
    return pd.concat([data['id'], career], axis=1)


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
    return skill

    language = data['language']
    language = language.str.replace('언어', '')
    language = split(strip(language))
    return pd.concat([data['id'], language], axis=1)


def award(data):
    award = data['award']
    regex = re.compile('[0-9]{4}년[0-9]{0,1}월')
    award = award.str.replace('\n','').replace(' ','')
    award = award.apply(regex.findall)
    return pd.concat([data['id'], award], axis=1)


def certificate(data):
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
    return pd.concat([data['id'], certificate], axis=1)

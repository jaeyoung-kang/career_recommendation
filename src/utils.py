import pandas as pd
import numpy as np
import re


def strip(series):
    series = series.str.strip('\r\n\s ')
    return series


def split(series):
    series = series.str.split('\r?\n,?\s+')  # 공백문자 제거
    return series


def preprocess_basic(data):
    data = data.drop(['Unnamed: 0', 'name'], axis=1)
    data = data.drop_duplicates()
    data = data[data.isna().sum(axis=1) < data.shape[1]]
    data = data.reset_index(drop=True)
    data['id'] = data.index
    return data


def preprocess_school(data):
    school = data['school']
    school = school.str.replace('학력', '')
    school = split(strip(school))
    return pd.concat([data['id'], school], axis=1)


def preprocess_language(data):
    language = data['language']
    language = language.str.replace('언어', '')
    language = split(strip(language))
    return pd.concat([data['id'], language], axis=1)


def preprocess_skill(data):
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
    skill.name = 'skill'
    return pd.concat([data['id'], skill], axis=1)


def preprocess_certificate(data):
    data['index'] = data.index 
    nan_index = []
    val_index = []
    certificate = data[['index', 'certificate']]
    for i in range(0, len(data)):
        if type(certificate['certificate'][i]) != str:
            nan_index.append(i)
        else:
            val_index.append(i)
    cert_val_index = certificate['index'][val_index]
    cert_val_cert = certificate['certificate'][certificate['index'][val_index]]
    cert_val = pd.concat([cert_val_index, cert_val_cert], axis=1)
    for i in cert_val['index']:
        tmp = cert_val['certificate'][i]
        tmp2 = tmp.replace("\r\n", "\n")
        cert_val['certificate'][i] = tmp2

    for i in cert_val2['index']:
        tmp = cert_val2['certificate'][i]
        tmp2 = tmp.replace(" ", '')
        cert_val2['certificate'][i] = tmp2

    for i in cert_val2['index']:
        tmp = cert_val2['certificate'][i]
        tmp2 = tmp.replace("자격증\n\n\n", '')
        cert_val2['certificate'][i] = tmp2

    new_df = pd.DataFrame(columns=['index', 'cert1', 'cert2', 'cert3', 'cert4'])

    len_list = []
    for i in cert_val2['index']:
        tmp = cert_val2['certificate'][i]
        tmp2 = tmp.split('\n\n\n\n')
        length = len(tmp2)
        for single_cert in tmp2:
            tmp3 = single_cert.replace('\n', ' ')
            result = tmp3.split(' ')
            new_result = [item for item in result if item != '']
            k = len(new_result)
            if k == 1:
                final_result = [i] + new_result + ['dummy'] + ['dummy'] + ['dummy']
                new_df = new_df.append({'index' : final_result[0],
                                    'cert1' : final_result[1],
                                    'cert2' : final_result[2],
                                    'cert3' : final_result[3],
                                    'cert4' : final_result[4]}, ignore_index = True)
            elif k == 2:
                final_result = [i] + new_result + ['dummy'] + ['dummy']
                new_df = new_df.append({'index' : final_result[0],
                                    'cert1' : final_result[1],
                                    'cert2' : final_result[2],
                                    'cert3' : final_result[3],
                                    'cert4' : final_result[4]}, ignore_index = True)
            elif k == 3:
                final_result = [i] + new_result + ['dummy']
                new_df = new_df.append({'index' : final_result[0],
                                    'cert1' : final_result[1],
                                    'cert2' : final_result[2],
                                    'cert3' : final_result[3],
                                    'cert4' : final_result[4]}, ignore_index = True)            
            elif k >= 4:
                final_result = [i] + new_result[0:3]
                final_result.append('-'.join(new_result[3:]))
                new_df = new_df.append({'index' : final_result[0],
                                    'cert1' : final_result[1],
                                    'cert2' : final_result[2],
                                    'cert3' : final_result[3],
                                    'cert4' : final_result[4]}, ignore_index = True)
    return new_df


def preprocess_award(data):
    award = data['award']
    regex = re.compile('[0-9]{4}년[0-9]{0,1}월')
    award = award.str.replace('\n','').replace(' ','')
    award = award.apply(regex.findall)
    return pd.concat([data['id'], award], axis=1)


def preprocess_career(data):
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

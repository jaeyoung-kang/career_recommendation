import pandas as pd
import numpy as np
import re


def strip(series):
    series = series.str.strip('\r\n\s ')
    return series


def split(series):
    series = series.str.split('\r?\n,?\s+')  # 공백문자 제거
    return series


def basic(data):
    data = data.drop(['Unnamed: 0', 'name'], axis=1)
    data = data.drop_duplicates()
    data = data[data.isna().sum(axis=1) < data.shape[1]]
    data = data.reset_index(drop=True)
    data['id'] = data.index
    return data


def school(data):
    school = data['school']
    school = school.str.replace('학력', '')
    school = split(strip(school))
    return pd.concat([data['id'], school], axis=1)


def language(data):
    language = data['language']
    language = language.str.replace('언어', '')
    language = split(strip(language))
    return pd.concat([data['id'], language], axis=1)


def skill(data):
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


def award(data):
    award = data['award']
    regex = re.compile('[0-9]{4}년[0-9]{0,1}월')
    award = award.str.replace('\n','').replace(' ','')
    award = award.apply(regex.findall)
    return pd.concat([data['id'], award], axis=1)


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

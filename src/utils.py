import pandas as pd
import numpy as np


def strip(series):
    series = series.str.strip('\r\n\s ')
    return series

def split(series):
    series = series.str.split('\r?\n,?\s+')  # 공백문자 제거
    return series

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
    lst=[]
    for i in data['award']:
        lst.append(i)
    lst_1=[]
    for i in lst:
        lst_1.append(str(i).replace('\n','').replace(' ',''))
    return data['award']


def preprocess_career(df):
    df = df.drop(['school','project','language','award','certificate'],axis=1)
    df = df.dropna()
    df= df.reset_index(drop=True)

    data = pd.DataFrame(columns=['career'])
    for k in range(len(df['career'][0:10])):
        print(k)
        cr = list(map(lambda x : x.replace(' ',''), df['career'][k].split('\n'))) 
        crd= []
        for i in range(2,len(cr)):
            if len(cr[i]) > 0:
                crd.append(cr[i])
        cd = pd.DataFrame(crd,columns=['career'])
        cdd = cd.loc[cd[cd['career'].str.contains('년','월',na=False)].index-1]
        cddd = cd.loc[cd[cd['career'].str.contains('년','월',na=False)].index]
        cddd.columns=['time']
        unna = pd.DataFrame(np.full((len(cd.loc[cd[cd['career'].str.contains('년','월',na=False)].index-1]), 1), df['Unnamed: 0'][k]),columns=['Unnamed: 0'])
        cd = pd.concat([cdd.reset_index(drop=True),unna,cddd.reset_index(drop=True)],axis=1)
        data = data.append(cd)
    data.columns = ['career','ID','time']
    return data

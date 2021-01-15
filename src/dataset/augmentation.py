import numpy as np


CAREER_TASKS=[
    '인사관리', '금융', '소프트웨어엔지니어', '콘텐츠제작', '광고기획자', '연구원',
    'UX/UI디자이너', '디자이너', '교육', '기획/PM', '웹개발자', '풀스택개발자', '데이터사이언티스트',
    '마케팅', '대표이사', '경영지원', '행정및경영지원', '비즈니스', '운영', 'SW 개발', '매니저',
    '영업관리', '프론트엔드개발자', '기자', '백엔드개발자', '회계및재무관리',
    'Researcher/Analyst', '그래픽디자이너', '품질관리', 'IOS개발자', '통번역', '투자',
    '에디터', '팀원', '디자인', '컨설턴트', '법무', '서비스운영', 'cmo', '엔지니어', '작가',
    '하드웨어엔지니어', '보안관리', 'associate', '무역', 'accountexecutive', '게임 개발',
    'HW 개발', '기계', '조교', '생산관리', '고객상담', '바리스타'
]

def make_binary_target(
    data,
    target_col,
    target_lst=None,
    positive_ratio=0.5,
    sep = ','
):
    label = target_col + '_label'
    all_target_col = 'all_' + target_col
    if target_lst is None:
        target_lst = CAREER_TASKS
    data[all_target_col] = sep.join(target_lst)
    data[all_target_col] = data[all_target_col].str.split(sep)
    data = data.explode(all_target_col)
    data = data.reset_index(drop=True)

    data.loc[data[target_col] == data[all_target_col], label] = 1
    data.loc[data[target_col] != data[all_target_col], label] = 0
    data = data.drop(target_col, axis=1)
    data = data.rename({all_target_col: target_col}, axis=1)

    data = data.sort_values(label, ascending=False)
    data = data.drop_duplicates(
        subset=set(data.columns) - set([label]),
    ).sort_index()

    n_positive = int(data[label].sum())
    if positive_ratio > 0: # 0보다 작은 경우 그대로 아웃풋
        n_negative = int(n_positive * (1 - positive_ratio) / positive_ratio)
        negative_index = np.random.choice(data[data[label] == 0].index, n_negative)
        positive_index = data[data[label] == 1].index
        data = data.loc[np.concatenate([positive_index, negative_index])]

    return data.sort_index().reset_index(drop=True)

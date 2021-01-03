import pandas as pd
import re
from konlpy.tag import Mecab


class MajorCleaner:
    def __init__(self):
        self.mecab = Mecab()
        self.top_sections = [
            '경영', '컴퓨터', '디자인', '정보', '전자', '산업', '경제', '영어', '언어', '소프트웨어', '시각',
           '국제', '미디어', '전기', '통신', '기계', '국어', '문화', '사회', '영상', '시스템', '교육',
           '광고', '디지털', '방송', '홍보', '심리', '관광', '콘텐츠', '관리', '통계', '행정', '건축',
           '예술', '커뮤니케이션', '신문', '통상', '환경', '금융', '정치', '마케팅', '언론', '수학', '무역',
           '글로벌', '생명', '법학', '멀티미디어', '물리', '화학'
        ]
        self.top_majors = [
            '경영', '컴퓨터', '디자인', '언어', '경제', '디자인시각', '전자', '영어', '소프트웨어', '기계',
            '정보통신', '디자인산업', '사회', '정보', '전기전자', '정보컴퓨터', '행정', '심리', '국제', '산업',
            '교육', '방송신문', '건축', '광고홍보', '국어', '통계', '소프트웨어컴퓨터', '법학', '정치', '수학',
            '화학', '생명', '관리', '물리', '예술', '국제통상', '환경', '경영관광', '경영정보', '경영산업',
            '마케팅', '미디어', '멀티미디어', '전기', '금융', '경영국제', '문화콘텐츠', '무역', '문화',
            '미디어커뮤니케이션', '언론정보', '영상', '전기전자컴퓨터', '시스템', '디자인디지털미디어', '관광',
            '디자인커뮤니케이션', '디자인영상', '시스템정보', '정보컴퓨터통신', '디지털미디어', '경제금융', '경영글로벌',
            '디자인시각정보', '미디어영상', '국제무역', '디자인시각영상', '커뮤니케이션', '글로벌', '문화언어', '경영관리',
            '전자정보통신', '전자정보', '교육영어', '산업시스템', '콘텐츠', '방송영상', '디지털콘텐츠', '정보통계',
            '디지털', '전자통신', '언론홍보', '광고디자인', '디자인멀티미디어', '디자인미디어', '광고', '전기정보',
            '전기컴퓨터', '디자인콘텐츠', '교육국어', '경영문화예술', '미디어콘텐츠', '산업시스템정보', '글로벌미디어',
            '기계디자인시스템', '시스템컴퓨터', '경영디자인', '산업심리', '경영경제',
        ] + [
            '관광문화', '미디어정보', '방송', '디자인환경', '언론영상홍보', '통상', '전자컴퓨터', '디지털마케팅',
            '건축디자인', '산업정보', '디자인산업정보', '교육언어', '교육수학', '디자인디지털', '경영시스템',
            '경영시스템정보', '교육컴퓨터', '문화정보', '경제통상', '디지털정보', '경제글로벌', '물리전자', '경영예술',
            '관리정보', '경영마케팅', '글로벌통상', '언론영상', '사회정보', '경제국제무역', '언론', '전자컴퓨터통신',
            '관광영어', '기계시스템', '미디어소프트웨어', '컴퓨터통신', '관리디자인', '문화산업', '관광국제', '관리마케팅',
            '국제정치', '광고디자인영상', '교육사회', '국제금융', '통신', '디자인정보', '시스템환경', '경영디지털',
            '영상콘텐츠', '디지털전자', '광고언론', '건축사회환경', '경제사회', '디자인컴퓨터', '산업소프트웨어', '경제정치',
            '경영산업정보', '소프트웨어정보',
        ]

    def _select_main_major(self, major_name):
        sections = self.mecab.nouns(major_name)
        main_sections = [section for section in sections if section in self.top_sections]
        main_sections = sorted(set(main_sections))
        return ''.join(main_sections)

    def _transform_one(self, major_name):
        clean_major = self._select_main_major(major_name)
        if clean_major in self.top_majors:
            return clean_major
        for section in self.top_sections:
            if section in major_name:
                return section
        return ''
    
    def transform(self, majors):
        if isinstance(majors, str):
            _majors = pd.Series([majors])
        elif isinstance(majors, list):
            _majors = pd.Series(majors)
        else:
            assert isinstance(majors, pd.Series), 'str or list or pd.Series'
            _majors = majors
        
        _clean_majors = _majors.apply(self._transform_one)
        
        if isinstance(majors, str):
            return _clean_majors[0]
        elif isinstance(majors, list):
            return _clean_majors.tolist()
        return _clean_majors


def preprocess_major(major):
    major = major.copy()
    # ** 과정 제거 ex) 양성과정 
    major = major.apply(lambda x: '' if '과정' in x else x)  
    # **대학교, 대학원, 융합
    major = major.str.replace('대학교', '')
    major = major.str.replace('대학원', '')
    major = major.str.replace('융합', '')
    # 특수문자 제거
    major = major.str.replace('.', '').str.replace('/', '').str.replace('-', '').str.replace('&', '')
    # 학부 -> 학과
    major = major.str.replace('학부', '학과')  
    # **대학 ?? -> ??
    major = major.str.split('대학 ').apply(lambda x: x[-1])  
    # *학 -> *학과
    major = major.str.replace(r'학$', '학과')  
    major = major.str.replace('학 ', '학과 ')
    # **(??) -> **
    major = major.str.split('(').apply(lambda x: x[0])
    # **학과 ?? -> **학과
    major = major.apply(
        lambda x: \
        re.split('[학]과', x.replace(' ', ''))[0] \
        + (re.search(r'[학]과', x).group(0) if re.search(r'[학]과', x) else '')
    )
    # **전공 -> **
    major = major.str.replace(r'전공$', '')
    
    # 영어
    major = major.str.lower()
    major = manual_translation(major)
    
    # **학과, **과, **학, **공 -> **
    major = major.str.replace(r'공?학과$', '').str.replace(r'공?학$', '')
    major = major.str.replace(r'과?학과$', '').str.replace(r'과?학$', '')
    major = major.str.replace(r'과$', '')
    major = major.str.replace(r'공$', '')
    # 한글자는 학 추가 ex)수 -> 수학
    major = major.apply(lambda x: x + '학' if len(x) == 1 else x)
    
    # *어*문 -> 언어
    major = major.str.replace(r'어*문$', '어')
    major = major.str.replace(r'[가-힣]*[^영국디웨제리케니싱튜투헤웹]어', '언어')
    
    return major


def manual_translation(series):
    series = series.str.replace('engineering', '공')
    series = series.str.replace('businessadministration', '경영')
    series = series.str.replace('economics', '경제')
    series = series.str.replace('computerscience', '컴퓨터')
    series = series.str.replace('business', '경영')
    series = series.str.replace('management', '관리')
    series = series.str.replace('marketing', '마케팅')
    series = series.str.replace('electrical', '전기')
    series = series.str.replace('industrial', '산업')
    series = series.str.replace('design', '디자인')
    series = series.str.replace('mathematics', '수')
    series = series.str.replace('finance', '금융')
    series = series.str.replace('computer', '컴퓨터')
    series = series.str.replace('software', '소프트웨어')
    series = series.str.replace('global', '국제')
    series = series.str.replace('graphic', '그래픽')
    series = series.str.replace('and', '')
    series = series.str.replace('information', '정보')
    series = series.str.replace('system', '시스템')
    series = series.str.replace('art', '예술')
    return series


if __name__ == "__main__":
    df = pd.read_csv('data.txt', index_col='Unnamed: 0')
    df['school_major_name'] = df['school_major_name'].fillna('')
    df['school_major_name'] = preprocess_major(df['school_major_name'])

    major_cleaner = MajorCleaner()
    df['school_major_name'] = major_cleaner.transform(df['school_major_name'])

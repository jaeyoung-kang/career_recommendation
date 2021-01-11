import numpy as np
import pandas as pd

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DeepFM

from src.utils.label_encoder import MultiFeatureLabelEncoder
from src.utils.label_encoder import VariableLenghthLabelEncoder


class DeepFMTrainer:
    def __init__(
        self,
        target,
        sparse_features,
        variable_length_features=None,
        target_is_sparse=False,
    ):
        self.multi_label_encoder = MultiFeatureLabelEncoder()
        self.sparse_features = sparse_features

        if isinstance(target, list):
            self.target = target
        else:
            self.target = [target]
        if target_is_sparse:
            self.target_label_encoder = MultiFeatureLabelEncoder()
        else:
            self.target_label_encoder = None

        self.variable_length_features = variable_length_features
        if self.variable_length_features is not None:
            self.variable_length_label_encoders = {}
            for feat in self.variable_length_features:
                self.variable_length_label_encoders[feat] = VariableLenghthLabelEncoder()
        else:
            self.variable_length_label_encoders = None
        self.variable_length_features_max_len = None

        self.vocabulary_size_dict = {}
        self.model = None

    def fit(
        self,
        train_data,
        embedding_dim=4,
        task='binary',
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
        device='cpu',
        valid_ratio=0.2,
        batch_size=256,
        epochs=5,
    ):

        self.multi_label_encoder.fit(
            data=train_data,
            features=self.sparse_features,
        )
        if self.variable_length_features:
            for feat in self.variable_length_features:
                self.variable_length_label_encoders[feat].fit(
                    train_data[feat],
                )
        train_data = self.label_encoded_data(train_data)

        if self.target_label_encoder:
            train_data[self.target] = self.target_label_encoder.fit_transform(
                data=train_data[target_label_encoder],
                features=self.target,
            )

        if self.variable_length_features:
            self.variable_length_features_max_len = {}
            for feat in self.variable_length_features:
                genres_length = np.array(list(map(len, train_data[feat])))
                self.variable_length_features_max_len[feat] = min(5, max(genres_length))

        for feat in self.sparse_features:
            self.vocabulary_size_dict[feat] = train_data[feat].max() + 2
        if self.variable_length_features:
            for feat in self.variable_length_features:
                self.vocabulary_size_dict[feat] = train_data[feat].explode().max() + 1
        print('Label Encoding ...')
        print()

        self.model = self.build_model(
            embedding_dim=embedding_dim,
            task=task,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            device=device,
        )
        print('Build Model ...')
        print()

        model_input = self.build_model_input(train_data)
        print('Model Input ...\n\texmple)')
        for key, value in model_input.items():
            if isinstance(value, pd.Series):
                print('\t', key, ': ', value.iloc[0])
            else:
                print('\t', key, ': ', value[0])

        print()

        self.model.fit(
            model_input, train_data[self.target].values,
            batch_size=batch_size, epochs=epochs, validation_split=valid_ratio, verbose=2,
        )

    def predict(
        self,
        test_data,
    ):
        test_data = self.label_encoded_data(test_data)
        return self.predict_encoded_data(test_data)

    def predict_encoded_data(
        self, 
        test_data,
    ):
        if isinstance(test_data, np.ndarray):
            test_data = pd.DataFrame(test_data, columns=self.sparse_features)
            for feat in self.variable_length_features:
                test_data[feat] = ''
                test_data[feat] = test_data[feat].str.split()
        model_input = self.build_model_input(test_data)
        return self.model.predict(model_input)

    def label_encoded_data(self, data):
        data = self.multi_label_encoder.transform(data)
        if self.variable_length_features:
            for feat in self.variable_length_features:
                data[feat] = self.variable_length_label_encoders[feat].transform(
                    data[feat],
                )
        return data

    def build_model(
        self,
        embedding_dim=4,
        task='binary',
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
        device='cpu',
    ):
        fixlen_feature_columns = [
            SparseFeat(
                feat, 
                vocabulary_size=self.vocabulary_size_dict[feat], 
                embedding_dim=embedding_dim,
            ) for feat in self.sparse_features
        ]

        if self.variable_length_features:
            varlen_feature_columns = [
                VarLenSparseFeat(
                    SparseFeat(
                        feat, 
                        vocabulary_size=self.vocabulary_size_dict[feat], 
                        embedding_dim=embedding_dim,
                    ),
                    maxlen=self.variable_length_features_max_len[feat],
                    combiner='mean',
                ) for feat in self.variable_length_features
            ] 
        else:
            varlen_feature_columns = []
        
        linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

        model = DeepFM(linear_feature_columns, dnn_feature_columns, task=task, device=device)
        model.compile(optimizer, loss, metrics)
        return model
    
    def build_model_input(self, data):
        model_input = {name: data[name] for name in self.sparse_features}
        if self.variable_length_features:
            for feat in self.variable_length_features:
                pad_variable_length_features = pad_sequences(
                    data[feat], maxlen=self.variable_length_features_max_len[feat], padding='post',
                )
                model_input[feat] = pad_variable_length_features
        return model_input

    def test(
        self,
        **kargs,
    ):
        target_col = 'career_task'
        data = {feat: None for feat in self.sparse_features}
        if self.variable_length_features is not None:
            data.update({feat: None for feat in self.variable_length_features})
        data.pop(target_col)

        data.update(**kargs)
        data = pd.DataFrame([data])
        if 'school_major_name' in data.columns:
            data.loc[0, 'school_major_name'] = self.school_major_name_list(data.loc[0, 'school_major_name'])
            data['school_major_name'] = data['school_major_name'].str.split(',')

        data = data.merge(
            right=self.target_df,
            left_index=True,
            right_index=True,
        )
        data['predict'] = self.predict(data)
        result = data.iloc[
            data.reset_index().groupby('index')['predict'].idxmax().tolist()[0]
        ]
        return result[target_col]

    @property
    def target_df(
        self
    ):
        career_task = [
            '인사관리', '금융', '소프트웨어엔지니어', '콘텐츠제작', '광고기획자', '연구원',
            'UX/UI디자이너', '디자이너', '교육', '기획/PM', '웹개발자', '풀스택개발자', '데이터사이언티스트',
            '마케팅', '대표이사', '경영지원', '행정및경영지원', '비즈니스', '운영', 'SW 개발', '매니저',
            '영업관리', '프론트엔드개발자', '기자', '백엔드개발자', '회계및재무관리',
            'Researcher/Analyst', '그래픽디자이너', '품질관리', 'IOS개발자', '통번역', '투자',
            '에디터', '팀원', '디자인', '컨설턴트', '법무', '서비스운영', 'cmo', '엔지니어', '작가',
            '하드웨어엔지니어', '보안관리', 'associate', '무역', 'accountexecutive', '게임 개발',
            'HW 개발', '기계', '조교', '생산관리', '고객상담', '바리스타'
       ]
        df = pd.DataFrame({'career_task': career_task}, index=[0] * len(career_task))
        return df

    def school_major_name_list(
        self,
        major_name
    ):
        sections = [
            '경영', '컴퓨터', '디자인', '정보', '전자', '산업', '경제', '영어', '언어', '소프트웨어', '시각',
           '국제', '미디어', '전기', '통신', '기계', '국어', '문화', '사회', '영상', '시스템', '교육',
           '광고', '디지털', '방송', '홍보', '심리', '관광', '콘텐츠', '관리', '통계', '행정', '건축',
           '예술', '커뮤니케이션', '신문', '통상', '환경', '금융', '정치', '마케팅', '언론', '수학', '무역',
           '글로벌', '생명', '법학', '멀티미디어', '물리', '화학'
        ]
        major_list = []
        for section in sections:
            if section in major_name:
                major_list += [section]
        return ','.join(major_list)

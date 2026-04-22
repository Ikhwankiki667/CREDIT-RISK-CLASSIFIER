import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    """
    Kelas Modular untuk membersihkan dan mentransformasi data mentah
    menjadi format yang bisa dibaca oleh algoritma Machine Learning.
    """
    def __init__(self):
        self.preprocessor = None
        self.numeric_features = []
        self.categorical_features = []

    def fit_transform(self, df):
        """
        Mempelajari pola data (fit) sekaligus mengubahnya (transform).
        Dipanggil HANYA SAAT TRAINING.
        """
        # Pisahkan fitur (X) dan target (y)
        X = df.drop('loan_status', axis=1)
        y = df['loan_status']

        self.numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Pipeline untuk angka: Isi data kosong dengan median, lalu standarisasi (Scaling)
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Pipeline untuk teks: Isi data kosong dengan modus, lalu ubah jadi angka (OneHot)
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Gabungkan kedua pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.numeric_features),
                ('cat', cat_transformer, self.categorical_features)
            ])

        # Eksekusi pemrosesan data X
        X_processed = self.preprocessor.fit_transform(X)
        
        return X_processed, y

    def transform(self, df_input):
        """
        Mengubah data baru berdasarkan pola yang sudah dipelajari.
        Dipanggil SAAT PREDIKSI NASABAH BARU.
        """
        if self.preprocessor is None:
            raise Exception("Preprocessor belum di-fit dengan data training!")
        return self.preprocessor.transform(df_input)
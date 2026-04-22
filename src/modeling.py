from sklearn.ensemble import RandomForestClassifier

class CreditRiskModel:
    """
    Kelas Modular khusus untuk membangun, melatih, dan menggunakan
    algoritma Random Forest Classifier.
    """
    def __init__(self):
        # max_depth=8 agar model tidak overfitting dan probabilitasnya masuk akal
        self.model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        self.default_class_index = 1

    def train(self, X_processed, y):
        """Melatih model menggunakan data yang sudah di-preprocess."""
        self.model.fit(X_processed, y)
        
        # Deteksi otomatis: Kelas mana yang merupakan 'Gagal Bayar' (Biasanya bernilai 1)
        classes = list(self.model.classes_)
        if 1 in classes:
            self.default_class_index = classes.index(1)
        else:
            self.default_class_index = 0

    def predict_default_prob(self, X_input_processed):
        """Mengeluarkan probabilitas (persentase) risiko gagal bayar (PD)."""
        return self.model.predict_proba(X_input_processed)[0][self.default_class_index]
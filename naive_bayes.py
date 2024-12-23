def prediksi_naive_bayes(data_prediction, prior_prob, conditional_prob):
    """
    Melakukan prediksi menggunakan model Naive Bayes.

    Parameters:
        - data_prediction (pd.DataFrame): Dataset untuk prediksi.
        - prior_prob (pd.Series): Probabilitas prior setiap kelas.
        - conditional_prob (dict): Probabilitas kondisional untuk setiap fitur.

    Returns:
        - predictions (list): Daftar prediksi kelas untuk setiap baris data.
    """
    predictions = []

    # Validasi input
    if data_prediction.empty:
        raise ValueError("Dataset untuk prediksi kosong.")
    if prior_prob.empty:
        raise ValueError("Probabilitas prior kosong.")
    if not conditional_prob:
        raise ValueError("Probabilitas kondisional kosong.")

    for index, row in data_prediction.iterrows():
        class_prob = {}

        for class_name in prior_prob.index:
            try:
                prob = prior_prob[class_name]

                for column in data_prediction.columns:
                    if column not in ['Nama Siswa', 'Hasil']:
                        prob *= conditional_prob.get(column, {}).get((class_name, row[column]), 0)

                class_prob[class_name] = prob
            except KeyError as e:
                raise KeyError(f"Kolom '{column}' tidak ditemukan dalam data atau probabilitas.") from e

        if not class_prob:
            raise ValueError(f"Tidak ada probabilitas yang dihitung untuk baris ke-{index}.")

        prediction = max(class_prob, key=class_prob.get)
        predictions.append(prediction)

    return predictions


def compare_results(dataset1, dataset2, kolom):
    """
    Membandingkan hasil prediksi dengan data manual.

    Parameters:
        - dataset1 (pd.DataFrame): Dataset pertama (manual).
        - dataset2 (pd.DataFrame): Dataset kedua (prediksi).
        - kolom (str): Nama kolom untuk dibandingkan.

    Returns:
        - dict: Statistik dan detail perbandingan.
    """
    if dataset1.empty or dataset2.empty:
        raise ValueError("Salah satu atau kedua dataset kosong.")
    if kolom not in dataset1.columns:
        raise KeyError(f"Kolom '{kolom}' tidak ditemukan di dataset pertama.")
    if kolom not in dataset2.columns:
        raise KeyError(f"Kolom '{kolom}' tidak ditemukan di dataset kedua.")
    if len(dataset1) != len(dataset2):
        raise ValueError(f"Dataset memiliki jumlah baris yang berbeda: {len(dataset1)} vs {len(dataset2)}.")

    dataset1 = dataset1.reset_index(drop=True)
    dataset2 = dataset2.reset_index(drop=True)

    comparison = dataset1[kolom] == dataset2[kolom]

    differences = dataset1[~comparison].copy()
    differences['Dataset Manual'] = differences[kolom]
    differences['Dataset Prediksi'] = dataset2.loc[differences.index, kolom]

    correct = dataset1[comparison].copy()
    correct['Dataset Manual'] = correct[kolom]
    correct['Dataset Prediksi'] = dataset2.loc[correct.index, kolom]

    total_rows = len(dataset1)
    total_differences = len(differences)
    total_correct = len(correct)

    if total_rows == 0:
        raise ValueError("Dataset kosong setelah reset index.")

    difference_percentage = (total_differences / total_rows) * 100
    correct_percentage = (total_correct / total_rows) * 100

    return {
        'differences': differences,
        'correct': correct,
        'total_differences': total_differences,
        'total_correct': total_correct,
        'total_rows': total_rows,
        'difference_percentage': difference_percentage,
        'correct_percentage': correct_percentage
    }


def classify_class(nilai):
    """
    Mengklasifikasikan nilai ke dalam kategori kelas.

    Parameters:
        - nilai (float): Nilai yang akan diklasifikasikan.

    Returns:
        - str: Kategori kelas.
    """
    try:
        if not (0 <= nilai <= 100):
            raise ValueError(f"Nilai {nilai} di luar rentang yang valid (0-100).")

        if 80 <= nilai <= 100:
            return 'SB'  # Sangat Baik
        elif 70 <= nilai <= 79:
            return 'B'  # Baik
        elif 60 <= nilai <= 69:
            return 'C'  # Cukup
        elif 55 <= nilai <= 59:
            return 'K'  # Kurang
        elif 50 <= nilai <= 54:
            return 'SK'  # Sangat Kurang
        else:
            return 'Tidak Diketahui'
    except TypeError as e:
        raise TypeError(f"Nilai '{nilai}' tidak valid untuk klasifikasi.") from e

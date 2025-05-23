出貨量預測：時間序列分析專案

專案簡介
本專案展示了如何利用 Python 進行時間序列預測工作。內容包括：

指數平滑法（Holt-Winters）： 採用多種模型（加法、乘法、無趨勢以及非季節性）進行出貨量預測，並透過圖形呈現各模型擬合效果及預測結果。

平穩性檢定： 透過 ADF 與 KPSS 測試檢查各個產品系列的資料平穩性，判斷是否需要先進行差分。

SARIMAX 模型： 利用季節性 ARIMA 模型分別對 PC（desktop）與筆記本（laptop）系列進行建模與預測，並用殘差檢定來驗證模型穩定性。

結果解讀：

圖形： 程式會繪製真實數據與各模型預測結果的比對圖，方便觀察擬合及預測情形。

評估指標： 輸出 RMSE、MAE 及 MAPE 等預測誤差指標，以評估模型效果。

平穩性檢定： 透過 ADF 與 KPSS 測試輸出平穩性檢定結果，幫助決定是否需要對資料做差分處理。

Задание 1: Сравнение CNN и полносвязных сетей (40 баллов)
Создайте файл homework_cnn_vs_fc_comparison.py:

1.1 Сравнение на MNIST (20 баллов)
 Сравните производительность на MNIST:
 - Полносвязная сеть (3-4 слоя)
 - Простая CNN (2-3 conv слоя)
 - CNN с Residual Block
 
 Для каждого варианта:
 - Обучите модель с одинаковыми гиперпараметрами
 - Сравните точность на train и test множествах
 - Измерьте время обучения и инференса
 - Визуализируйте кривые обучения
 - Проанализируйте количество параметров

 - Добавил класс fullyConnectedModel,посчитали количество параметров для каждой модели и обучили, вывели результаты через графики loss и accuracy 
Using device: cuda
Simple CNN parameters: 421642
Residual CNN parameters: 160906
FUlly CON.parameters: 567434
![image](https://github.com/user-attachments/assets/511ef8a9-a29c-423a-a650-08e964a3a04e)
Epoch 5/5:
Train Loss: 0.0388, Train Acc: 0.9878
Test Loss: 0.0784, Test Acc: 0.9769
--------------------------------------------------
80.91369724273682
![image](https://github.com/user-attachments/assets/c86d8742-004c-4c06-86fa-436412fed14e)
- Epoch 5/5:
Train Loss: 0.0247, Train Acc: 0.9916
Test Loss: 0.0260, Test Acc: 0.9925
--------------------------------------------------
89.69417595863342

![image](https://github.com/user-attachments/assets/0786cb99-cff7-4c9f-ba92-2dbe1d30b9bf)
- Epoch 5/5:
Train Loss: 0.0208, Train Acc: 0.9935
Test Loss: 0.0502, Test Acc: 0.9860
--------------------------------------------------
100.30287504196167
1.2 Сравнение на CIFAR-10 (20 баллов)
 Сравните производительность на CIFAR-10:
 - Полносвязная сеть (глубокая)
 - CNN с Residual блоками
 - CNN с регуляризацией и Residual блоками
 
 Для каждого варианта:
 - Обучите модель с одинаковыми гиперпараметрами
- Сравните точность и время обучения
- Проанализируйте переобучение
 - Визуализируйте confusion matrix
 - Исследуйте градиенты (gradient flow)

- Сделали на данных cifar
Simple CNN parameters: 620362
Residual CNN parameters: 160906
FUlly CON.parameters: 1738890
![image](https://github.com/user-attachments/assets/6967145e-5402-4836-8f03-6aa50267260d)\
 -Epoch 5/5:
Train Loss: 0.4546, Train Acc: 0.8426
Test Loss: 0.6137, Test Acc: 0.7844
--------------------------------------------------
111.12743282318115
![image](https://github.com/user-attachments/assets/3204da37-471d-41f0-b93b-a97725e64056)
- Epoch 5/5:
Train Loss: 1.2754, Train Acc: 0.5494
Test Loss: 1.3972, Test Acc: 0.5084
--------------------------------------------------
73.79678559303284
![image](https://github.com/user-attachments/assets/6d444ff9-9288-448f-962d-ccb5cdf2a13e)
![image](https://github.com/user-attachments/assets/3892b13a-797a-4d58-9c3b-19245db1f59d)
![image](https://github.com/user-attachments/assets/77d28de9-ad83-4e46-a18a-d855b8421a66)
![image](https://github.com/user-attachments/assets/c8c3a724-3ac6-4377-9198-0b311de59124)
![image](https://github.com/user-attachments/assets/c6c65688-e56c-4b76-b811-88b5d2142a6f)
![image](https://github.com/user-attachments/assets/a4cf8f93-5e40-40bb-a7c7-e24b61857927)
![image](https://github.com/user-attachments/assets/cdf2b0a0-b7ca-44bc-b3a1-e39e19788964)













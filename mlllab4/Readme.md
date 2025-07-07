![image](https://github.com/user-attachments/assets/d9127425-681b-4ae9-a29f-534cb5188698)Задание 1: Сравнение CNN и полносвязных сетей (40 баллов)
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

2.1 Влияние размера ядра свертки (15 баллов)
 Исследуйте влияние размера ядра свертки:
 - 3x3 ядра
 - 5x5 ядра
 - 7x7 ядра
 - Комбинация разных размеров (1x1 + 3x3)
 
 Для каждого варианта:
- Поддерживайте одинаковое количество параметров
 - Сравните точность и время обучения
- Проанализируйте рецептивные поля
 - Визуализируйте активации первого слоя

   3x3 ![image](https://github.com/user-attachments/assets/9904694f-64be-4f57-bb99-f3ac4a759941)
   5x5 ![image](https://github.com/user-attachments/assets/214c182e-79b5-4d87-b7e7-4587304d8b78)
   7x7 ![image](https://github.com/user-attachments/assets/29a83b20-ac3e-429e-9fc8-f831a5b7b562)
   - Комбинация разных размеров (1x1 + 3x3) ![image](https://github.com/user-attachments/assets/ddec6e14-a571-4d6a-a350-bcab7466e6c3)
   - Проблем по обучению не выявлено,переобучения нет, комбинация разных размеров показала худший результат из всех ,но все равно результат отличный

     ![image](https://github.com/user-attachments/assets/647c39b0-319a-46b3-9ec8-e840ffaca3d7)

     2.2 Влияние глубины CNN (15 баллов)
 Исследуйте влияние глубины CNN:
 - Неглубокая CNN (2 conv слоя)
 - Средняя CNN (4 conv слоя)
 - Глубокая CNN (6+ conv слоев)
 - CNN с Residual связями
 
 Для каждого варианта:
 - Сравните точность и время обучения
 - Проанализируйте vanishing/exploding gradients
 - Исследуйте эффективность Residual связей
 - Визуализируйте feature maps

   Создали 4 новых класса и обучили
   ![image](https://github.com/user-attachments/assets/4fb5743e-abdc-473c-a8aa-91da00a511ea)
   ![image](https://github.com/user-attachments/assets/18a02035-1a8d-4f2f-9acb-7ff98b9dbf2f)
   Градиенты в норме, все слои участвуют в обучении.Это Значит что CNN С RESIDUAL решают проблему с потухновением градиентов
   ![image](https://github.com/user-attachments/assets/1d506027-d48d-4654-ab6d-ea0634d04e54)
   ![image](https://github.com/user-attachments/assets/a679cec3-7d75-4f0d-8ee2-c513b0a252e8)
    Модель страдает от исчезающих градиентов
   ![image](https://github.com/user-attachments/assets/022762c5-8763-4b0f-a66a-3a57d62ac1cf)
   ![image](https://github.com/user-attachments/assets/37347e93-7e43-4c98-9d08-2cc29be66166)
   Сеть страдает от исчезающих градиентов, что резко снижает её обучаемость.
   





















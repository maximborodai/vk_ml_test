# vk_ml_test

Решение задачи ранжирования поисковой выдачи. Сравнение моделей производилось по метрике NDCG. Также решение с итоговой моеделью было завернуто в docker.

## Запуск
- Install and run Docker
- Build Docker image using `docker build . -t ml_server`
- Run Docker container using `docker run --rm -it -p 80:80 ml_server`
- Go to `http://127.0.0.1:80/docs` to see all available methods of the API

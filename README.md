# Fasial Analiz

**Система автоматизации учёта посещаемости на основе OpenCV и Flask.** Распознаёт лица в реальном времени, сохраняет записи в ежедневный CSV‑журнал и позволяет управлять пользователями через веб‑интерфейс.

## Технологии

* Python 3.7+
* Flask
* OpenCV (Haar Cascade)
* scikit-learn (KNeighborsClassifier)
* pandas, numpy, joblib
* HTML, CSS, Bootstrap 5

## Установка и запуск

```bash
git clone <URL>
cd <папка>
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python server.py
```

Откройте в браузере [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Работа с приложением

* **/** — главная страница, просмотр журнала и кнопка «Принять участие».
* **/add** — форма для добавления нового пользователя (захват 10 снимков, переобучение модели).
* **/listusers** — список всех пользователей с возможностью удаления.
* **/deleteuser?user=<имя\_ID>** — удаление выбранного пользователя и обновление модели.

## Структура проекта

```
project_root/
├── Attendance/Attendance-<MM_DD_YY>.csv
├── model/haarcascade_frontalface_default.xml
├── static/
│   ├── css/style.css
│   ├── faces/<имя_ID>/…
│   └── face_recognition_model.pkl
├── templates/home.html
├── app.py        # логика детекции и работы с CSV
├── server.py     # Flask‑маршруты
├── requirements.txt
└── README.md
```

## Обновление модели

При добавлении или удалении пользователей модель автоматически переобучается на снимках в `static/faces`.

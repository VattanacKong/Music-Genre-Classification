# Music Genre Classification

  These Wav2Vec2 models was finetuned on [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) which contains 10 classes such as:
  <ul>
    <li>blues</li>
    <li>classical</li>
    <li>country</li>
    <li>disco</li>
    <li>hiphop</li>
    <li>jazz</li>
    <li>metal</li>
    <li>pop</li>
    <li>reggae</li>
    <li>rock</li>
  </ul>

## Download Models

> You can get the finetuned models [here.](https://drive.google.com/drive/folders/1tbu7Hlxu4Xom1wnI0wkrB1hqntKIXdJM?usp=sharing)

## Project Structure

The following is the structure of the repository:

```
MUSIC GENRE CLASSIFICATION/
├── Inference Models/
│   ├── model-5secs-checkpoint-4100/
│   ├── model-10secs-checkpoint-5900/
│   ├── model-20secs-checkpoint-5400/
│   └── lstm_model.pth
├── .gitignore
├── app.py
├── requirements.txt
└── Training.ipynb
```

## Set Up

To test the app, follow these steps:

1. Create a new virtual environment.
   
2. Navigate to the project directory:
   ```
    cd "Music Genre Classification"
   ```

3. Install the required dependencies:
   ```
     pip install -r requirements.txt
   ```

4. Run the streamlit app:
   ```
     streamlit run app.py
   ```


   


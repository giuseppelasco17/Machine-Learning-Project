# Machine Learning 2020/2021 

Marco Marcucci, Giuseppe Lasco

## Struttura del codice

Il codice si compone dei seguenti moduli:

 - `training.py` che contiene l'insieme delle funzioni che permettono di effettuare pre-processamento dei dati e 
 addestramento dei modelli;
 - `tuning.py` che permette il tuning degli iper-parametri attraverso la libreria `keras_tuner`;
 - `eval_test.py` che permette di valutare il miglior modello sul testing-set;
 - `models` che contiene l'insieme delle implementazioni dei modelli testati.



## Manuale d'uso

Per valutare la metrica `Accuracy` su un testing set eseguire i seguenti passaggi:

**Nota bene:**

 - **Utilizzare python 3.8 e tensorflow 2.4.1 per eseguire `training.py` e `tuning.py`. 
Esistono dei conflitti tra `keras` e l'ultima versione di `tensorflow`**.
 - **Avviare il progetto da Windows, per evitare problemi legati ai path dei file**

1. Installare le seguenti librerie per il corretto funzionamento dei moduli:
    
    - scikit-learn 0.24.1
    - pandas 1.2.3
    - tensorflow 2.4.1
    - numpy 1.19.5
    - keras 2.4.3
    - keras_tuner 1.0.3
    - matplotlib 3.3.4
   
2. Inserire i percorso dei testing_sets (compresa l’estensione `.csv`) nel file `config.properties`;
3. Avviare il `main()` del modulo `eval_test`;
4. Il risultato verrà stampato a schermo.
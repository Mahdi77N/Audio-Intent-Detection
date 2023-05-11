# Audio Intent Detection Using CNN-LSTM Neural Networks
This report presents a study in audio intent detection, with a focus on recognizing the intent behind speech in audio recordings. The dataset consisted of approximately 10,000 audio recordings, each with accompanying information such as the speaker's age, gender, and native language. The audio files were then transformed into Mel-Frequency Cepstral Coefficients (MFCCs) and used as input for our CNN and LSTM base models. The results showed that the proposed models significantly outperformed a naive baseline, providing a satisfying solution for the problem of audio intent detection. For more information of the pipline including trimming, converting to MFCC, the model, the dataset and the resualts, read the pdf report file.




## Prerequisites
To run this project, you will need to have the following packages installed:

- Keras
- TensorFlow
- Matplotlib
- NumPy
- Pandas
- Scikit-learn
- Librosa




## Usage

1. Install the required packages.
2. Provide the path to the CSV file with relevant information to the read_data() function.
3. Provide the test data to the test_output_csv() function, and it will generate the output for you.



## Results

The CNN model has a higher F1-Score of 96.6% compared to the CNN-LSTM model with an F1-Score of 96.2%. Despite the slight difference in scores, the CNN-LSTM model still performed well. Our use of the latest and advanced techniques in the Attention-based CNN-LSTM model was successful, leading to the best results among the three models with an F1-Score of 97.3%.

## License

This project is under the MIT license.


## Contact

If you have any questions or feedback, please send an email to m1998naderi@gmail.com.

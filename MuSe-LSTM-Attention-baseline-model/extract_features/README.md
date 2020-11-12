## **Extract features:**
Extract BERT features:
python extract_text_embedding.py

Extract VGGish features:
python extract_vggish_embedding.py


##### Virtual environments:
- For the extraction of the vggish features: \
`pip install numpy pandas opencv-python tensorflow-gpu==1.15.0 keras==2.2.4 resampy`
- For the extraction of the vgg2 features (visual) Keras is required: \
`pip install numpy pandas opencv-python tensorflow-gpu==1.13.1 keras==2.2.4 dlib keras-vggface`
- For the other features PyTorch is required: \
`pip install numpy pandas resampy soundfile opencv-python torch transformers`
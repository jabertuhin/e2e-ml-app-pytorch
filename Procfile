pytorch: pip install http://download.pytorch.org/whl/cpu/torch-1.4.0%2Bcpu-cp36-cp36m-linux_x86_64.whl
train: python text_classification/train.py --data-url https://raw.githubusercontent.com/madewithml/lessons/master/data/news.csv --lower --shuffle --use-glove
web: sh setup.sh && streamlit run text_classification/streamlit.py
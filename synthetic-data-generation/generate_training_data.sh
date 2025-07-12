wget https://zenodo.org/records/6802614/files/Totalsegmentator_dataset.zip
unzip Totalsegmentator_dataset.zip
python step0_preprocess_totalsegmentator.py
python step1_generate_labels.py
python step2_generate_views.py

import eda.tidy_dataset as tds
tidy = tds.PreProcessor("cup98LRN.txt")

learning = tidy.get_tidy_data()

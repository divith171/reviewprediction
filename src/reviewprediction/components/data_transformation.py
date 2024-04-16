import os
from reviewprediction import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
import numpy as np
import spacy
import regex as re
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import  StandardScaler
import json
from reviewprediction.entity.config_entity import DataTransformationConfig
import pickle
import joblib

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
   
    def complete_transformation(self):
            final_dataframe = pd.read_csv(self.config.data_path)
            labels = final_dataframe['review_score']
            final_dataframe.drop('review_score', axis=1, inplace=True)
            X_train, X_test, y_train, y_test = train_test_split(final_dataframe, labels, stratify=labels, test_size=0.2, random_state=0)
            def train_response(frame):
                    f1 = frame[frame.iloc[:,1] == 0]
                    f2 = frame[frame.iloc[:,1] == 1]
                    global dict_frame, dict_f1, dict_f2
                    
                    dict_frame = dict(frame.iloc[:,0].value_counts())
                    
                    dict_f1 = dict(f1.iloc[:,0].value_counts())
                    
                    dict_f2 = dict(f2.iloc[:,0].value_counts())

                     # Save the dictionaries as pickle files
                    pickle_path='artifacts'
                    if os.path.exists(pickle_path + '/dict_frame_product_category.pkl'):
                        with open(pickle_path + '/dict_frame_payment_sequential.pkl', 'wb') as f:
                            pickle.dump(dict_frame, f)
                    else:
                         with open(pickle_path + '/dict_frame_product_category.pkl', 'wb') as f:
                            pickle.dump(dict_frame, f)

                    if os.path.exists(pickle_path + '/dict_f1_product_category.pkl'):
                         with open(pickle_path + '/dict_f1_payment_sequential.pkl', 'wb') as f:
                           pickle.dump(dict_f1, f)   
                    
                    else:   
                         with open(pickle_path + '/dict_f1_product_category.pkl', 'wb') as f:
                           pickle.dump(dict_f1, f)
                    
                    if os.path.exists(pickle_path + '/dict_f2_product_category.pkl'):
                         with open(pickle_path + '/dict_f2_payment_sequential.pkl', 'wb') as f:
                           pickle.dump(dict_f2, f)   
                    
                    else:   
                         with open(pickle_path + '/dict_f2_product_category.pkl', 'wb') as f:
                           pickle.dump(dict_f2, f)
                    
                    
                    state_0, state_1 = [],[],
                    for i in range(len(frame)):
                        if frame.iloc[:,1][i] == 0:
                                state_0.append(dict_f1.get(frame.iloc[:,0][i],0)/dict_frame[frame.iloc[:,0][i]])
                                state_1.append(float(1-state_0[-1]))
                        else:
                                state_1.append(dict_f2.get(frame.iloc[:,0][i],0)/dict_frame[frame.iloc[:,0][i]])
                                state_0.append(float(1-state_1[-1])) 
                    df3 = pd.DataFrame({'State_0':state_0, 'State_1':state_1})
                    return df3.to_numpy()

            def test_response(test):
                    t_state_0, t_state_1 = [],[]
                    for i in range(len(test)):
                        if dict_frame.get(test[i]):
                            t_state_0.append(dict_f1.get(test[i],0)/dict_frame.get(test[i]))
                            t_state_1.append(dict_f2.get(test[i],0)/dict_frame.get(test[i]))
                        else:
                            t_state_0.append(0.5)
                            t_state_1.append(0.5)
                    df4 = pd.DataFrame({'State_0':t_state_0, 'State_1':t_state_1})
                    return df4.to_numpy() 
            
            X_train_resp_prod_cat = train_response(pd.concat([X_train['product_category_name'], y_train], axis=1).reset_index(drop=True))
            X_test_resp_prod_cat = test_response(X_test['product_category_name'].values)


            ohe_order_item = OneHotEncoder()
            ohe_order_item.fit(X_train['order_item_id'].values.reshape(-1,1))
            joblib.dump(ohe_order_item, 'artifacts/ohe_order_item.pkl')
            X_train_order_item = ohe_order_item.transform(X_train['order_item_id'].values.reshape(-1,1)).toarray()
            X_test_order_item = ohe_order_item.transform(X_test['order_item_id'].values.reshape(-1,1)).toarray()
            X_train_resp_payment_seq = train_response(pd.concat([X_train['payment_sequential'], y_train], axis=1).reset_index(drop=True))
            X_test_resp_payment_seq = test_response(X_test['payment_sequential'].values)

            ohe_payment_type = OneHotEncoder()
            ohe_payment_type.fit(X_train['payment_type'].values.reshape(-1,1))
            joblib.dump(ohe_payment_type, 'artifacts/ohe_payment_type.pkl')
            X_train_payment_type = ohe_payment_type.transform(X_train['payment_type'].values.reshape(-1,1)).toarray()
            X_test_payment_type = ohe_payment_type.transform(X_test['payment_type'].values.reshape(-1,1)).toarray()

            enc_price = OrdinalEncoder()
            enc_price.fit(X_train['price_category'].values.reshape(-1,1))
            joblib.dump(enc_price, 'artifacts/enc_price.pkl')
            enc_price.categories_ = [np.array([ 'cheap', 'affordable', 'expensive'], dtype=object)]
            X_train_cat_price = enc_price.transform(X_train['price_category'].values.reshape(-1,1))
            X_test_cat_price = enc_price.transform(X_test['price_category'].values.reshape(-1,1))


            sp = spacy.load('pt_core_news_sm')
            all_stopwords = sp.Defaults.stop_words

            def process_texts(texts): 

                processed_text = []
                dates = '^([0]?[1-9]|[1|2][0-9]|[3][0|1])[./-]([0]?[1-9]|[1][0-2])[./-]([0-9]{4}|[0-9]{2})$'
                
                for text in texts:
                    text = re.sub(r'\r\n|\r|\n', ' ', text) 
                    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', text) 
                    text = re.sub(dates, ' ', text) 
                    text = re.sub('[ \t]+$', '', text)
                    text = re.sub('\W', ' ', text)
                    text = re.sub('[0-9]+', ' ', text)
                    text = re.sub('\s+', ' ', text)
                    text = ' '.join(e for e in text.split() if e.lower() not in all_stopwords) 
                    processed_text.append(text.lower().strip())
                    
                return processed_text


            X_train_comment_preprocess = process_texts(X_train['review_comment_message'])
            X_test_comment_preprocess = process_texts(X_test['review_comment_message'])


            model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            model = SentenceTransformer(model_name)
            encoded_train_reviews = model.encode(X_train_comment_preprocess)
            extracted_reviews = [review[0] for review in encoded_train_reviews]
            X_train = X_train.assign(embedded_review_comment_message=extracted_reviews)

            encoded_test_reviews=model.encode(X_test_comment_preprocess)
            extracted_test_reviews = [review[0] for review in encoded_test_reviews]
            X_test = X_test.assign(embedded_review_comment_message=extracted_test_reviews)

            
            strn = StandardScaler()
            strn.fit(X_train[['price','freight_value','product_photos_qty','product_weight_g', 'product_length_cm',
                'product_height_cm', 'product_width_cm', 'payment_value','purchase-delivery difference','estimated-actual delivery difference','purchase_delivery_diff_per_price']])
            os.makedirs('artifacts', exist_ok=True)
            joblib.dump(strn, 'artifacts/strn.pkl')
            
            X_train_strn = strn.transform(X_train[['price','freight_value','product_photos_qty','product_weight_g', 'product_length_cm',
                'product_height_cm', 'product_width_cm', 'payment_value','purchase-delivery difference','estimated-actual delivery difference','purchase_delivery_diff_per_price']])
            X_test_strn = strn.transform(X_test[['price','freight_value','product_photos_qty','product_weight_g', 'product_length_cm',
                'product_height_cm', 'product_width_cm', 'payment_value','purchase-delivery difference','estimated-actual delivery difference','purchase_delivery_diff_per_price']])


            X_train_final = np.concatenate((X_train_strn,X_train_resp_prod_cat,X_train_order_item,
                X_train_resp_payment_seq,X_train_payment_type,X_train_cat_price,X_train['review_availability'].values.reshape(-1,1),
                np.vstack(X_train['embedded_review_comment_message'].values)), axis=1)

            X_test_final = np.concatenate((X_test_strn,X_test_resp_prod_cat, X_test_order_item,
                X_test_resp_payment_seq,X_test_payment_type,X_test_cat_price,X_test['review_availability'].values.reshape(-1,1),
                np.vstack(X_test['embedded_review_comment_message'].values)), axis=1)



            # Load the root directory from your config
            root_dir = self.config.root_dir

            # Construct file paths using the root directory
            train_data_path = os.path.join(root_dir, "train_data.json")
            test_data_path = os.path.join(root_dir, "test_data.json")

            # Combine features and label
            train_data = []
            for features, label in zip(X_train_final, y_train):
                train_data.append({'features': features.tolist(), 'label': label})

            test_data = []
            for features, label in zip(X_test_final, y_test):
                test_data.append({'features': features.tolist(), 'label': label})

            # Save to JSON files
            with open(train_data_path, "w") as f:
                json.dump(train_data, f)

            with open(test_data_path, "w") as f:
                json.dump(test_data, f)

            logger.info("Splited data into training and test sets")
            

            # Print the number of samples and features 
            num_train_samples = len(train_data)
            num_features = len(train_data[0]['features'])  #
            print(f"Train data (estimated equivalent of shape): ({num_train_samples}, {num_features})")

            num_test_samples = len(test_data)
            num_features = len(test_data[0]['features'])  #
            print(f"Test data (estimated equivalent of shape): ({num_test_samples}, {num_features})")
        
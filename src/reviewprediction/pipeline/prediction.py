import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import os 
import sys
from reviewprediction import logger
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import spacy
import regex as re
from sentence_transformers import SentenceTransformer


class PredictionPipeline:
    def __init__(self):
        path = 'artifacts\model_trainer\model.joblib'
        if os.path.exists(path):
         self.model = joblib.load(path)
        else:
         raise FileNotFoundError(f"Model file not found at: {path}")
    
    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction
    
class CustomData():
    def __init__(self,
                    order_item_id: int,
                        price: float,
                        freight_value:  float,
                        product_category_name: object ,
                        product_name_length: float,  
                        product_description_length: float,
                        product_photos_qty: float,
                        product_weight_g: float,
                        product_length_cm: float,
                        product_height_cm: float,
                        product_width_cm: float,
                        seller_zip_code_prefix: int,  
                        seller_city: object ,
                        seller_state: object ,
                        order_status: object ,
                        customer_zip_code_prefix: int,
                        customer_city: object ,
                        customer_state: object, 
                        review_id : object ,
                        review_comment_title: object ,
                        review_comment_message: object ,
                        review_creation_date: object ,
                        review_answer_timestamp: object ,
                        payment_sequential: int , 
                        payment_type: object ,
                        payment_installments: int, 
                        payment_value: float,
                        purchase_delivery_difference: int  ,
                        estimated_actual_delivery_difference: int  ,
                        price_category: object ,
                        purchase_delivery_diff_per_price: float,
                        review_availability: int ) :
                
            
            self.order_item_id= order_item_id
            self.price=price
            self.freight_value=freight_value
            self.product_category_name=product_category_name
            self.product_name_length=product_name_length
            self.product_description_lenght=product_description_length
            self.product_photos_qty= product_photos_qty
            self.product_weight_g= product_weight_g
            self.product_length_cm=product_length_cm
            self.product_height_cm =product_height_cm
            self.product_width_cm=product_width_cm
            self.seller_zip_code_prefix = seller_zip_code_prefix 
            self.seller_city = seller_city
            self.seller_state=seller_state
            self.order_status=order_status
            self.customer_zip_code_prefix=customer_zip_code_prefix  
            self.customer_city=customer_city
            self.customer_state=customer_state
            self.review_id = review_id
            self.review_comment_title = review_comment_title 
            self.review_comment_message=review_comment_message
            self.review_creation_date=review_creation_date
            self.review_answer_timestamp = review_answer_timestamp
            self.payment_sequential =payment_sequential
            self.payment_type = payment_type
            self.payment_installments =payment_installments  
            self.payment_value = payment_value
            self.purchase_delivery_difference= purchase_delivery_difference  
            self.estimated_actual_delivery_difference= estimated_actual_delivery_difference
            self.price_category =price_category
            self.purchase_delivery_diff_per_price =purchase_delivery_diff_per_price
            self.review_availability=review_availability

    def data_transformer(self):
            
            try:
            
                data ={   
                    'order_item_id': [self.order_item_id]  ,
                    'price': [self.price],
                    'freight_value':  [self.freight_value],
                    'product_category_name': [self.product_category_name],
                    'product_name_length':[self.product_name_length],
                    'product_description_lenght': [self.product_description_lenght],
                    'product_photos_qty': [self.product_photos_qty],
                    'product_weight_g': [self.product_weight_g],
                    'product_length_cm': [self.product_length_cm],
                    'product_height_cm': [self.product_height_cm],
                    'product_width_cm': [self.product_width_cm],
                    'seller_zip_code_prefix': [self.seller_zip_code_prefix],
                    'seller_city': [self.seller_state],
                    'seller_state': [self.seller_state],
                    'order_status': [self.order_status],
                    'customer_zip_code_prefix': [self.customer_zip_code_prefix],
                    'customer_city': [self.customer_city],
                    'customer_state': [self.customer_city] ,
                    'review_id' : [self.review_id],
                    'review_comment_title': [self.review_comment_title],
                    'review_comment_message': [self.review_comment_message],
                    'review_creation_date': [self.review_creation_date],
                    'review_answer_timestamp': [self.review_answer_timestamp],
                    'payment_sequential': [self.payment_sequential],
                    'payment_type': [self.payment_type],
                    'payment_installments': [self.payment_installments],
                    'payment_value': [self.payment_value],
                    'purchase-delivery difference': [self.purchase_delivery_difference] ,
                    'estimated-actual delivery difference': [self.estimated_actual_delivery_difference] ,
                    'price_category': [self.price_category],
                    'purchase_delivery_diff_per_price': [self.purchase_delivery_diff_per_price],
                    'review_availability': [self.review_availability]
                }
                df=pd.DataFrame(data)
                print("Form data before transformation:", df)
                def test_response(test,dict_frame,dict_f1,dict_f2):
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
                    

                prod_cat_dict_frame = pickle.load(open('artifacts/dict_frame_product_category.pkl','rb'))
                prod_cat_dict_f1 = pickle.load(open('artifacts/dict_f1_product_category.pkl','rb'))
                prod_cat_dict_f2 = pickle.load(open('artifacts/dict_f2_product_category.pkl','rb'))

                pay_seq_dict_frame = pickle.load(open('artifacts/dict_frame_payment_sequential.pkl','rb'))
                pay_seq_dict_f1 = pickle.load(open('artifacts/dict_f1_payment_sequential.pkl','rb'))
                pay_seq_dict_f2 = pickle.load(open('artifacts/dict_f2_payment_sequential.pkl','rb'))
                
                strn = joblib.load('artifacts/strn.pkl')
                '''X_test_strn = strn.transform(df.loc[:, ['price', 'freight_value', 'product_photos_qty', 'product_weight_g',
                                                'product_length_cm', 'product_height_cm', 'product_width_cm',
                                                'payment_value', 'purchase-delivery difference',
                                                'estimated-actual delivery difference', 'purchase_delivery_diff_per_price']].T)'''
                

                numeric_columns = ['price', 'freight_value', 'product_photos_qty', 'product_weight_g',
                   'product_length_cm', 'product_height_cm', 'product_width_cm',
                   'payment_value', 'purchase-delivery difference','estimated-actual delivery difference','purchase_delivery_diff_per_price']

                # Select the numeric columns
                X_test_strn = df[numeric_columns]

                # Transform the selected columns using the StandardScaler
                X_test_strn_scaled = strn.transform(X_test_strn)





                X_test_resp_prod_cat = test_response(df['product_category_name'].values,prod_cat_dict_frame,prod_cat_dict_f1,prod_cat_dict_f2)
                ohe_order_item = joblib.load('artifacts/ohe_order_item.pkl')
                X_test_order_item = ohe_order_item.transform(df['order_item_id'].values.astype(int).reshape(-1,1)).toarray()
                X_test_resp_payment_seq = test_response(df['payment_sequential'].values,pay_seq_dict_frame,pay_seq_dict_f1,pay_seq_dict_f2)
                ohe_payment_type = joblib.load('artifacts/ohe_payment_type.pkl')
                X_test_payment_type = ohe_payment_type.transform(df['payment_type'].values.reshape(-1,1)).toarray()
                enc_price = joblib.load('artifacts/enc_price.pkl')
                enc_price.categories_ = [np.array([ 'cheap', 'affordable', 'expensive'], dtype=object)]
                X_test_cat_price = enc_price.transform(df['price_category'].values.reshape(-1,1))

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


                X_test_comment_preprocess = process_texts(df['review_comment_message'])
                model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                model = SentenceTransformer(model_name)
                encoded_test_reviews=model.encode(X_test_comment_preprocess)
                extracted_test_reviews = [review[0] for review in encoded_test_reviews]
                df = df.assign(embedded_review_comment_message=extracted_test_reviews)

                df['review_availability'] = 1 if df['review_comment_message'].values[0] != 'indisponível' else 0
                X_test_final = np.concatenate((X_test_strn_scaled,X_test_resp_prod_cat, X_test_order_item,
                X_test_resp_payment_seq,X_test_payment_type,X_test_cat_price,df['review_availability'].values.reshape(-1,1),
                np.vstack(df['embedded_review_comment_message'].values)), axis=1)


                return X_test_final
            except Exception as e:
             print("Error occurred during data transformation:",e)
       
